import argparse
import torch
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader, ShaDowKHopSampler
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import Reddit2, Flickr

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import subprocess
import random
from argo import ARGO

def get_split_idx(train_mask, val_mask, test_mask):
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]
    return {'train': train_idx, 'valid': val_idx, 'test': test_idx}

# replace node_index with node_mask
def get_mask(comp_core):
    mask = 0
    for core in comp_core:
        mask += 2 ** core
    return hex(mask)


# GraphSAGE & GCN  model
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, model_name):
        super().__init__()

        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        if model_name == "sage":
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        elif model_name == "gcn":
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            raise NotImplementedError

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def _train(**kwargs):
    total_loss = 0
    loader = kwargs['loader']
    model = kwargs['model']
    sampler = kwargs['sampler']
    opt = kwargs['opt']
    load_core = kwargs['load_core']
    comp_core = kwargs['comp_core']
    epoch = kwargs['epoch']
   
    device = torch.device("cpu")
    
    # set compute cores 【taskset】
    torch.set_num_threads(len(comp_core))
    pid = os.getpid()
    core_mask = get_mask(comp_core)
    subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])

    if sampler == "neighbor":
        # set loader cores
        with loader.enable_cpu_affinity(loader_cores = load_core):
            for batch in loader:
                opt.zero_grad()
                out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
                y = batch.y[:batch.batch_size].squeeze().long()
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
                total_loss += loss.item()
                del batch   

    elif sampler == "shadow":
        for batch in loader:
            batchsize = len(batch.y)
            opt.zero_grad()
            out = model(batch.x, batch.edge_index.to(device))[:batchsize]
            y = batch.y[:batchsize].squeeze().long()
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            del batch
            total_loss += loss.item()

    return total_loss


def train(args, dataset, rank, world_size, comp_core, load_core, counter, b_size, ep):
    dist.init_process_group('gloo', rank=rank, world_size=world_size)  
    device = torch.device('cpu')
    data = dataset[0].to(device, 'x', 'y')
    train_idx = split_idx['train'].to(device)
    hidden = args.hidden
    num_layers = args.layer
    model_name = args.model
    sampler = args.sampler
    in_size = dataset.num_features
    num_classes = dataset.num_classes
    # create GNN model
    model = GNN(in_size, hidden, num_classes, num_layers, model_name).to(device)
    model = DistributedDataParallel(model)
    num_of_samplers = len(load_core)
    # create loader
    if args.sampler.lower() == "neighbor":
        train_sampler = DistributedSampler(
            train_idx,
            num_replicas = world_size,
            rank=rank
        ) 
        train_loader = NeighborLoader(
            data,
            input_nodes=split_idx['train'],
            num_neighbors=[15, 10, 5],
            batch_size=b_size//world_size,
            num_workers=num_of_samplers,
            persistent_workers=True, # using memory cache to speed up
            sampler=train_sampler
        )
        # TODO: assert len(sampler.fanouts) == args.layer
    elif args.sampler.lower() == "shadow":
        train_loader = ShaDowKHopSampler(
            data,
            node_idx=split_idx['train'],
            depth=2,
            num_neighbors=5,
            batch_size=b_size//world_size,
            num_workers=num_of_samplers,
        )
    else:
        raise NotImplementedError
        
    # training loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    params = {
        # training
        'loader': train_loader,
        'model': model,
        'sampler': sampler,
        'opt': opt,
        # logging
        'rank': rank,
        'train_size': len(train_idx),
        'batch_size': b_size,
        'device': device,
        'process': world_size
    }

    PATH = "./PyG/model.pt"
    if counter[0] != 0:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(ep):
        if args.sampler.lower() == "neighbor":
            train_sampler.set_epoch(epoch)
        params['epoch'] = epoch
        model.train()
        params['load_core'] = load_core
        params['comp_core'] = comp_core
        loss = _train(**params)
        if rank == 0:
            print("Loss: ", loss)
    dist.barrier()
    EPOCH = counter[0]
    LOSS = loss
    if rank == 0:
        torch.save({'epoch': EPOCH,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': LOSS}, PATH)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products",  "reddit", "yelp", "flickr"])
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024 * 4)
    parser.add_argument("--layer",
                        type=int,
                        default=3)
    parser.add_argument('--sampler',
                        type=str,
                        default='neighbor',
                        choices=["neighbor", "shadow"])
    parser.add_argument('--model',
                        type=str,
                        default='sage',
                        choices=["sage", "gcn"])
    parser.add_argument("--hidden",
                        type=int,
                        default= 128)
    arguments = parser.parse_args()
    storage_path = "dataset"
    if arguments.dataset == "ogbn-products":
        dataset = PygNodePropPredDataset(name = 'ogbn-products', root = storage_path)
        split_idx = dataset.get_idx_split()
        # evaluator = Evaluator(name='ogbn-products')
    elif arguments.dataset == "ogbn-papers100M":
        dataset = PygNodePropPredDataset(name = 'ogbn-papers100M', root = storage_path)
        split_idx = dataset.get_idx_split()
    elif arguments.dataset == "flickr":
        dataset = Flickr(root=storage_path)
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask) 
    elif arguments.dataset == "reddit":
        dataset = Reddit2(root=storage_path)
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask)
    
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29506'
    mp.set_start_method('fork', force=True)
    runtime = ARGO(n_search = 10, epoch = 20, batch_size = arguments.batch_size)
    runtime.run(train, args=(arguments, dataset))
    print("program finished.")

