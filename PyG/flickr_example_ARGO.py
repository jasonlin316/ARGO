import argparse
import torch
import torch.nn.functional as F

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.datasets import Flickr

# TODO: 1. include necessary packages
import os
import subprocess
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from argo import ARGO


# TODO: 2. Add `get_mask` function for `taskset` core binding
# replace node_index with node_mask
def get_mask(comp_core):
    mask = 0
    for core in comp_core:
        mask += 2 ** core
    return hex(mask)

# GNN model
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

# Training function
# TODO: 5. Modify the input parameters of the training function
def train(args, device, data, rank, world_size, comp_core, load_core, counter, b_size, ep): 
    # TODO: 3.1. Setup PyTorch Distributed Data Parallel (DDP)
    dist.init_process_group('gloo', rank=rank, world_size=world_size) # newly added
    model = GNN(dataset.num_features, 128, dataset.num_classes, num_layers=3, model_name=model_name)
    model = DistributedDataParallel(model) # newly added
    model = model.to(device)
    train_idx = split_idx['train'].to(device)
    model.train()
    # TODO: 6. modify the dataloader and add DistributedSampler
    # Add DistributedSampler for multi-thread data loading
    train_sampler = DistributedSampler(
            train_idx,
            num_replicas = world_size,
            rank=rank
        ) 
    train_loader = NeighborLoader(
        data,
        input_nodes = train_idx,
        num_neighbors=[15, 10, 5],
        batch_size=b_size//world_size, # modified
        num_workers=len(load_core), # modified
        persistent_workers=True,
        sampler = train_sampler # newly added
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
     
    #TODO: 8. Use `taskset` to bind trainer-core
    torch.set_num_threads(len(comp_core))
    pid = os.getpid()
    core_mask = get_mask(comp_core)
    subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])

    # TODO: 10. Load the model before training and save it afterward
    PATH = "PyG/model.pt"
    if counter[0] != 0:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    #TODO: 7. Change the number of epochs
    for epoch in range(ep): # change num_epochs to ep
        total_loss = total_correct = total_cnt =  0
        
        # TODO: 9. Set loader cores affinity
        with train_loader.enable_cpu_affinity(loader_cores = load_core): # set loader cores
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)[:batch.batch_size]
                y = batch.y[:batch.batch_size].squeeze().long()
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)   
                total_correct += int(out.argmax(dim=-1).eq(y).sum())
                total_cnt += batch.batch_size
    

        loss = total_loss / len(train_loader)
        approx_acc = total_correct / total_cnt
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Accuracy: {approx_acc:.4f}')
    dist.barrier()
    if rank == 0:
        torch.save({'epoch': counter[0],
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }, PATH)
        
# Functions for data preprocessing
def get_split_idx(train_mask, val_mask, test_mask):
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]
    return {'train': train_idx, 'valid': val_idx, 'test': test_idx}     

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("GNN training")
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--batch-size", type=int, default=1000)
    args = argparser.parse_args()
    model_name = "sage"
    sampler = "neighbor"
    device = torch.device('cpu')
    dataset = Flickr(root='dataset')
    split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask) 
    data = dataset[0].to(device, 'x', 'y')
    
    # TODO: 3.2.Setup the mp environment
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    mp.set_start_method('fork', force=True)

    # TODO: 4.Enable ARGO by initializing the runtime system, and wrapping the training function
    runtime = ARGO(n_search = 15, epoch = args.num_epochs, batch_size = args.batch_size) #initialization
    runtime.run(train, args=(args, device, data)) # wrap the training function







