# Reaches around 0.7930 test accuracy.

import argparse
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit2, Yelp, Flickr

# new imports
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil
import os
import csv
import subprocess

# replace node_index with node_mask
def get_split_idx(train_mask, val_mask, test_mask):
    train_idx = torch.where(train_mask)[0]
    val_idx = torch.where(val_mask)[0]
    test_idx = torch.where(test_mask)[0]
    return {'train': train_idx, 'valid': val_idx, 'test': test_idx}

def get_mask(comp_core):
    mask = 0
    for core in comp_core:
        mask += 2 ** core
    return hex(mask)

def is_cpu_proc(num_cpu_proc, rank=None):
    if rank is None:
        rank = dist.get_rank()
    return rank < num_cpu_proc

def assign_cores(num_cpu_proc,n_samp,n_train):
    assert is_cpu_proc(num_cpu_proc), "For CPU Comp process only"
    rank = dist.get_rank()
    load_core, comp_core = [], []
    n = psutil.cpu_count(logical=False)
    # n = 72
    size = num_cpu_proc
    num_of_samplers = n_samp
    load_core = list(range(n//size*rank,n//size*rank+num_of_samplers))
    comp_core = list(range(n//size*rank+num_of_samplers,n//size*rank+num_of_samplers+n_train))

    return load_core, comp_core
# GraphSAGE model
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

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

    def inference(self, x_all):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id].to(device)
                edge_index = batch.edge_index.to(device)
                x = self.convs[i](x, edge_index)
                x = x[:batch.batch_size]
                if i != self.num_layers - 1:
                    x = x.relu()
                xs.append(x.cpu())

                pbar.update(batch.batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all


def run(rank, times, model, process_num):
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '29518'
    dist.init_process_group('gloo', rank=rank, world_size=process_num)  
    device = torch.device('cpu' if is_cpu_proc(process_num, rank) 
                          else 'cuda')
    if not is_cpu_proc(process_num):
        torch.cuda.set_device(device)
    train_idx = split_idx['train'].to(device)

    for run in range(times):  
        model.reset_parameters()
        model = DistributedDataParallel(model)
        train(model, rank, train_idx)
    # dist.barrier()
    # dist.destroy_process_group()

        
def train(model, rank, train_idx):
    load_core, comp_core = assign_cores(process_num, load_core_num, compute_core_num)

    # set compute cores 【taskset】
    torch.set_num_threads(len(comp_core))
    pid = os.getpid()
    # print("[TASKSET] rank {}, pid: {}".format(rank, pid))
    core_mask = get_mask(comp_core)
    subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])
    # print("[TASKSET] rank {}, using compute core: {}".format(rank, comp_core))


    
    train_sampler = DistributedSampler(
        train_idx,
        num_replicas=process_num,
        rank=rank
    )
    train_loader = NeighborLoader(
        data,
        input_nodes=split_idx['train'],
        num_neighbors=[15, 10, 5],
        batch_size=4096//process_num,
        num_workers=len(load_core),
        persistent_workers=True, # using memory cache to speed up
        sampler=train_sampler
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(1):
        train_sampler.set_epoch(epoch)
        start_time = time.time()
        model.train()
        total_loss = total_correct = total_cnt =  0
        with train_loader.enable_cpu_affinity(loader_cores = load_core):
            # print("[LOADER] rank {}: loading data using core {} ".format(rank,load_core))
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
                y = batch.y[:batch.batch_size].squeeze()
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss)   
                # yelp is multi-label classification
                if (args.dataset == 'yelp'):
                    # True: larger than 0.5, False: less than 0.5
                    out = out > 0.5
                    total_correct += int(out.eq(y).sum())
                    total_cnt += batch.batch_size * dataset.num_classes
                else:
                    total_correct += int(out.argmax(dim=-1).eq(y).sum())
                    total_cnt += batch.batch_size

        end_time = time.time()
        with open('./DDP_profile/DDP_time.txt', 'a') as f:
            f.write(f'Rank {rank} Epoch {epoch} Train_time: {end_time - start_time}\n')

        loss = total_loss / len(train_loader)
        approx_acc = total_correct / total_cnt
        # print(f'Rank {rank}|Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}')
        # print("total_time: ", end_time - start_time)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_process",
        default= "1",
    )

    parser.add_argument(
        "--n_sampler",
        default= "2",
        help="loader core number"
    )

    parser.add_argument(
        "--n_trainer",
        default= "8",
        help="trainer core number"
    )

    parser.add_argument(
        "--run_times",
        default= "1",
        help="repeat times to run the experiment",
    )

    parser.add_argument(
        "--dataset",
        default= "ogbn-products",
        help="dataset name",
        choices=["ogbn-products", "flickr", "yelp", "reddit"],
    )


    args = parser.parse_args()
    args.mode = "cpu"
    # print(f"Training in {args.mode} mode.")
    times = int(args.run_times)

    process_num = int(args.cpu_process)
    load_core_num = int(args.n_sampler)
    compute_core_num = int(args.n_trainer)

    # max core = 32 do assertion
    assert process_num * (load_core_num + compute_core_num) <= 32, "exceed max core number"

    device = torch.device('cpu')
    print("device: ", device, "process_num: ", process_num, "load_core_num: ", load_core_num)
    # device = torch.device('cpu')
    if args.dataset == "ogbn-products":
        dataset = PygNodePropPredDataset(name = 'ogbn-products')
        split_idx = dataset.get_idx_split()
        # evaluator = Evaluator(name='ogbn-products')
    elif args.dataset == "flickr":
        dataset = Flickr(root='./dataset/flickr')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask) 
    elif args.dataset == "yelp":
        dataset = Yelp(root='./dataset/yelp')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask)
    elif args.dataset == "reddit":
        dataset = Reddit2(root='./dataset/reddit')
        split_idx = get_split_idx(dataset[0].train_mask, dataset[0].val_mask, dataset[0].test_mask)

    data = dataset[0].to(device, 'x', 'y')

    subgraph_loader = NeighborLoader(
        data,
        input_nodes=None,
        num_neighbors=[-1],
        batch_size=4096,
        num_workers=4,
        persistent_workers=True,
    )

    model = SAGE(dataset.num_features, 128, dataset.num_classes, num_layers=3)
    model = model.to(device)
   
    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29506'

    processes = []
    try:
        mp.set_start_method('fork')
        print("set start method to fork")
    except RuntimeError:
        pass
   
    tik = time.time()
    for rank in range(process_num):
        p = mp.Process(target=run, args=(rank, times, model, process_num))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    t = time.time() - tik

    meta_data = []
    meta_data.append(t)
    meta_data.append(args.cpu_process)
    meta_data.append(args.n_sampler)
    meta_data.append(args.n_trainer)

    with open('PyG/grid_serach_{}.csv'.format(args.dataset), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(meta_data)







