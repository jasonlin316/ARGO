import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF

import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy
from multiprocessing import Process, Manager
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil
import os
import csv



comp_core = []
load_core = []
totalN = 0
totalE = 0
T_exe = 0

import subprocess

def run_stream_benchmark():
    try:
        # Compile the STREAM benchmark C code
        subprocess.run("gcc -O3 -march=native -o stream stream.c", shell=True, check=True)

        # Run the STREAM benchmark and capture the output
        result = subprocess.run("./stream", shell=True, capture_output=True, text=True)
        
        # Parse the output to get the memory bandwidth
        output_lines = result.stdout.split("\n")

        # Search for the line containing "Copy Bandwidth" and extract the value
        copy_bandwidth = None
        for line in output_lines:
            if "Triad" in line:
                copy_bandwidth = float(line.split()[1])
                break

        return copy_bandwidth

    except Exception as e:
        print(f"Error running STREAM benchmark: {e}")
        return None

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=3
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, load_core, comp_core):
    model.eval()
    ys = []
    y_hats = []
    with dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata["feat"]
                ys.append(blocks[-1].dstdata["label"])
                y_hats.append(model(blocks, x))
                accuracy = MulticlassAccuracy(num_classes=47)
    return accuracy(torch.cat(y_hats), torch.cat(ys))


def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(pred, label)


def train(rank, size, args, device, g, dataset, model,meta_data):
    # create sampler & dataloader
    dist.init_process_group('gloo', rank=rank, world_size=size)
    model = DistributedDataParallel(model)
    n = psutil.cpu_count(logical = False)

    if size == 1:
        load_core = list(range(0,8))
        comp_core = list(range(8,n))
    elif size == 2:
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,n//2))
        else:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n))
    elif size == 4:
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,n//4))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//2))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//4*3))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n))

    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    sampler = NeighborSampler(
        [15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=(4096//size),
        use_ddp=True,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=256,
        use_ddp=True,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    cnt = 0
    num_nodes = []
    num_edges = []
    num_layer = 0
    total_nodes = 0
    total_edges = 0
    exe_time = 0
    if rank == 0:
        for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
            if it == 10: break
            num_layer = len(blocks)
            cnt += 1
            for i in range(num_layer):
                if it == 0:
                    num_nodes.append(blocks[i].number_of_nodes())
                    num_edges.append(blocks[i].number_of_edges())
                else:
                    num_nodes[i] += blocks[i].number_of_nodes()
                    num_edges[i] += blocks[i].number_of_edges()
        for i in range(num_layer):
            num_nodes[i]/=cnt
            num_edges[i]/=cnt
            total_nodes += num_nodes[i]
            total_edges += num_edges[i]

    for epoch in range(2):
        model.train()
        total_loss = 0
        start = time.time()
        # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                # if (it+1) == 100: break
                x = blocks[0].srcdata["feat"]
                y = blocks[-1].dstdata["label"]
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
        end = time.time()
        exe_time = end - start
        if rank == 0 and epoch == 1:
            meta_data.append(exe_time)
            meta_data.append(total_nodes)
            meta_data.append(total_edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    parser.add_argument(
        "--process",
        default= "1",
        choices=["1", "2", "4"],
    )

    args = parser.parse_args()
    args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    size = 1
    if args.process == "2":
        size = 2
    elif args.process == "4":
        size = 4
        
    # load and preprocess dataset
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    hidden_size = 128
    model = SAGE(in_size, hidden_size, out_size).to(device)

    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    
    processes = []

    mp.set_start_method('fork')
    with Manager() as manager:
        meta_data = manager.list()
        for rank in range(size):
            p = dmp.Process(target=train, args=(rank, size, args, device, g, dataset, model, meta_data))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        mem_bw = run_stream_benchmark()
        meta_data.append(psutil.cpu_count(logical = False))
        meta_data.append(mem_bw)
        meta_data.append(size)
        meta_data.append(in_size)
        meta_data.append(hidden_size)
        meta_data.append(out_size)
        print(meta_data)
        file = open('regression.csv', 'a', newline ='')
        # writing the data into the file
        with file:   
            write = csv.writer(file)
            write.writerows(meta_data)

    print("program finished.")
