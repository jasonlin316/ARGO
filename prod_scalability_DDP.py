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
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil



comp_core = []
load_core = []

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


def train(rank, size, args, device, g, dataset, model):
    # create sampler & dataloader
    dist.init_process_group('gloo', rank=rank, world_size=size)
    model = DistributedDataParallel(model)
    n = psutil.cpu_count(logical = False)

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

    if n > 16+4:
        cores = 1
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,4+cores))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//4+4+cores))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//2+4+cores))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n//4*3+4+cores))
        start = time.time()
        for epoch in range(10):
            model.train()
            total_loss = 0
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        msg = "4 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_DDP_products.txt", "a") as text_file:
            text_file.write(msg)
    if n > 16+8:
        cores = 2
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,4+cores))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//4+4+cores))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//2+4+cores))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n//4*3+4+cores))
        start = time.time()
        for epoch in range(10):
            model.train()
            total_loss = 0
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        msg = "8 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_DDP_products.txt", "a") as text_file:
            text_file.write(msg)
    if n > 16+16:
        cores = 4
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,4+cores))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//4+4+cores))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//2+4+cores))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n//4*3+4+cores))
        start = time.time()
        for epoch in range(10):
            model.train()
            total_loss = 0
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        msg = "16 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_DDP_products.txt", "a") as text_file:
            text_file.write(msg)
    if n > 16+32:
        cores = 8
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,4+cores))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//4+4+cores))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//2+4+cores))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n//4*3+4+cores))
        start = time.time()
        for epoch in range(10):
            model.train()
            total_loss = 0
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        msg = "32 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_DDP_products.txt", "a") as text_file:
            text_file.write(msg)
    if n > 16+64:
        cores = 16
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,4+cores))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//4+4+cores))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//2+4+cores))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n//4*3+4+cores))
        start = time.time()
        for epoch in range(10):
            model.train()
            total_loss = 0
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        msg = "64 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_DDP_products.txt", "a") as text_file:
            text_file.write(msg)
    if n > 16+128:
        cores = 32
        if rank == 0:
            load_core = list(range(0,4))
            comp_core = list(range(4,4+cores))
        elif rank == 1:
            load_core = list(range(n//4,n//4+4))
            comp_core = list(range(n//4+4,n//4+4+cores))
        elif rank == 2:
            load_core = list(range(n//2,n//2+4))
            comp_core = list(range(n//2+4,n//2+4+cores))
        else:
            load_core = list(range(n//4*3,n//4*3+4))
            comp_core = list(range(n//4*3+4,n//4*3+4+cores))
        start = time.time()
        for epoch in range(10):
            model.train()
            total_loss = 0
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
        msg = "128 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_DDP_products.txt", "a") as text_file:
            text_file.write(msg)
    cores = (n-16)//4
    if rank == 0:
        load_core = list(range(0,4))
        comp_core = list(range(4,4+cores))
    elif rank == 1:
        load_core = list(range(n//4,n//4+4))
        comp_core = list(range(n//4+4,n//4+4+cores))
    elif rank == 2:
        load_core = list(range(n//2,n//2+4))
        comp_core = list(range(n//2+4,n//2+4+cores))
    else:
        load_core = list(range(n//4*3,n//4*3+4))
        comp_core = list(range(n//4*3+4,n//4*3+4+cores))
    start = time.time()
    for epoch in range(10):
        model.train()
        total_loss = 0
        with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
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
    msg = str(n-16) + " cores: " + str(exe_time/10) + " sec\n"
    with open("DGL_DDP_products.txt", "a") as text_file:
        text_file.write(msg)
    


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
        default= "4",
        choices=["1", "2", "4"],
    )

    args = parser.parse_args()
    args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    size = 4
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
    model = SAGE(in_size, 128, out_size).to(device)

    # model training
    master_addr = '127.0.0.1'
    master_port = '29500'
    
    processes = []
    n = psutil.cpu_count(logical = False)

    mp.set_start_method('fork')
    for rank in range(size):
        p = dmp.Process(target=train, args=(rank, size, args, device, g, dataset, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("program finished.")
