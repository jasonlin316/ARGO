import argparse
import math
import os
from contextlib import nullcontext

import dgl
import dgl.nn as dglnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.dataloading.dataloader import _divide_by_worker, _TensorizedDatasetIter
from dgl.multiprocessing import call_once_and_share
from multiprocessing import Process, Manager
from tqdm import tqdm
from dgl.data import AsNodePredDataset
from dgl.data import FlickrDataset, YelpDataset, RedditDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler, ShaDowKHopSampler, SAINTSampler
)
from ogb.nodeproppred import DglNodePropPredDataset
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from torch.profiler import profile, record_function, ProfilerActivity
from torchmetrics.classification import MulticlassAccuracy
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl.multiprocessing as dmp
from torch.nn.parallel import DistributedDataParallel
import psutil

from skopt import gp_minimize
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt

from load_mag_to_shm import fetch_datas_from_shm
# from auto_tuner import AutoTuner

import csv
TRACE_NAME = 'mixture_product_{}.json'
OUTPUT_TRACE_NAME = "combine.json"
import subprocess

    
class GNN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=3, model_name='sage'):
        super().__init__()
        self.layers = nn.ModuleList()

        # GraphSAGE-mean
        if model_name.lower() == 'sage':
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
            for i in range(num_layers-2):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        # GCN
        elif model_name.lower() == 'gcn':
            kwargs = {'norm': 'both', 'weight': True, 'bias': True, 'allow_zero_in_degree': True}
            self.layers.append(dglnn.GraphConv(in_size, hid_size, **kwargs))
            for i in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hid_size, hid_size, **kwargs))
            self.layers.append(dglnn.GraphConv(hid_size, out_size, **kwargs))
        else:
            raise NotImplementedError

        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        if hasattr(blocks, '__len__'):
            for l, (layer, block) in enumerate(zip(self.layers, blocks)):
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        else:
            for l, layer in enumerate(self.layers):
                h = layer(blocks, h)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        return h


def assign_cores(num_cpu_proc,n_samp,n_train):
    rank = dist.get_rank()
    load_core, comp_core = [], []
    n = psutil.cpu_count(logical=False)
    size = num_cpu_proc
    num_of_samplers = n_samp
    load_core = list(range(n//size*rank,n//size*rank+num_of_samplers))
    comp_core = list(range(n//size*rank+num_of_samplers,n//size*rank+num_of_samplers+n_train))

    return load_core, comp_core

def _train(loader, model, opt, **kwargs):
    total_loss = 0
    # if kwargs['rank'] == 0:
    #     pbar = tqdm(total=kwargs['train_size'])
    #     epoch = kwargs['epoch']
    #     pbar.set_description(f'Epoch {epoch:02d}')

    device = torch.device("cpu")
    for it, (input_nodes, output_nodes, blocks) in enumerate(loader):
        if hasattr(blocks, '__len__'):
            x = blocks[0].srcdata["feat"].to(torch.float32)
            y = blocks[-1].dstdata["label"]
        else:
            x = blocks.srcdata["feat"].to(torch.float32)
            y = blocks.dstdata["label"]
        # y_hat = model(blocks, x)
        if kwargs['device'] == "cpu":  # for papers100M
            y = y.type(torch.LongTensor)
            y_hat = model(blocks, x)
        else:
            y = y.type(torch.LongTensor).to(device)
            y_hat = model(blocks, x).to(device)
        try:
            loss = F.cross_entropy(y_hat[:output_nodes.shape[0]], y[:output_nodes.shape[0]])
        except:
            loss = F.binary_cross_entropy_with_logits(
                y_hat[:output_nodes.shape[0]].float(),
                y[:output_nodes.shape[0]].float(),
                reduction="sum"
            )  
        opt.zero_grad()
        loss.backward()
        opt.step()

        del input_nodes, output_nodes, blocks

        total_loss += loss.item()  
        # if kwargs['rank'] == 0:
            # pbar.update(kwargs['batch_size'])
            # pbar.update(output_nodes.shape[0])
    # if kwargs['rank'] == 0:
    #     pbar.close()
    return total_loss


def _train_cpu(load_core, comp_core, **kwargs):
    with kwargs['loader'].enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core):
        loss = _train(**kwargs)
    return loss

def train(rank, world_size, n_train, n_samp, args, g, data, counter):

    num_classes, train_idx = data
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    device = torch.device("cpu")

    hidden = args.hidden
    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    model = GNN(in_size, hidden, num_classes,
                num_layers=args.layer, model_name=args.model).to(device)
    model = DistributedDataParallel(model)
    
    num_of_samplers = n_samp
    
    # create loader
    drop_last, shuffle = True, True

    if args.sampler.lower() == 'neighbor':
        sampler = NeighborSampler(
            [15, 10, 5],
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
        assert len(sampler.fanouts) == args.layer
    elif args.sampler.lower() == 'shadow':
        sampler = ShaDowKHopSampler(
            [10, 5],
            output_device=device,
            prefetch_node_feats=["feat"],
        )
    else:
        raise NotImplementedError
    
    train_dataloader = DataLoader(
        g,
        train_idx.to(device),
        sampler,
        device=device,
        batch_size=1024,
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_of_samplers
    )

    # training loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    params = {
        # training
        'loader': train_dataloader,
        'model': model,
        'opt': opt,
        # logging
        'rank': rank,
        'train_size': len(train_idx),
        'batch_size': 1024,
        'device': device,
        'process': world_size
    }

    PATH = "model.pt"
    if counter[0] != 0:
        checkpoint = torch.load(PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    for epoch in range(1):
        params['epoch'] = epoch
        model.train()
        if params.get('load_core', None) is None or params.get('comp_core', None):
            params['load_core'], params['comp_core'] = assign_cores(world_size,n_samp,n_train)
        loss = _train_cpu(**params)
        if rank == 0:
            print("loss:", loss)
    
    dist.barrier()
    EPOCH = counter[0]
    LOSS = loss
    torch.save({
                'epoch': EPOCH,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': LOSS,
                }, PATH)
    # dist.barrier()
    # dist.destroy_process_group()

def launch(x, arguments, g, data, counter):
    n_proc = x[0]
    n_samp = x[1]
    n_train = x[2]
    
    tik = time.time()
    for i in range(n_proc):
        p = dmp.Process(target=train, args=(i, n_proc, n_train, n_samp, arguments, g, data, counter))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    t = time.time() - tik
    counter[0] = counter[0] + 1

    return t

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/data1/dgl') #/home/jason/DDP_GNN/dataset
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products", "mag240M", "reddit", "yelp", "flickr"])
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024 * 4)
    parser.add_argument('--sampler',
                        type=str,
                        default='neighbor',
                        choices=["neighbor", "shadow"])
    parser.add_argument('--model',
                        type=str,
                        default='sage',
                        choices=["sage", "gcn"])
    parser.add_argument('--layer',
                        type=int,
                        default=3)
    parser.add_argument("--hidden",
                        type=int,
                        default= 128)
    arguments = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    # load and preprocess dataset
    
    if arguments.dataset in ["reddit","flickr","yelp"]:
        if arguments.dataset == "reddit":
            dataset = RedditDataset()
        elif arguments.dataset == "flickr":
            dataset = FlickrDataset()
        else:
            dataset = YelpDataset()
        g = dataset[0]
        train_mask = g.ndata['train_mask']
        idx = []
        for i in range(len(train_mask)):
            if train_mask[i]:
                idx.append(i)
        dataset.train_idx = torch.tensor(idx)
    else:
        dataset = AsNodePredDataset(DglNodePropPredDataset(arguments.dataset, "/data1/dgl"))
        g = dataset[0]

    data = (
        dataset.num_classes,
        dataset.train_idx,
    )

    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    hidden_size = int(arguments.hidden)

    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'
    mp.set_start_method('fork', force=True)
    processes = []

    # auto_tuner = AutoTuner()
    space = [(2, 8), (1, 4), (1,32)] 
    counter = [0]
    result = gp_minimize(lambda x: launch(x, arguments, g, data, counter), space, n_calls=10, random_state=1, acq_func='EI')
    print(str(result.x_iters))
    print(str(result.func_vals))

    # result = GOAT(launch, space, n_calls=10, random_state=1, acq_func='EI', args=(g, data, counter), batch_size )

    print(str(result.x))
    for epoch in range(5):
        x = result.x
        launch(x, arguments, g, data, counter)
    # n_procs, n_train, n_samp = auto_tuner.update(t)
