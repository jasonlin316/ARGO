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
    NeighborSampler, ShaDowKHopSampler,
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

from load_mag_to_shm import fetch_datas_from_shm
from merge import merge_trace_files

import csv

TRACE_NAME = 'mixture_product_{}.json'
OUTPUT_TRACE_NAME = "combine.json"


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

def get_last_level_cache_size():
    try:
        # Execute the "lscpu" command and capture its output
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        output = result.stdout

        # Parse the output to find the L3 cache size
        lines = output.split('\n')
        for line in lines:
            if 'L3 cache:' in line:
                cache_info = line.split(':')
                if len(cache_info) == 2:
                    cache_size_str = cache_info[1].strip()
                    cache_size_mb = int(cache_size_str.replace(' MiB', ''))
                    return cache_size_mb
        return None

    except Exception as e:
        print("Error occurred:", e)
        return None

    
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


class UnevenDDPTensorizedDataset(torch.utils.data.IterableDataset):
    """Custom Dataset wrapper that returns a minibatch as tensors or dicts of tensors.
    When the dataset is on the GPU, this significantly reduces the overhead.

    This class additionally saves the index tensor in shared memory and therefore
    avoids duplicating the same index tensor during shuffling.
    """

    def __init__(self, indices, total_batch_size, sub_batch_sizes, drop_last, shuffle):
        self.rank = dist.get_rank()
        self.seed = 0
        self.epoch = 0
        self._mapping_keys = None
        self.drop_last = drop_last
        self._shuffle = shuffle

        self.prefix_sum_batch_size = sum(sub_batch_sizes[:self.rank])
        self.batch_size = sub_batch_sizes[self.rank]

        len_indices = len(indices)
        if self.drop_last and len_indices % total_batch_size != 0:
            self.num_batches = math.ceil((len_indices - total_batch_size) / total_batch_size)
        else:
            self.num_batches = math.ceil(len_indices / total_batch_size)
        self.total_size = self.num_batches * total_batch_size
        # If drop_last is False, we create a shared memory array larger than the number
        # of indices since we will need to pad it after shuffling to make it evenly
        # divisible before every epoch.  If drop_last is True, we create an array
        # with the same size as the indices so we can trim it later.
        self.shared_mem_size = self.total_size if not self.drop_last else len_indices
        self.num_indices = len_indices

        self._id_tensor = indices
        # self._device = self._id_tensor.device
        self.device = self._id_tensor.device

        self._indices = call_once_and_share(
            self._create_shared_indices, (self.shared_mem_size,), torch.int64)

    def _create_shared_indices(self):
        indices = torch.empty(self.shared_mem_size, dtype=torch.int64)
        num_ids = self._id_tensor.shape[0]
        torch.arange(num_ids, out=indices[:num_ids])
        torch.arange(self.shared_mem_size - num_ids, out=indices[num_ids:])
        return indices

    def shuffle(self):
        """Shuffles the dataset."""
        # Only rank 0 does the actual shuffling.  The other ranks wait for it.
        if self.rank == 0:
            np.random.shuffle(self._indices[:self.num_indices].numpy())
            if not self.drop_last:
                # pad extra
                self._indices[self.num_indices:] = \
                    self._indices[:self.total_size - self.num_indices]
        dist.barrier()

    def __iter__(self):
        start = self.prefix_sum_batch_size * self.num_batches
        end = start + self.batch_size * self.num_batches
        indices = _divide_by_worker(self._indices[start:end], self.batch_size, self.drop_last)
        id_tensor = self._id_tensor[indices]
        return _TensorizedDatasetIter(
            id_tensor, self.batch_size, self.drop_last, self._mapping_keys, self._shuffle)

    def __len__(self):
        return self.total_size


def is_cpu_proc(num_cpu_proc, rank=None):
    if rank is None:
        rank = dist.get_rank()
    return rank < num_cpu_proc


def device_mapping(num_cpu_proc):
    assert not is_cpu_proc(num_cpu_proc), "For GPU Comp process only"
    return dist.get_rank() - num_cpu_proc

# num_of_samplers = 8

def assign_cores(num_cpu_proc):
    assert is_cpu_proc(num_cpu_proc), "For CPU Comp process only"
    rank = dist.get_rank()
    load_core, comp_core = [], []
    n = psutil.cpu_count(logical=False)
    size = num_cpu_proc
    if size <= 4:
        num_of_samplers = 4//size
    else:
        num_of_samplers = 8//size
    # num_of_samplers = 8
    load_core = list(range(n//size*rank,n//size*rank+num_of_samplers))
    comp_core = list(range(n//size*rank+num_of_samplers,n//size*(rank+1)))
    # comp_core = list(range(n//size*rank+num_of_samplers,n//size*rank+num_of_samplers+30))

    return load_core, comp_core


def get_subbatch_size(args, rank=None) -> int:
    if rank is None:
        rank = dist.get_rank()
    world_size = dist.get_world_size()
    cpu_batch_size = int(args.batch_size * args.cpu_gpu_ratio)
    if is_cpu_proc(args.cpu_process, rank):
        return cpu_batch_size // args.cpu_process + \
            (cpu_batch_size % args.cpu_process if rank == args.cpu_process - 1 else 0)
    else:
        return (args.batch_size - cpu_batch_size) // args.gpu_process + \
            ((args.batch_size - cpu_batch_size) % args.gpu_process if rank == world_size - 1 else 0)


def _train(loader, model, opt, **kwargs):
    total_loss = 0
    if kwargs['rank'] == 0:
        pbar = tqdm(total=kwargs['train_size'])
        epoch = kwargs['epoch']
        pbar.set_description(f'Epoch {epoch:02d}')

    process = kwargs['process']
    device = torch.device("cpu" if is_cpu_proc(process)
                          else "cuda:{}".format(device_mapping(process)))
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
        torch.cuda.empty_cache()

        total_loss += loss.item()  # avoid cuda memory accumulation
        if kwargs['rank'] == 0:
            pbar.update(kwargs['batch_size'])
            # pbar.update(output_nodes.shape[0])
    if kwargs['rank'] == 0:
        pbar.close()
    return total_loss


def _train_cpu(load_core, comp_core, **kwargs):
    with kwargs['loader'].enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core):
        loss = _train(**kwargs)
    return loss


def train(rank, world_size, args, g, data, hidden, meta_data):
    num_classes, train_idx = data
    dist.init_process_group('gloo', rank=rank, world_size=world_size)

    device = torch.device("cpu" if is_cpu_proc(args.cpu_process)
                          else "cuda:{}".format(device_mapping(args.cpu_process)))

    if not is_cpu_proc(args.cpu_process):
        torch.cuda.set_device(device)

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    model = GNN(in_size, hidden, num_classes,
                num_layers=args.layer, model_name=args.model).to(device)
    model = DistributedDataParallel(model)

    # g = dgl.add_self_loop(g)  # for GCN model, not work for Mag
    if args.cpu_process <= 4:
        num_of_samplers = 4//args.cpu_process
    else:
        num_of_samplers = 8//args.cpu_process
    
    # create loader
    drop_last, shuffle = True, True
    sub_batch_sizes = [get_subbatch_size(args, r) for r in range(world_size)]
    if rank == 0: print('SubBatch sizes:', sub_batch_sizes)
    train_indices = UnevenDDPTensorizedDataset(
        train_idx.to(device),
        # test_idx.to(device),
        args.batch_size,
        sub_batch_sizes,
        drop_last,
        shuffle
    )
    if args.sampler.lower() == 'neighbor':
        if args.neighbor == 0:
            sampler = NeighborSampler(
                [10, 10, 10],
                prefetch_node_feats=["feat"],
                prefetch_labels=["label"],
            )
            assert len(sampler.fanouts) == args.layer
        elif args.neighbor == 1:
            sampler = NeighborSampler(
                [12, 10, 8],
                prefetch_node_feats=["feat"],
                prefetch_labels=["label"],
            )
            assert len(sampler.fanouts) == args.layer
        else:
            sampler = NeighborSampler(
                [10, 10, 5],
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
        train_indices,  # train_idx.to(device)
        sampler,
        device=device,
        batch_size=get_subbatch_size(args),
        use_ddp=True,
        use_uva=not is_cpu_proc(args.cpu_process),
        drop_last=drop_last,
        shuffle=shuffle,
        num_workers=num_of_samplers
    )
    cnt = 0
    num_nodes = []
    num_edges = []
    num_layer = 0
    total_nodes = 0
    total_edges = 0

    # training loop
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    params = {
        # training
        'loader': train_dataloader,
        'model': model,
        'opt': opt,
        # logging
        'rank': rank,
        'train_size': len(train_indices),
        'batch_size': args.batch_size,
        'device': device,
        'process': args.cpu_process
    }
    for epoch in range(2):
        params['epoch'] = epoch
        model.train()
        tik = time.time()
        if is_cpu_proc(args.cpu_process):
            if params.get('load_core', None) is None or params.get('comp_core', None):
                params['load_core'], params['comp_core'] = assign_cores(args.cpu_process)
            loss = _train_cpu(**params)
        else:
            loss = _train(**params)
        if rank == 0 and epoch == 1:
            meta_data.append(time.time() - tik)
            meta_data.append(0) # total nodes
            meta_data.append(0) # total edges
    
    #ToDo: total nodes/edges of all processes
    for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
        if it == 10: break
        num_layer = 3
        cnt += 1
        if hasattr(blocks, '__len__'):
            for i in range(num_layer):
                if it == 0:
                    num_nodes.append(blocks[i].number_of_nodes())
                    num_edges.append(blocks[i].number_of_edges())
                else:
                    num_nodes[i] += blocks[i].number_of_nodes()
                    num_edges[i] += blocks[i].number_of_edges()
        else:
            if it == 0:
                num_nodes.append(blocks.number_of_nodes())
                num_edges.append(blocks.number_of_edges())
            else:
                num_nodes[0] += blocks.number_of_nodes()
                num_edges[0] += blocks.number_of_edges()
    if hasattr(blocks, '__len__'):
        for i in range(num_layer):
            num_nodes[i]/=cnt
            num_edges[i]/=cnt
            total_nodes += num_nodes[i]
            total_edges += num_edges[i]
    else:
        num_nodes[0]/=cnt
        num_edges[0]/=cnt
        total_nodes += num_nodes[0]
        total_edges += num_edges[0]
    
    meta_data[1] += total_nodes
    meta_data[2] += total_edges
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='/data1/dgl')
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products", "mag240M", "reddit", "yelp", "flickr"])
    parser.add_argument("--cpu_process",
                        type=int,
                        default=1,
                        choices=[0, 1, 2, 4, 8])
    parser.add_argument("--gpu_process",
                        type=int,
                        default=0)
    parser.add_argument("--cpu_gpu_ratio",
                        type=float,
                        default=1)
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
    parser.add_argument("--neighbor",
                        type=int,
                        default=0,
                        choices=[0,1,2])
    arguments = parser.parse_args()
    print("Program starts")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    cache_size_mb = get_last_level_cache_size()

    # Assure Consistency
    if arguments.cpu_gpu_ratio == 0 or arguments.cpu_process == 0:
        arguments.cpu_gpu_ratio = 0
        arguments.cpu_process = 0
    if arguments.cpu_gpu_ratio == 1 or arguments.gpu_process == 0:
        arguments.cpu_gpu_ratio = 1
        arguments.gpu_process = 0
    nprocs = arguments.cpu_process + arguments.gpu_process
    assert nprocs > 0
    print(f'\nUse {arguments.cpu_process} CPU Comp processes and {arguments.gpu_process} GPUs\n'
          f'The batch size is {arguments.batch_size} with {arguments.cpu_gpu_ratio} cpu/gpu workload ratio\n'
          f'Sampler: {arguments.sampler}, Model: {arguments.model}, Layer: {arguments.layer}\n')

    # load and preprocess dataset
    print('Use Dataset:', arguments.dataset)
    tik = time.time()
    if arguments.dataset == 'mag240M':
        dataset = MAG240MDataset(root='../HiPC')
        print('Start Loading Graph Structure')
        (g,), _ = dgl.load_graphs('./graph.dgl')
        g = g.formats(["csc"])
        print('Graph Structure Loading Finished!')
        paper_offset = dataset.num_authors + dataset.num_institutions
        dataset.train_idx = torch.from_numpy(dataset.get_idx_split("train")) + paper_offset
        g.ndata["feat"] = fetch_datas_from_shm()
        g.ndata["label"] = torch.cat([torch.empty((paper_offset,), dtype=torch.long),
                                      torch.LongTensor(dataset.paper_label[:])])
        print('Graph Feature/Label Loading Finished!')
    elif arguments.dataset in ["reddit","flickr","yelp"]:
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
        dataset = AsNodePredDataset(DglNodePropPredDataset(arguments.dataset, arguments.data_path))
        g = dataset[0]

    """
    Note 1: This func avoid creating certain graph formats in each sub-process to save memory
    Note 2: This func will init CUDA. It is not possible to use CUDA in a child process 
            created by fork(), if CUDA has been initialized in the parent process. 
    """
    # g.create_formats_()

    data = (
        dataset.num_classes,
        dataset.train_idx,
    )

    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    hidden_size = int(arguments.hidden)

    tok = time.time()
    print(f"Data loading finished, Elapsed Time: {time.time() - tik: .1f}s\n")

    # multi-processes training
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29501'

    # train(0, nprocs, arguments, g, data)
    mp.set_start_method('fork')
    processes = []
    with Manager() as manager:
        meta_data = manager.list()
        for i in range(nprocs):
            p = dmp.Process(target=train, args=(i, nprocs, arguments, g, data, hidden_size, meta_data))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        mem_bw = run_stream_benchmark()
        meta_data.append(psutil.cpu_count(logical = False))
        meta_data.append(mem_bw)
        meta_data.append(cache_size_mb)
        meta_data.append(arguments.cpu_process)
        meta_data.append(in_size)
        meta_data.append(hidden_size)
        meta_data.append(out_size)
        print(meta_data)
        with open('new_regression_test.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(meta_data)

    print("Program finished")
