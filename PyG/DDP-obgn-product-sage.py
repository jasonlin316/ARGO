# Reaches around 0.7930 test accuracy.

import argparse
import os.path as osp
import time

import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from tqdm import tqdm

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv

# new imports
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import psutil
import os
from torch.profiler import profile, ProfilerActivity
import os
# import merge

def get_core_num(process_num, rank, n, load_core_num=4):
    
    if process_num == 1:
        load_core = list(range(0,load_core_num))
        comp_core = list(range(load_core_num,n))

    elif process_num == 2:
        if rank == 0:
            load_core = list(range(0,load_core_num))
            comp_core = list(range(load_core_num,n//2))
        else:
            load_core = list(range(n//2,n//2+load_core_num))
            comp_core = list(range(n//2+load_core_num,n))

    elif process_num == 4:
        if rank == 0:
            load_core = list(range(0,load_core_num))
            comp_core = list(range(load_core_num,n//4))
        elif rank == 1:
            load_core = list(range(n//4,n//4+load_core_num))
            comp_core = list(range(n//4+load_core_num,n//2))
        elif rank == 2:
            load_core = list(range(n//2,n//2+load_core_num))
            comp_core = list(range(n//2+load_core_num,n//4*3))
        else:
            load_core = list(range(n//4*3,n//4*3+load_core_num))
            comp_core = list(range(n//4*3+load_core_num,n))
    return load_core, comp_core


parser = argparse.ArgumentParser()

parser.add_argument(
    "--process",
    default= "1",
    choices=["1", "2", "4"],
)

parser.add_argument(
    "--l_core",
    default= "4",
    help="load core number"
)

parser.add_argument(
    "--run_times",
    default= "1",
    help="repeat times to run the experiment",
)
    


args = parser.parse_args()
args.mode = "cpu"
print(f"Training in {args.mode} mode.")
times = int(args.run_times)

process_num = int(args.process)
load_core_num = int(args.l_core)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device, "process_num: ", process_num, "load_core_num: ", load_core_num)
# device = torch.device('cpu')
dataset = PygNodePropPredDataset(name = 'ogbn-products')
split_idx = dataset.get_idx_split()
evaluator = Evaluator(name='ogbn-products')
data = dataset[0].to(device, 'x', 'y')
trace_list = []


subgraph_loader = NeighborLoader(
    data,
    input_nodes=None,
    num_neighbors=[-1],
    batch_size=4096,
    num_workers=4,
    persistent_workers=True,
)


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
    n = psutil.cpu_count(logical=False)
    load_core, comp_core = get_core_num(process_num, rank, n, load_core_num)
    print("rank {}: loading data using core {} ".format(rank,load_core))
    # set compute cores
    os.environ["OMP_NUM_THREADS"] = str(len(comp_core))
    os.environ["OMP_PLACES"] = "cores"
    os.environ["OMP_PROC_BIND"] = "close"
    os.environ["KMP_AFFINITY"] = "proclist=[" + ",".join(str(core) for core in comp_core) + "]"
    # cancel the KMP warning
    os.environ["KMP_WARNINGS"] = "off"

    torch.set_num_threads(len(comp_core))
    print("rank {}, using compute core: {}".format(rank, comp_core))

    dist.init_process_group('gloo', rank=rank, world_size=process_num)  
    train_idx = split_idx['train'].to(device)

    # 在DDP_time中追加写入一空行
    with open('./DDP_profile/DDP_time.txt', 'a') as f:
        f.write('\n')
        if rank == 0:
            f.write(f'Load_num:{load_core_num}| Process_num:{process_num}\n')
            f.write("=========================================================\n")
    
    for run in range(times):
        if rank == 0:
            print(f'\nRun {run:02d}:\n')
        model.reset_parameters()
        train(model, rank, load_core, train_idx)
        # combine_name = "DDP_profile/combine_p{}_l{}.json".format(process_num, load_core_num)
        # merge.merge_trace_files(trace_list, combine_name)
        test_loading_time(rank, train_idx, process_num, load_core_num)

def test_loading_time(rank, train_idx, process_num, load_core_num):
    print("rank {}: testing loading time".format(rank))
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
        num_workers=load_core_num,
        persistent_workers=True,
        sampler=train_sampler
    )
    start_time = time.time()
    for batch in train_loader:
        pass
    end_time = time.time()
    # 在DDP_time.txt中追加写入每个进程的加载时间
    avg_load_time = (end_time - start_time) / process_num
    with open('./DDP_profile/DDP_time.txt', 'a') as f:
        f.write(f'Rank {rank} Load Time: {avg_load_time}\n')
    print(f'Load_num:{load_core_num}| Process_num:{process_num} | Rank {rank} Load Time: {avg_load_time}\n')
        

def train(model, rank, load_core, train_idx):
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
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
            persistent_workers=True,
            sampler=train_sampler
        )

        model = DistributedDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

        best_val_acc = final_test_acc = 0.0
        test_accs = []
        trace_name = "DDP_profile/trace_p{}_l{}_r{}.json".format(process_num, load_core_num, rank)
        for epoch in range(1):
            start_time = time.time()
            model.train()
            if rank == 0:
                pbar = tqdm(total=split_idx['train'].size(0)//process_num)
                pbar.set_description(f'Rank {rank} Epoch {epoch:02d}')

            total_loss = total_correct = total_cnt =  0
            with train_loader.enable_cpu_affinity(loader_cores = load_core):
                for batch in train_loader:
                    optimizer.zero_grad()
                    out = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
                    y = batch.y[:batch.batch_size].squeeze()
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss)
                    total_correct += int(out.argmax(dim=-1).eq(y).sum())
                    total_cnt += batch.batch_size
                    if rank == 0:
                        pbar.update(batch.batch_size)
            if rank == 0:
                pbar.close()

            end_time = time.time()
            with open('./DDP_profile/DDP_time.txt', 'a') as f:
                f.write(f'Rank {rank} Epoch {epoch} Train_time: {end_time - start_time}\n')

            loss = total_loss / len(train_loader)
            approx_acc = total_correct / total_cnt
            print(f'Rank {rank}|Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {approx_acc:.4f}')
            
            # if epoch % 2 == 0:
            # train_acc, val_acc, test_acc = test()
            # print(f'Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            #     f'Test: {test_acc:.4f}')

            # if val_acc > best_val_acc:
            #     best_val_acc = val_acc
            #     final_test_acc = test_acc
            #     test_accs.append(final_test_acc)
    process_name = "process"+str(rank)
    prof.export_chrome_trace(trace_name)
    trace_list.append(trace_name)
    # test_acc = torch.tensor(test_accs)
    # print(test_accs, test_acc)
    # print('============================')
    # print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
    



@torch.no_grad()
def test():
    model.eval()

    out = model.inference(data.x)

    y_true = data.y.cpu()
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    val_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],

        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, val_acc, test_acc

model = SAGE(dataset.num_features, 256, dataset.num_classes, num_layers=3)
model = model.to(device)

# model training
master_addr = '127.0.0.1'
master_port = '29500'

processes = []
try:
    mp.set_start_method('fork')
    print("forked")
except RuntimeError:
    pass
        
for rank in range(process_num):
    p = mp.Process(target=run, args=(rank, times, model, process_num))
    p.start()
    processes.append(p)
for p in processes:
    p.join()

print("program finished.")







