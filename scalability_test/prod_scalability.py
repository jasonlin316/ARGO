import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import dgl
import dgl.nn as dglnn
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from ogb.nodeproppred import DglNodePropPredDataset
import tqdm
import argparse
from torchmetrics.classification import MulticlassAccuracy
from torch.profiler import profile, record_function, ProfilerActivity
import time
import psutil

comp_core = []
load_core = []

class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
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
        feat = g.ndata['feat']
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=['feat'])
        dataloader = DataLoader(
                g, torch.arange(g.num_nodes()).to(g.device), sampler, device=device,
                batch_size=batch_size, shuffle=False, drop_last=False,
                num_workers=0)
        buffer_device = torch.device('cpu')
        pin_memory = (buffer_device != device)

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(), self.hid_size if l != len(self.layers) - 1 else self.out_size,
                device=buffer_device, pin_memory=pin_memory)
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x) # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]:output_nodes[-1]+1] = h.to(buffer_device)
            feat = y
        return y

def evaluate(model, graph, dataloader):
    model.eval()
    ys = []
    y_hats = []
    with dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
        for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            with torch.no_grad():
                x = blocks[0].srcdata['feat']
                ys.append(blocks[-1].dstdata['label'])
                y_hats.append(model(blocks, x))
                accuracy = MulticlassAccuracy(num_classes=47)
    return accuracy(torch.cat(y_hats), torch.cat(ys))

def layerwise_infer(device, graph, nid, model, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(graph, device, batch_size) # pred in buffer_device
        pred = pred[nid]
        label = graph.ndata['label'][nid].to(pred.device)
        accuracy = MulticlassAccuracy(num_classes=47)

        return accuracy(pred, label)

def train(args, device, g, dataset, model):
    # create sampler & dataloader
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    if args.algo == 'mini':
        sampler = NeighborSampler([15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
                                prefetch_node_feats=['feat'],
                                prefetch_labels=['label'])
        # sampler = NeighborSampler([25, 10])  # fanout for [layer-0, layer-1, layer-2]
                          
    else:
        sampler = MultiLayerFullNeighborSampler(2)
    use_uva = (args.mode == 'mixed')
    
    train_dataloader = DataLoader(g, train_idx, sampler, device=device,
                                  batch_size=4096, shuffle=True,
                                  drop_last=False, num_workers = 4,
                                  use_uva=use_uva)

    val_dataloader = DataLoader(g, val_idx, sampler, device=device,
                                batch_size=1024, shuffle=True,
                                drop_last=False, num_workers = 4,
                                use_uva=use_uva)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
            for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                x = blocks[0].srcdata['feat']
                y = blocks[-1].dstdata['label']
                y_hat = model(blocks, x)
                loss = F.cross_entropy(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
        # acc = evaluate(model, g, val_dataloader)
        # end = time.time()
        # print("Epoch ", epoch , "; epoch time: ", (end-start), " sec")
        # print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
        #       .format(epoch, total_loss / (it+1), acc.item()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='mixed', choices=['cpu', 'mixed', 'puregpu'],
                        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
                             "'puregpu' for pure-GPU training.")
    parser.add_argument("--algo", default='mini', choices=['mini', 'full'])
    parser.add_argument("--core", default='half', choices=['half', 'full'])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = 'cpu'
    print(f'Training in {args.mode} mode.')
    load_core = list(range(0,4))
    
    # load and preprocess dataset
    print('Loading data')
    dataset = AsNodePredDataset(DglNodePropPredDataset('ogbn-products', root = './Dataset/'))
    g = dataset[0]
    g = g.to('cuda' if args.mode == 'puregpu' else 'cpu')
    device = torch.device('cpu' if args.mode == 'cpu' else 'cuda')

    # create GraphSAGE model
    in_size = g.ndata['feat'].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 128, out_size).to(device)

    print('Training...')
    n = psutil.cpu_count(logical = False)
    if n >= 4+4:
        comp_core = list(range(4,4+4))
        start = time.time()
        train(args, device, g, dataset, model)
        end = time.time()
        exe_time = end - start
        msg = "4 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_products.txt", "a") as text_file:
            text_file.write(msg)
    if n >= 8+4:
        comp_core = list(range(4,4+8))
        start = time.time()
        train(args, device, g, dataset, model)
        end = time.time()
        exe_time = end - start
        msg = "8 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_products.txt", "a") as text_file:
            text_file.write(msg)
    if n >= 16+4:
        comp_core = list(range(4,4+16))
        start = time.time()
        train(args, device, g, dataset, model)
        end = time.time()
        exe_time = end - start
        msg = "16 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_products.txt", "a") as text_file:
            text_file.write(msg)
    if n >= 32+4:
        comp_core = list(range(4,4+32))
        start = time.time()
        train(args, device, g, dataset, model)
        end = time.time()
        exe_time = end - start
        msg = "32 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_products.txt", "a") as text_file:
            text_file.write(msg)
    if n >= 64+4:
        comp_core = list(range(4,4+64))
        start = time.time()
        train(args, device, g, dataset, model)
        end = time.time()
        exe_time = end - start
        msg = "64 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_products.txt", "a") as text_file:
            text_file.write(msg)
    if n >= 128+4:
        comp_core = list(range(4,4+128))
        start = time.time()
        train(args, device, g, dataset, model)
        end = time.time()
        exe_time = end - start
        msg = "128 cores: " + str(exe_time/10) + " sec\n"
        with open("DGL_products.txt", "a") as text_file:
            text_file.write(msg)
    
    # use all cores
    comp_core = list(range(4,n))
    start = time.time()
    train(args, device, g, dataset, model)
    end = time.time()
    exe_time = end - start
    msg = str(n-4) + " cores: " + str(exe_time/10) + " sec\n"
    with open("DGL_products.txt", "a") as text_file:
        text_file.write(msg)


    
   
