"""Training graphsage w/ fp16.

Usage:

python bf16.py --fp16 --dataset

Note that GradScaler is not acitvated because the model successfully converges
without gradient scaling.

DGL's Message Passing APIs are not compatible with fp16 yet, hence we disabled
autocast when calling these APIs (e.g. apply_edges, update_all), see
https://github.com/yzh119/sage-fp16.git

In the default setting, using fp16 saves around 1GB GPU memory (from 4052mb
to 3042mb).
"""

import argparse
import time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset
from dgl.dataloading import DataLoader, NeighborSampler, MultiLayerFullNeighborSampler
from torch.profiler import profile, record_function, ProfilerActivity

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 use_fp16=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.use_fp16 = use_fp16
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            feat_src = feat_dst = self.feat_drop(feat)
            h_self = feat_dst
            graph.srcdata['h'] = feat_src
            if self.use_fp16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    graph.srcdata['h'] = graph.srcdata['h'].float()
                    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                
            else:
                graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))

            h_neigh = graph.dstdata['neigh']

            # GraphSAGE GCN does not require fc_self.
            # rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            rst = self.fc_neigh(h_neigh) + 0.01
            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 use_fp16):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type, use_fp16=use_fp16))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type, use_fp16=use_fp16))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type, use_fp16=use_fp16)) # activation None

    # def forward(self, graph, inputs):
    #     h = self.dropout(inputs)
    #     for l, layer in enumerate(self.layers):
    #         h = layer(graph, h)
    #         if l != len(self.layers) - 1:
    #             h = self.activation(h)
    #             h = self.dropout(h)
    #     return h

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

def evaluate(model, graph, features, labels, nid):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = AsNodePredDataset(DglNodePropPredDataset('ogbn-products'))
    g = data[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = data.num_classes
    n_edges = 100
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    load_core = list(range(0,4))
    
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        torch.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = train_mask.nonzero().squeeze()
    val_nid = val_mask.nonzero().squeeze()
    test_nid = test_mask.nonzero().squeeze()

    # graph preprocess and calculate normalization factor
    g = dgl.remove_self_loop(g)
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type,
                      args.fp16)

    
    # from torch.cuda.amp import GradScaler, autocast

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #if args.fp16:
    #    scaler = GradScaler()
    sampler = NeighborSampler([15, 10, 5],  # fanout for [layer-0, layer-1, layer-2]
                            prefetch_node_feats=['feat'],
                            prefetch_labels=['label'])
    train_idx = data.train_idx
    train_dataloader = DataLoader(g, train_idx, sampler, device='cpu',
                                  batch_size=4096, shuffle=True,
                                  drop_last=False, num_workers = 4,
                                  use_uva=False)
    # initialize graph
    dur = []
    # comp_core = list(range(4,4+1))
    # for epoch in range(1):
    #     model.train()
    #     optimizer.zero_grad()
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
    #         with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
    #             for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    #                 with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    #                     if (it+1) == 15 : break
    #                     x = blocks[0].srcdata['feat']
    #                     y = blocks[-1].dstdata['label']
    #                     y_hat = model(blocks, x)
    #                     loss = F.cross_entropy(y_hat, y)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #     prof.export_chrome_trace('product_f16_4096_1t.json')
    
    # comp_core = list(range(4,4+2))
    # for epoch in range(1):
    #     model.train()
    #     optimizer.zero_grad()
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
    #         with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
    #             for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    #                 with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    #                     if (it+1) == 15 : break
    #                     x = blocks[0].srcdata['feat']
    #                     y = blocks[-1].dstdata['label']
    #                     y_hat = model(blocks, x)
    #                     loss = F.cross_entropy(y_hat, y)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #     prof.export_chrome_trace('product_f16_4096_2t.json')

    # comp_core = list(range(4,4+4))
    # for epoch in range(1):
    #     model.train()
    #     optimizer.zero_grad()
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
    #         with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
    #             for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    #                 with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    #                     if (it+1) == 15 : break
    #                     x = blocks[0].srcdata['feat']
    #                     y = blocks[-1].dstdata['label']
    #                     y_hat = model(blocks, x)
    #                     loss = F.cross_entropy(y_hat, y)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #     prof.export_chrome_trace('product_f16_4096_4t.json')              
    
    # comp_core = list(range(4,4+8))
    # for epoch in range(1):
    #     model.train()
    #     optimizer.zero_grad()
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
    #         with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
    #             for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    #                 with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    #                     if (it+1) == 15 : break
    #                     x = blocks[0].srcdata['feat']
    #                     y = blocks[-1].dstdata['label']
    #                     y_hat = model(blocks, x)
    #                     loss = F.cross_entropy(y_hat, y)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #     prof.export_chrome_trace('product_f16_4096_8t.json')
        
    # comp_core = list(range(4,4+16))
    # for epoch in range(1):
    #     model.train()
    #     optimizer.zero_grad()
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
    #         with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
    #             for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
    #                 with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    #                     if (it+1) == 15 : break
    #                     x = blocks[0].srcdata['feat']
    #                     y = blocks[-1].dstdata['label']
    #                     y_hat = model(blocks, x)
    #                     loss = F.cross_entropy(y_hat, y)
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #     prof.export_chrome_trace('product_f16_4096_16t.json')

    comp_core = list(range(4,4+32))
    for epoch in range(1):
        model.train()
        optimizer.zero_grad()
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=True) as prof:
            with train_dataloader.enable_cpu_affinity(loader_cores = load_core, compute_cores =  comp_core):
                for it, (input_nodes, output_nodes, blocks) in enumerate(train_dataloader):
                    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                        if (it+1) == 15 : break
                        x = blocks[0].srcdata['feat']
                        y = blocks[-1].dstdata['label']
                        y_hat = model(blocks, x)
                        loss = F.cross_entropy(y_hat, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
        # prof.export_chrome_trace('product_f16_4096_32t.json')
        
    # print()
    # acc = evaluate(model, g, features, labels, test_nid)
    # print("Test Accuracy {:.4f}".format(acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--n-epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=128,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="mean",
                        help="Aggregator type: mean/gcn/pool/lstm")
    args = parser.parse_args()
    print(args)

    main(args)