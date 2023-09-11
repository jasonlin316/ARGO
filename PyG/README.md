This is the README file for running autotuner with DDP-PyG.

* `DDP-obgn-product-sage.py` - the earliest DDP-PyG program, can only train with obgn-product dataset, with limited choices of n_sampler and n_trainer 
* `gnn_train.py` - mainly designed as the obejective function for Bayesian Optimization in `bo.py`, running 1 training epoch and print the epoch time, can choose dataset from [`obgn-products`, `flickr`, `reddit`, `yelp`]

* `gnn_train2.py` - mainly designed for grid-search training. It will record the epoch time in `.csv` file. Can choose dataset from [`obgn-products`, `flickr`, `reddit`, `yelp`]
* `grid_search.sh` - run `bash PyG/grid_search.sh` to launch exhaustive search. It might take few days to run. 
* `verify_gen.py` - use this to generate `grid_search.sh` bash file. 
---
## TODO: BO.py cannot run normally 
 Running this command in the terminal:
 
 `torchrun PyG/gnn_train.py --cpu_process 1 --n_sampler 1 --n_trainer 1 --dataset flickr`


 `gnn_train.py` is able to run. 

However, it freezes when called by subprocess in `bo.py` as follow:
```
command = ["torchrun", "PyG/gnn_train.py", "--dataset", arguments.dataset , '--cpu_process', str(1), '--n_sampler', str(1), '--n_trainer', str(1)]
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=50)
```
I found it got stuck when running line 119 in `gnn_train2.py`, which is

`model = DistributedDataParallel(model)`

But I don't know why yet. 