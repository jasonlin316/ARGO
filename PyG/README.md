This is the README for running autotuner with DDP-PyG.

* `DDP-obgn-product-sage.py` - the earliest DDP-PyG program, can only train with obgn-product dataset, with limited choices of n_sampler and n_trainer 
* `gnn_train.py` - mainly designed as the obejective function for Bayesian Optimization in `bo.py`, running 1 training epoch and print the epoch time, can choose dataset from [`obgn-products`, `flickr`, `reddit`, `yelp`]

* `gnn_train2.py` - mainly designed for grid-search training. It will record the epoch time in `.csv` file. Can choose dataset from [`obgn-products`, `flickr`, `reddit`, `yelp`]
* `grid_search.sh` - run `bash PyG/grid_search.sh` to launch exhaustive search. It might take few days to run. The result would be saved as `grid_search_<dataset>.csv` in `PyG` folder. 
* `verify_gen.py` - use this to generate `grid_search.sh` bash file. 
* `bo.py` - run `python -W ignore PyG/bo.py --dataset <dataset>` to launch the Bayesian Optimization. Result would be saved in `PyG/bo_result` folder. 
* `bo.sh` - run `bash PyG/bo.sh` to run Bayesian Optimization for each dataset. 