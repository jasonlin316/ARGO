# ARGO: An Auto-Tuning Runtime System for Scalable GNN Training on Multi-Core Processor

This README includes how to:
1. [Set up the environment](#1-setting-up-the-environment)
2. [Run the example code](#2-running-the-example-GNN-program)
3. [Modify your own GNN program to enable ARGO.](#3-enabling-ARGO-on-your-own-GNN-program)

While we use the Deep Graph Library (DGL) as an example here, ARGO is also compatible with PyTorch-Geometric (PyG) and details can be found in the PyG folder.

## 1. Setting up the environment

1. Clone the repository:

```shell
git clone https://github.com/jasonlin316/DDP_GNN.git
```

2. Download Anaconda and install
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-2023.03-Linux-x86_64.sh
```

3. Create a virtual environment:

```shell
conda create -n py38 python=3.8.1
```

4. Active the virtual environment:

```shell
conda activate py38
```

5. Install required packages:

```shell
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch
conda install -c dglteam dgl
conda install -c conda-forge ogb
conda install -c conda-forge torchmetrics
conda install -c conda-forge scikit-optimize
```
Note: there exist a bug in the older version of the Scikit-Optimization library.  
To fix the bug, find the "transformer.py" which should be located in
   ```~/anaconda3/envs/py38/lib/python3.8/site-packages/skopt/space/transformers.py```. Once open the file, replace all ```np.int``` with ```int```.

6. Download the OGB datasets (optional if you are not running any)
```
python ogb_example.py --dataset <ogb_dataset>
```
- Available choices [ogbn-products, ogbn-papers100M]  

The program will ask if you want to download the dataset; please enter "y" for the program to proceed. You may terminate the program after the dataset is downloaded. 


## 2. Running the example GNN program
### Usage
  ```
  python main.py --dataset ogbn-products --sampler shadow --model sage
  ``` 
  Important Arguments: 
  - `--dataset`: the training datasets. Available choices [ogbn-products, ogbn-papers100M, reddit, flickr, yelp]
  - `--sampler`: the mini-batch sampling algorithm. Available choices [shadow, neighbor]
  - `--model`: GNN model. Available choices [gcn, sage]
  - `--layer`: number of GNN layers.
  - `--hidden`: hidden feature dimension.
  - `--batch_size`: the size of the mini-batch.

Note: the default number of layer is 3. If you want to change the number of layers for the Neighbor Sampler, please update the sample size in ```line 114```.



## 3. Enabling ARGO on your own GNN program
   
