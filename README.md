# ARGO: An Auto-Tuning Runtime System for Scalable GNN Training on Multi-Core Processor

This README includes how to:
1. [Set up the environment](#1-setting-up-the-environment)
2. [Run the example code](#2-running-the-example-GNN-program)
3. [Modify your own GNN program to enable ARGO.](#3-enabling-ARGO-on-your-own-GNN-program)

While we use the Deep Graph Library (DGL) as an example here, ARGO is also compatible with PyTorch-Geometric (PyG) and details can be found in the PyG folder.

## 1. Setting up the environment

1. Clone the repository:

```shell
git clone https://github.com/jasonlin316/ARGO.git
```
Note: Anonymous GitHub does not support ```git clone```, sorry for the inconvenience. 

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
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c dglteam dgl
conda install -c conda-forge ogb
conda install -c conda-forge scikit-optimize
```
Note: there exist a bug in the older version of the Scikit-Optimization library.  
To fix the bug, find the "transformer.py" which should be located in  
   ```~/anaconda3/envs/py38/lib/python3.8/site-packages/skopt/space/transformers.py```  
Once open the file, replace all ```np.int``` with ```int```.

6. Download the OGB datasets (optional if you are not running any)
```
python ogb_example.py --dataset <ogb_dataset>
```
- Available choices [ogbn-products, ogbn-papers100M]  

The program will ask if you want to download the dataset; please enter "y" for the program to proceed. You may terminate the program after the dataset is downloaded.
This extra step is not required for other datasets (e.g., reddit) because they will download automatically. 

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

In this section, we provide a step-by-step tutorial on how to enable ARGO on a DGL program. We use the ```ogb_example.py``` file in this repo as an example.  

Note: we also provide the complete example file ```ogb_example_ARGO.py``` which followed the steps below to enable ARGO on ```ogb_example.py```.

1. First, include all necessary packages on top of the file. Please place your file and ```argo.py``` in the same directory.

   ```
   import os
   import torch.distributed as dist
   from torch.nn.parallel import DistributedDataParallel
   import torch.multiprocessing as mp
   from argo import ARGO
   ```

2. Setup PyTorch Distributed Data Parallel (DDP). 
    1. Add the initialization function on top of the training program, and wrap the ```model``` with the DDP wrapper
     ```
     def train(...):
       dist.init_process_group('gloo', rank=rank, world_size=world_size) # newly added
       model = SAGE(...) # original code
       model = DistributedDataParallel(model) # newly added
       ...
     ```
    2. In the main program, add the following before launching the training function
    
     ```
     os.environ['MASTER_ADDR'] = '127.0.0.1'
     os.environ['MASTER_PORT'] = '29501'
     mp.set_start_method('fork', force=True)
     train(args, device, data) # original code for launching the training function
     ```

3. Enable ARGO by initializing the runtime system, and wrapping the training function
   ```
   runtime = ARGO(n_search = 15, epoch = args.num_epochs, batch_size = args.batch_size) #initialization
   runtime.run(train, args=(args, device, data)) # wrap the training function
   ```
   ARGO takes three input paramters: number of searches ```n_search```, number of epochs, and the mini-batch size. Increasing ```n_search``` potentially leads to a better configuration with less epoch time; however, searching itself also causes extra overhead. We recommend setting ```n_search``` from 15 to 45 for an optimal overall performance. Details of ```n_search``` can be found in the paper.

4. Modify the input of the training function, by directly adding ARGO parameters after the original inputs.
   This is the original function:
   ```
   def train(args, device, data):
   ```
   Add ```rank, world_size, comp_core, load_core, counter, b_size, ep``` like this:
   ```
   def train(args, device, data, rank, world_size, comp_core, load_core, counter, b_size, ep):
   ```

6. Modify the ```dataloader``` function in the training function
   ```
   dataloader = dgl.dataloading.DataLoader(
           g,
           train_nid,
           sampler,
           batch_size=b_size, # modified
           shuffle=True,
           drop_last=False,
           num_workers=len(load_core), # modified
           use_ddp = True) # newly added
   ```

7. Enable core-binding by adding ```enable_cpu_affinity()``` before the training for-loop, and also change the number of epochs into the variable ```ep```: 
   ```
   with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
     for epoch in range(ep): # change num_epochs to ep
   ```

8. Last step! Load the model before training and save it afterward.  
   Original Program:
   ```
   with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
     for epoch in range(ep): 
       ... # training operations
   ```
   Modified:
   ```
   PATH = "model.pt"
   if counter[0] != 0:
     checkpoint = th.load(PATH)
     model.load_state_dict(checkpoint['model_state_dict'])
     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
     epoch = checkpoint['epoch']
     loss = checkpoint['loss']
   
   with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
     for epoch in range(ep): 
       ... # training operations
   
   dist.barrier()
   if rank == 0:
     th.save({'epoch': counter[0],
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss,
                 }, PATH)
   
   ```
10. Done! You can now run your GNN program with ARGO enabled.
   ```
   python -W ignore <Your code>.py
   ```
