# ARGO for PyTorch-Geometric

DGL and PyG can both use ARGO, but their setup is slightly different.

This README includes how to:

1. [Set up the PyG environment](#1-setting-up-the-environment)
2. [Run the example code](#2-running-the-example-PyG-GNN-program)
3. [Modify your own PyG-GNN program to enable ARGO.](#3-enable-ARGO-on-your-own-PyG-GNN-program)

## 1. Setting up the environment

> The steps in bold are different from the DGL steps.

1. Clone the repository:

   ```shell
   git clone https://github.com/jasonlin316/ARGO.git
   cd ARGO
   ```

   Note: Anonymous GitHub does not support ```git clone```, sorry for the inconvenience. 

2. Download Anaconda and install

   ```shell
   wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
   bash Anaconda3-2023.03-Linux-x86_64.sh
   ```

3. **Create a virtual environment called GNN:**

   ```shell
   conda env create -f PyG/environment.yml
   ```

4. **Active the virtual environment:**

   ```shell
   conda activate GNN
   ```

5. Note: there exists a bug in the older version (before v0.9.0) of the Scikit-Optimization library. 
   To fix the bug, find the "transformer.py" which should be located in  
   ```~/anaconda3/envs/py38/lib/python3.8/site-packages/skopt/space/transformers.py```  
   Once open the file, replace all ```np.int``` with ```int```.

6. Download the OGB datasets (optional if you are not running any)

   ```shell
   python ogb_example.py --dataset <ogb_dataset>
   ```

- Available choices [ogbn-products, ogbn-papers100M]  

The program will ask if you want to download the dataset; please enter "y" for the program to proceed. You may terminate the program after the dataset is downloaded.
This extra step is not required for other datasets (e.g., reddit) because they will download automatically. 

## 2. Running the example PyG-GNN program

### Usage

  ```shell
python PyG/main.py --dataset ogbn-products --sampler shadow --model sage
  ```

  Important Arguments: 

  - `--dataset`: the training datasets. Available choices [ogbn-products, ogbn-papers100M, reddit, flickr, yelp]
  - `--sampler`: the mini-batch sampling algorithm. Available choices [shadow, neighbor]
  - `--model`: GNN model. Available choices [gcn, sage]
  - `--layer`: number of GNN layers.
  - `--hidden`: hidden feature dimension.
  - `--batch_size`: the size of the mini-batch.

>  Note: the default number of layer is 3. If you want to change the number of layers for the Neighbor Sampler, please update the sample size in ```line 137 of PyG/main.py```.



## 3. Enable ARGO on your own PyG-GNN program

In this section, we provide a step-by-step tutorial on how to enable ARGO on a PyG program. We use the ```flickr_example.py``` file in `PyG` folder as an example.  

>  Note: we also provide the complete example file ```flickr_example_ARGO.py``` which followed the steps below to enable ARGO on ```flickr_example.py```.

1. First, include all necessary packages on top of the file. Please place your file and ```argo.py``` in the same directory.

   ```python
   import os
   import subprocess
   import torch.distributed as dist
   from torch.utils.data.distributed import DistributedSampler
   from torch.nn.parallel import DistributedDataParallel
   import torch.multiprocessing as mp
   from argo import ARGO
   ```

2.  Add `get_mask` function for `taskset` core binding

      ```python
      def get_mask(comp_core):
          mask = 0
          for core in comp_core:
              mask += 2 ** core
          return hex(mask)
      ```

3. Setup PyTorch Distributed Data Parallel (DDP). 

   1. Add the initialization function on top of the training program, and wrap the ```model``` with the DDP wrapper

    ```python
   def train(...):
     dist.init_process_group('gloo', rank=rank, world_size=world_size) # newly added
     model = SAGE(...) # original code
     model = DistributedDataParallel(model) # newly added
     ...
    ```

   2. In the main program, add the following before launching the training function

    ```python
   os.environ['MASTER_ADDR'] = '127.0.0.1'
   os.environ['MASTER_PORT'] = '29501'
   mp.set_start_method('fork', force=True)
   train(args, device, data) # original code for launching the training function
    ```

4. Enable ARGO by initializing the runtime system, and wrapping the training function

   ```python
   runtime = ARGO(n_search = 15, epoch = args.num_epochs, batch_size = args.batch_size) #initialization
   runtime.run(train, args=(args, device, data)) # wrap the training function
   ```

>  ARGO takes three input paramters: number of searches ```n_search```, number of epochs, and the mini-batch size. Increasing ```n_search``` potentially leads to a better configuration with less epoch time; however, searching itself also causes extra overhead. We recommend setting ```n_search``` from 15 to 45 for an optimal overall performance. Details of ```n_search``` can be found in the paper.

5. Modify the input of the training function, by directly adding ARGO parameters after the original inputs.
   This is the original function:

   ```python
   def train(args, device, data):
   ```

   Add ```rank, world_size, comp_core, load_core, counter, b_size, ep``` like this:

   ```python
   def train(args, device, data, rank, world_size, comp_core, load_core, counter, b_size, ep):
   ```

6. Modify the ```dataloader``` function in the training function

   ```python
   # modify the dataloader
       # Add DistributedSampler for multi-thread data loading
       train_sampler = DistributedSampler(
               train_idx,
               num_replicas = world_size,
               rank=rank
           ) 
       train_loader = NeighborLoader(
           data,
           input_nodes = train_idx,
           num_neighbors=[15, 10, 5],
           batch_size=b_size//world_size, # modified
           num_workers=len(load_core), # modified
           persistent_workers=True,
           sampler = train_sampler # newly added
       )
   ```

7. Change the number of epochs into the variable ```ep```: 

   ```python
   for epoch in range(ep): # change num_epochs to ep
   ```

   

8. Use `taskset` to bind trainer core

   ```python
   torch.set_num_threads(len(comp_core))
       pid = os.getpid()
       core_mask = get_mask(comp_core)
       subprocess.run(["taskset", "-a","-p", str(core_mask), str(pid)])
   ```

9. Set loader core affinity by adding ```enable_cpu_affinity()``` before the training for-loop:

   ```python
   with dataloader.enable_cpu_affinity(loader_cores=load_core, compute_cores=comp_core): 
     for epoch in range(ep): # change num_epochs to ep
   ```

   > Method `enable_cpu_affinity`  and `DistributedSampler` is only available for `NeighborLoader` in PyG. If you want to use other types of data loader, you can only bind trainer cores.  

10. Last step! Load the model before training and save it afterward.  
    Original Program:

    ```python
    # Step 7. Change the number of epochs
        for epoch in range(ep): # change num_epochs to ep
            total_loss = total_correct = total_cnt =  0
            
            # Step 9. Set loader cores affinity
            with train_loader.enable_cpu_affinity(loader_cores = load_core): # set loader cores
        ... # training operations
    ```

    Modified:

    ```python
    # Step 10. Load the model before training and save it afterward
        PATH = "PyG/model.pt"
        if counter[0] != 0:
            checkpoint = torch.load(PATH)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']
    
        # Step 7. Change the number of epochs
        for epoch in range(ep): # change num_epochs to ep
            total_loss = total_correct = total_cnt =  0
            
            # Step 9. Set loader cores affinity
            with train_loader.enable_cpu_affinity(loader_cores = load_core): # set loader cores
                for batch in train_loader:
                   ... # training operations
                
        dist.barrier()
        if rank == 0:
            torch.save({'epoch': counter[0],
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, PATH)
    
    ```

11. Done! You can now run your GNN program with ARGO enabled.

      ```shell
    python -W ignore PyG/<your_code>.py
      ```
