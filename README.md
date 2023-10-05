# ARGO: An Auto-Tuning Runtime System for Scalable GNN Training on Multi-Core Processor

This README includes how to (1) set up the environment, (2) run the example code, and (3) modify your own GNN program to enable ARGO.  
While we use the Deep Graph Library (DGL) as an example here, ARGO is also compatible with PyTorch-Geometric and details can be found in the PyG folder.

## Setting up the environment

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
conda install -c conda-forge matplotlib
```

## Running the example GNN program

Note: when running the program for the first time, the program will ask if you want to download the dataset; please enter "y" for the program to proceed.


## Enabling ARGO on your own GNN program
   
