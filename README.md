# DDP GNN

## Installation

1. Clone the repository:

```shell
git clone https://github.com/jasonlin316/DDP_GNN.git
```

2. Navigate to the project directory:

```shell
cd DDP_GNN
```

3. Download Anaconda and install
```shell
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh
bash Anaconda3-2023.03-Linux-x86_64.sh
```

3. Create a virtual environment:

```shell
conda create -n myenv python=3.8
```

4. Active the virtual environment:

```shell
conda activate myenv
```

5. Install the required packages:

```shell
conda install --file requirements.txt
```

6. Run the experiments

```shell
bash prod.sh
bash cite.sh
```
Note: when running the program for the first time, the program will ask if you want to download the dataset; please enter "y" for the program to proceed.

7. Obtain the four outputs:
```shell
DGL_products.txt
DGL_DDP_products.txt
DGL_citation.txt
DGL_DDP_citation.txt
```

