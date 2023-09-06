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
conda create -n py38 python=3.8.1
```

4. Active the virtual environment:

```shell
conda activate py38
```

5. Add the required Channels:

```shell
conda config --add channels conda-forge
conda config --add channels pytorch
conda config --add channels dglteam
conda config --add channels pyg
```

6. Install the required packages:

```shell
conda install --file requirements.txt
```

7. Fix a bug in the Scikit-Optimization library. Find "transformer.py" which should locate in
   ```~/anaconda3/envs/py38/lib/python3.8/site-packages/skopt/space/transformers.py```  
   Once open the file, replace all ```np.int``` with ```int``` (there are two of them, located in line 262 and line 275)

8. Try to run the experiment once to make sure the program runs correctly:  
   ```
   python -W ignore gnn_train.py --dataset ogbn-products --cpu_process 2 --n_sampler 2 --n_trainer 4
   ```  
   If we can see "total_time: (some number)" then the program executed successfully.  
   Note: when running the program for the first time, the program will ask if you want to download the dataset; please enter "y" for the program to proceed.
   
10. Run the exhaustive search (this can take up to a day)

```shell
bash verify.sh
```
Afterwards, a .csv file named "grid_serach_{dataset_name}.csv" will be generated for each dataset.

10. Next, we can run the auto-tuner to see what configuration it finds:
    
```shell
bash bo.sh
```

11. Two outputs will be generated for each dataset
```shell
convergence_plot_(dataset).png
bo_(dataset).txt
```

