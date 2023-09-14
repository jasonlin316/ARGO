import math

datasets = ['ogbn-products', 'flickr', 'reddit', 'yelp']
cpu_processes = [i for i in range(2, 9)]
n_samplers = [i for i in range(1, 9)] 
n_trainers = [i for i in range(1, 51)]

max_cores = 32

lines = []
for dataset in datasets:
    for cpu_process in cpu_processes:
        for n_sampler in n_samplers:
            if cpu_process * n_sampler > max_cores:
                continue
            max_trainer = math.floor((max_cores - cpu_process * n_sampler) / cpu_process)
            print(max_trainer)
            for n_trainer in n_trainers[:max_trainer]:
                line = f"timeout 5400 python -W ignore PyG/gnn_train2.py --dataset {dataset} "
                line += f"--cpu_process {cpu_process} --n_sampler {n_sampler} --n_trainer {n_trainer}"
                lines.append(line)

with open('PyG/grid_search.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('\n'.join(lines))