# To run this demo, you need to rewrite all the 'np.int' in the file '/skopt/space/transformers.py' into 'np.int_' first. 
import numpy as np
from skopt import gp_minimize
import subprocess
from skopt.plots import plot_convergence
import matplotlib.pyplot as plt
import psutil
import argparse
import time

total_cpu = psutil.cpu_count(logical=False)
# define the acquisition function, can be choose from ['LCB', 'EI', 'PI']
acq_func = 'EI'  

# define the objective function
def objective_function(x):
    n_process = x[0]
    n_sampler = x[1]
    n_trainer = x[2]

    if n_process*(n_sampler+n_trainer) > total_cpu:
        return max_val
    
    command = ["python", "gnn_train.py", "--model", "sage", "--sampler", "neighbor" , "--dataset", arguments.dataset, '--cpu_process', str(int(n_process)), '--n_sampler', str(int(n_sampler)), '--n_trainer', str(int(n_trainer))]
    # print(command)
    try:
        # Execute the external script and capture its output
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
        output_lines = result.split("\n")
        # Search for the line containing "total_time" and extract the value
        for line in output_lines:
            if "total_time" in line:
                objective_value = float(line.split()[1])
                break
        return objective_value
    
    except subprocess.CalledProcessError as e:
        # Handle errors if the external script fails
        print("External script failed with error:", e)
        return max_val



# Define the searching space of the parameters
space = [(2, 8), (1, 4), (1,32)] 

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='ogbn-products',
                    choices=["ogbn-papers100M", "ogbn-products", "mag240M", "reddit", "yelp", "flickr"])

arguments = parser.parse_args()
command = ["python", "gnn_train.py","--model", "sage", "--sampler", "neighbor" , "--dataset", arguments.dataset , '--cpu_process', str(1), '--n_sampler', str(1), '--n_trainer', str(4)]

tik = time.time()
result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
output_lines = result.split("\n")
for line in output_lines:
    if "total_time" in line:
        max_val = float(line.split()[1])
        break
print("upper bound:", max_val)

# Run the Bayesian optimization algorithm, using the Gaussian process as the surrogate model
result = gp_minimize(objective_function, space, n_calls=50, random_state=3, acq_func=acq_func)


with open("bo_{}.txt".format(arguments.dataset), "a") as text_file:
    text_file.write("Best parameter:" + str(result.x) + "\n")
    text_file.write("Minimum output:" + str(result.fun) + "\n")
    text_file.write("Parameters of each iteration:" + str(result.x_iters) + "\n")
    text_file.write("Output of each iteration:" + str(result.func_vals) + "\n")

print('total_time: ',time.time() - tik )

plot_convergence(result)
plt.savefig('convergence_plot_{}.png'.format(arguments.dataset))