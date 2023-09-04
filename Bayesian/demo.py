# To run this demo, you need to rewrite all the 'np.int' in the file '/skopt/space/transformers.py' into 'np.int_' first. 
import numpy as np
from skopt import gp_minimize

# define the acquisition function, can be choose from ['LCB', 'EI', 'PI']
acq_func = 'EI'  

# define the objective function
def objective_function(x):
    x1, x2 = x[0]-2, x[1]
    return -np.sin(x1)/(x1+0.001) - x2


# Define the searching space of the parameters
space = [(0, 10), (0, 10)] 

# Run the Bayesian optimization algorithm, using the Gaussian process as the surrogate model
result = gp_minimize(objective_function, space, n_calls=15, random_state=3, acq_func=acq_func)

# Print the result
print("Best parameter:", result.x)
print("Minimum output:", result.fun)

# Parameters of each iteration
print("Parameters of each iteration:", result.x_iters)
# Output of each iteration
print("Output of each iteration:", result.func_vals)



