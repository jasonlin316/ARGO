import random
import math
import subprocess
import csv
import argparse
import psutil

# Define the objective function that calls an external script
def objective_function(x, y, z):
    command = ["python", "gnn_train.py", "--model", "sage", "--sampler", "neighbor" , "--dataset", arguments.dataset, '--cpu_process', str(int(x)), '--n_sampler', str(int(y)), '--n_trainer', str(int(z))]
    # print(command)
    if x*(y+z) > total_cpu:
        return max_val
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

# Simulated Annealing algorithm
def simulated_annealing(initial_x, initial_y, initial_z, initial_temperature, cooling_rate, max_iterations):
    current_x, current_y, current_z = initial_x, initial_y, initial_z

    current_value = objective_function(current_x, current_y, current_z)

    best_x, best_y, best_z = current_x, current_y, current_z
    best_value = current_value

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate neighboring integer coordinates
        neighbor_x = random.randint(2, 8)
        neighbor_y = random.randint(1, 4)
        neighbor_z = random.randint(1, 32)

        # Calculate the value for the neighbor point
        neighbor_value = objective_function(neighbor_x, neighbor_y, neighbor_z)

        # Calculate the change in value for the neighbor point
        delta_value = neighbor_value - current_value

        # If the neighbor point is better or accepted with a probability, update the current point
        if delta_value < 0 or random.random() < math.exp(-delta_value / temperature):
            current_x, current_y, current_z = neighbor_x, neighbor_y, neighbor_z
            current_value = neighbor_value

            # Update the best point if necessary
            if current_value < best_value:
                best_x, best_y, best_z = current_x, current_y, current_z
                best_value = current_value

        # Reduce the temperature
        temperature *= cooling_rate
        print(round(temperature,2), round(neighbor_value,2), current_x, current_y, current_z)

    return best_x, best_y, best_z, best_value

# Example usage
if __name__ == "__main__":
    initial_x = 2
    initial_y = 1
    initial_z = 8

    initial_temperature = 100.0
    cooling_rate = 0.95
    max_iterations = 25

    total_cpu = psutil.cpu_count(logical=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type=str,
                        default='ogbn-products',
                        choices=["ogbn-papers100M", "ogbn-products", "mag240M", "reddit", "yelp", "flickr"])
    arguments = parser.parse_args()
    command = ["python", "gnn_train.py", "--model", "sage", "--sampler", "neighbor" , "--dataset", arguments.dataset, '--cpu_process', str(int(1)), '--n_sampler', str(int(2)), '--n_trainer', str(int(8))]
    result = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=600)
    output_lines = result.split("\n")
    # Search for the line containing "total_time" and extract the value
    for line in output_lines:
        if "total_time" in line:
            max_val = float(line.split()[1])
            break

    best_x, best_y, best_z, best_value = simulated_annealing(initial_x, initial_y, initial_z, initial_temperature, cooling_rate, max_iterations)

    print("Best (x, y, z):", (best_x, best_y, best_z))
    print("Best Value:", best_value)
    outp = [arguments.dataset,best_x,best_y,best_value]
