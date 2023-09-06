import random
import math
import subprocess
import csv
# Define the objective function that calls an external script
def exec(x, y):
    # Define the command to execute the external script with inputs x and y
    command = ["python", "-W ignore" , "gnn_train.py", '--dataset',  "flickr" , '--cpu_process', str(x), '--n_sampler', str(y)]
    try:
        # Execute the external script and capture its output
        output = subprocess.check_output(command, stderr=subprocess.STDOUT, universal_newlines=True, timeout=200)
        # Parse the output to obtain the objective value (replace this with your parsing logic)
        objective_value = float(output.strip())
        return objective_value
    except subprocess.CalledProcessError as e:
        # Handle errors if the external script fails
        print("External script failed with error:", e)
        return float("inf")  # Return a infinite value as an error indicator

# Define the objective function that calls an external script
def objective_function(x, y):
    if 2 <= x <= 4 and 1 <= y <= 4:
        return exec(x, y)
    elif 5 <= x <= 8 and 1 <= y <= 2:
        return exec(x, y)
    else:
        # Return a very high value for out-of-range inputs
        return float("inf")  # Return positive infinity for out-of-range inputs

# Simulated Annealing algorithm
def simulated_annealing(initial_x, initial_y, initial_temperature, cooling_rate, max_iterations):
    current_x, current_y = initial_x, initial_y
    current_value = objective_function(current_x, current_y)
    best_x, best_y = current_x, current_y
    best_value = current_value

    temperature = initial_temperature

    for iteration in range(max_iterations):
        # Generate neighboring integer coordinates
        neighbor_x = random.randint(1, 8)
        if 1 <= neighbor_x <= 4:
            neighbor_y = random.randint(1, 4)
        elif 5 <= neighbor_x <= 8:
            neighbor_y = random.randint(1, 2)

        # Calculate the value for the neighbor point
        neighbor_value = objective_function(neighbor_x, neighbor_y)

        # Calculate the change in value for the neighbor point
        delta_value = neighbor_value - current_value

        # If the neighbor point is better or accepted with a probability, update the current point
        if delta_value < 0 or random.random() < math.exp(-delta_value / temperature):
            current_x, current_y = neighbor_x, neighbor_y
            current_value = neighbor_value

            # Update the best point if necessary
            if current_value < best_value:
                best_x, best_y = current_x, current_y
                best_value = current_value

        # Reduce the temperature
        temperature *= cooling_rate
        print(round(temperature,2), round(neighbor_value,2), current_x, current_y)

    return best_x, best_y, best_value

# Example usage
if __name__ == "__main__":
    initial_x = 4
    initial_y = 1
    # if 2 <= initial_x <= 4:
    #     initial_y = random.randint(1, 4)
    # elif 5 <= initial_x <= 8:
    #     initial_y = random.randint(1, 2)

    initial_temperature = 100.0
    cooling_rate = 0.9
    max_iterations = 50

    best_x, best_y, best_value = simulated_annealing(initial_x, initial_y, initial_temperature, cooling_rate, max_iterations)

    print("Best (x, y):", (best_x, best_y))
    print("Best Value:", best_value)
    outp = ['flickr',best_x,best_y,best_value]
    with open('search.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(outp)
