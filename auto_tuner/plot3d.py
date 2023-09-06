import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read the CSV file into a pandas DataFrame
csv_file = 'data.csv'  # Replace with your CSV file's path
data = pd.read_csv(csv_file)

# Extract data from columns 'x', 'y', and 'z'
x = data['x']
y = data['y']
z = data['z']

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D surface
surface = ax.plot_trisurf(x, y, z, cmap='viridis')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Surface Plot')

# Add colorbar
fig.colorbar(surface, ax=ax, shrink=0.5, aspect=10)

# Show the plot
plt.show()
