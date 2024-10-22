import numpy as np
import laspy

# Load the .pts file (space-separated values)
pts_data = np.loadtxt('data/Vaihingen/Vaihingen3D_EVAL_WITH_REF.pts')

# Assuming the order of columns in your .pts file:
# X, Y, Z, Intensity, Return Number, Number of Returns
x = pts_data[:, 0]
y = pts_data[:, 1]
z = pts_data[:, 2]
intensity = pts_data[:, 3]
return_number = pts_data[:, 4].astype(int)  # Cast to int if necessary
number_of_returns = pts_data[:, 5].astype(int)

# Define the output LAS file path
las_file = 'data/Vaihingen/vaihingen_eval.las'

# Create a LAS file
header = laspy.LasHeader(point_format=1, version="1.2")  # Use PointFormat 1 for basic LAS data
las = laspy.LasData(header)

# Assign the coordinates and attributes to the LAS file
las.x = x
las.y = y
las.z = z
las.intensity = intensity.astype(np.uint16)  # LAS uses uint16 for intensity
las.return_number = return_number
las.number_of_returns = number_of_returns

# Save the LAS file
las.write(las_file)
print(f"Saved .las file to {las_file}")


