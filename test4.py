import numpy as np
from numba import cuda

# 1. Define the structural layout using standard numpy dtype
# This creates fields for coordinates (x, y) and an ID
point_dtype = np.dtype([
    ('id', np.int32),
    ('x', np.float32),
    ('y', np.float32)
])

# 2. Define the CUDA kernel
@cuda.jit
def process_points_kernel(data, multiplier):
    # Calculate global thread index
    idx = cuda.grid(1)
    
    # Boundary check to prevent accessing out-of-bounds memory
    if idx < data.size:
        # Access and modify fields directly by their string names
        data[idx]['x'] *= multiplier
        data[idx]['y'] *= multiplier

# 3. Create host data using the custom dtype
n_elements = 1024
host_data = np.zeros(n_elements, dtype=point_dtype)

# Initialize data for demonstration
for i in range(n_elements):
    host_data[i]['id'] = i
    host_data[i]['x'] = float(i)
    host_data[i]['y'] = float(i * 2)

# 4. Transfer data to the GPU device
device_data = cuda.to_device(host_data)

# 5. Configure execution block and grid dimensions
threads_per_block = 256
blocks_per_grid = (n_elements + (threads_per_block - 1)) // threads_per_block

# 6. Launch the kernel
scale_factor = 2.5
process_points_kernel[blocks_per_grid, threads_per_block](device_data, scale_factor)

# 7. Copy the modified structured data back to the host
host_data_updated = device_data.copy_to_host()

# Quick validation check on the first element
print(f"Original X: {1.0} -> Updated X: {host_data_updated[1]['x']}")