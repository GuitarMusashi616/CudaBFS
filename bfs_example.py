import cupy as cp
from numba import cuda

# 1. The Dynamic Catalyst Generation Kernel
@cuda.jit
def generate_next_frontier(current_frontier, output_arr, masks):
    """
    Each thread takes 1 state from the current frontier and applies
    all 16 mask catalysts to build the raw pool of next potential states.
    """
    idx = cuda.grid(1)
    if idx < current_frontier.size:
        state = current_frontier[idx]
        for i in range(16):
            # Apply catalyst modification
            new_state = state ^ masks[i]
            # Write to sequential global memory slot
            output_arr[idx * 16 + i] = new_state

# 2. Setup initial state space
# Let's start with a single 64-bit seed value
initial_state = 0x0000000000000001
frontier = cp.array([initial_state], dtype=cp.uint64)
visited = cp.array([initial_state], dtype=cp.uint64)

# Define 16 static catalyst masks (flipping different bit positions)
masks = cp.array([1 << i for i in range(16)], dtype=cp.uint64)

print(f"Starting real GPU BFS... Seed state: {hex(initial_state)}")
print("-" * 50)

# 3. True BFS Loop
generation = 1
while frontier.size > 0:
    # Allocate a raw pool for the next generation expansions
    # Every element in the frontier spawns 16 child states
    raw_next_pool = cp.zeros(frontier.size * 16, dtype=cp.uint64)
    
    # Configure grid blocks based on the size of the frontier
    threads_per_block = 256
    blocks = (frontier.size + threads_per_block - 1) // threads_per_block
    
    # Execution: Launch GPU Threads to compute mutations
    generate_next_frontier[blocks, threads_per_block](frontier, raw_next_pool, masks)
    
    # --- THE VISITED TRACKING LOGIC (The Set Difference) ---
    
    # Step A: Filter out duplicates generated *within* this new batch
    unique_next_pool = cp.unique(raw_next_pool)
    
    # Step B: Eliminate states we have already visited in previous generations.
    # cp.in1d returns a boolean mask indicating if elements exist in 'visited'
    already_seen_mask = cp.in1d(unique_next_pool, visited)
    
    # Step C: The new frontier is explicitly elements that are NOT already seen
    frontier = unique_next_pool[~already_seen_mask]
    
    # Step D: Update master visited set with the newly discovered frontier
    if frontier.size > 0:
        visited = cp.concatenate([visited, frontier])
    
    print(f"Gen {generation:02d} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
    
    generation += 1

print("-" * 50)
print(f"BFS Complete! Explored every reachable state combination.")
print(f"Total Unique 64-bit States Discovered: {visited.size}")