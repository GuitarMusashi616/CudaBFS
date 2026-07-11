import cupy as cp
import numpy as np
from numba import cuda
from frontier import load_cupy_items
from Stopwatch import Stopwatch
import warnings


@cuda.jit
def apply_items_kernel(frontier, always_masks, switch_counts, switch_keys, switch_values, output):
    # Get global position of the thread
    state_idx, item_idx = cuda.grid(2)
    
    # Boundary checks
    if state_idx >= frontier.shape[0] or item_idx >= always_masks.shape[0]:
        return
        
    # 1. Load the initial EffectState mask
    current_mask = frontier[state_idx]
    
    # 2. Step 1: Always add the item's base effect
    current_mask |= always_masks[item_idx]
    
    # 3. Step 2: Process the item's switches sequentially
    num_switches = switch_counts[item_idx]
    for j in range(num_switches):
        key_mask = switch_keys[item_idx, j]
        # Check if the key bit is set
        if (current_mask & key_mask) != 0:
            current_mask &= ~key_mask            # Clear the trigger bit
            current_mask |= switch_values[item_idx, j] # Set the value bits
            
    # 4. Map back to the flattened 1D output array (Size: len(frontier) * 16)
    output_idx = state_idx * 16 + item_idx
    output[output_idx] = current_mask     


def bfs():
    # initial_state = 0x0000000000000000
    stopwatch = Stopwatch()
    initial_state = 0
    frontier = cp.array([initial_state], dtype=cp.uint64)
    visited = cp.array([initial_state], dtype=cp.uint64)

    d_always_masks, \
        d_switch_counts, \
        d_switch_keys, \
        d_switch_values = load_cupy_items()

    print(f"Starting real GPU BFS... Seed state: {hex(initial_state)}")
    print("-" * 50)
    
    generation = 1
    while frontier.size > 0:
        # Allocate a raw pool for the next generation expansions
        raw_next_pool = cp.zeros(frontier.size * d_always_masks.size, dtype=cp.uint64)

        # Configure grid blocks based on the size of the frontier
        threads_per_block = (16, 16)
        blocks_per_grid_x = (frontier.size + threads_per_block[0] - 1) // threads_per_block[0]
        grid_dims = (blocks_per_grid_x, 1)

        # Launch kernel
        apply_items_kernel[grid_dims, threads_per_block](
            frontier, 
            d_always_masks, 
            d_switch_counts, 
            d_switch_keys, 
            d_switch_values, 
            raw_next_pool
        )

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
        
        print(f"Gen {generation:02d} | Time: {stopwatch.get_time(): 0.2f} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
        # print(f"Gen {generation:02d} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
        
        generation += 1

    print("-" * 50)
    print(f"BFS Complete! Explored every reachable state combination.")
    print(f"Total Unique 64-bit States Discovered: {visited.size}")




def main():
    # Assuming frontier size is N
    N = 10_000 
    d_frontier = cp.random.randint(0, 2**32, size=N, dtype=cp.uint64)
    d_output = cp.zeros(N * 16, dtype=cp.uint64)

    d_always_masks, \
        d_switch_counts, \
        d_switch_keys, \
        d_switch_values = load_cupy_items()

    # Execution configurations
    threads_per_block = (16, 16)
    blocks_per_grid_x = (N + threads_per_block[0] - 1) // threads_per_block[0]
    grid_dims = (blocks_per_grid_x, 1)

    # Launch kernel
    apply_items_kernel[grid_dims, threads_per_block](
        d_frontier, 
        d_always_masks, 
        d_switch_counts, 
        d_switch_keys, 
        d_switch_values, 
        d_output
    )

    print(d_output)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    
    bfs()