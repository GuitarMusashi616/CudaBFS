import cupy as cp
import numpy as np
import pandas as pd
from numba import cuda
from Stopwatch import Stopwatch
import warnings
from Effect import Effect
from Item import Item


NUM_ITEMS = 16
MAX_SWITCHES_PER_ITEM = 10


def create_d_arrays(items_data: list[Item]):
    always_masks = np.zeros(NUM_ITEMS, dtype=np.uint64)
    switch_counts = np.zeros(NUM_ITEMS, dtype=np.int32)
    switch_keys = np.zeros((NUM_ITEMS, MAX_SWITCHES_PER_ITEM), dtype=np.uint64)
    switch_values = np.zeros((NUM_ITEMS, MAX_SWITCHES_PER_ITEM), dtype=np.uint64)

    # Create rows with alwaysAddEffect and switchEffects for each Item
    for i, item in enumerate(items_data):
        always_masks[i] = item.always_add_effect.bitmask

        switch_counts[i] = len(item.switch_effects)
        
        for j, (key, values) in enumerate(item.switch_effects.items()):
            switch_keys[i, j] = key.bitmask
            
            # Combine multiple values into a single bitmask
            val_mask = 0
            for val in values:
                val_mask |= val.bitmask
            switch_values[i, j] = val_mask
    
    return cuda.to_device(always_masks), \
           cuda.to_device(switch_counts), \
           cuda.to_device(switch_keys), \
           cuda.to_device(switch_values)



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


@cuda.jit
def calculate_multipliers_kernel(states, effect_masks, effect_multipliers, multipliers):
    """Sum the multipliers for every effect bit set in each state."""
    state_index = cuda.grid(1)
    if state_index >= states.size:
        return

    state = states[state_index]
    multiplier = 1.0
    for effect_index in range(effect_masks.size):
        if state & effect_masks[effect_index]:
            multiplier += effect_multipliers[effect_index]
    multipliers[state_index] = multiplier


def calculate_state_multipliers(visited):
    """Return one summed effect multiplier per state in ``visited``.

    The state masks and the resulting scores remain on the GPU, so this only
    requires a single linear kernel launch regardless of how many states BFS
    discovers.
    """
    effects = list(Effect)
    effect_masks = np.asarray([effect.bitmask for effect in effects], dtype=np.uint64)
    effect_multipliers = np.asarray(
        [effect.multiplier for effect in effects], dtype=np.float32
    )
    multipliers = cp.empty(visited.size, dtype=cp.float32)

    threads_per_block = 256
    blocks_per_grid = (visited.size + threads_per_block - 1) // threads_per_block
    calculate_multipliers_kernel[blocks_per_grid, threads_per_block](
        visited,
        cuda.to_device(effect_masks),
        cuda.to_device(effect_multipliers),
        multipliers,
    )
    return multipliers


def show_top_multiplier_states(visited, multipliers, limit=500):
    """Print and return the highest-scoring reachable states.

    Scores are sorted descending on the GPU.  Only the requested leading rows
    are transferred to the host for human-readable output.
    """
    count = min(limit, visited.size)
    if count == 0:
        return pd.DataFrame(columns=["rank", "state_index", "state", "multiplier", "effects"])

    sorted_indices = cp.argsort(multipliers)[::-1]
    top_indices = sorted_indices[:count]

    host_indices = cp.asnumpy(top_indices)
    host_states = cp.asnumpy(visited[top_indices])
    host_multipliers = cp.asnumpy(multipliers[top_indices])
    effects = list(Effect)
    rows = []
    for rank, (state_index, state, multiplier) in enumerate(
        zip(host_indices, host_states, host_multipliers), start=1
    ):
        active_effects = ", ".join(
            effect.name for effect in effects if int(state) & effect.bitmask
        )
        rows.append({
            "rank": rank,
            "state_index": int(state_index),
            "state": hex(int(state)),
            "multiplier": float(multiplier),
            "effects": active_effects,
        })

    top_states = pd.DataFrame(rows)
    # print("\nTop states by summed effect multiplier:")
    # print(top_states.to_string(index=False))
    return top_states


def reconstruct_item_indices(state_index, parent_indices, item_indices):
    """Return the BFS item path to ``state_index`` in forward order.

    ``parent_indices`` and ``item_indices`` are the arrays returned by
    :func:`bfs`.  Parent links are indices into the visited-state array rather
    than duplicate 64-bit masks, so reconstruction only walks the path length.
    """
    path = []
    while state_index != -1:
        item_index = int(item_indices[state_index])
        if item_index != -1:  # The initial state has no incoming item.
            path.append(item_index)
        state_index = int(parent_indices[state_index])
    path.reverse()
    return path


def bfs():
    items = Item.from_json_file()
    d_arys = create_d_arrays(items)

    stopwatch = Stopwatch()
    initial_state = 0
    frontier = cp.array([initial_state], dtype=cp.uint64)
    visited = cp.array([initial_state], dtype=cp.uint64)

    # These are aligned with ``visited``.  A parent is stored as an index into
    # visited instead of another uint64 state, which is both smaller and makes
    # a path reconstruction a direct O(path length) walk.
    frontier_indices = cp.array([0], dtype=cp.int32)
    parent_indices = cp.array([-1], dtype=cp.int32)
    item_indices = cp.array([-1], dtype=cp.int8)

    print(f"Starting real GPU BFS... Seed state: {hex(initial_state)}")
    print("-" * 50)
    
    generation = 1
    while frontier.size > 0:
        # Allocate a raw pool for the next generation expansions
        raw_next_pool = cp.zeros(frontier.size * d_arys[0].size, dtype=cp.uint64)

        # Configure grid blocks based on the size of the frontier
        threads_per_block = (16, 16)
        blocks_per_grid_x = (frontier.size + threads_per_block[0] - 1) // threads_per_block[0]
        grid_dims = (blocks_per_grid_x, 1)

        # Launch kernel
        apply_items_kernel[grid_dims, threads_per_block](
            frontier, 
            d_arys[0],
            d_arys[1],
            d_arys[2],
            d_arys[3],
            raw_next_pool
        )

        # Step A: Keep the first expansion for every child.  Its raw-output
        # index identifies both the parent frontier row and the item used, so
        # provenance remains aligned without a separate search or kernel.
        unique_next_pool, first_output_indices = cp.unique(
            raw_next_pool, return_index=True
        )
        
        # Step B: Eliminate states we have already visited in previous generations.
        # cp.in1d returns a boolean mask indicating if elements exist in 'visited'
        already_seen_mask = cp.in1d(unique_next_pool, visited)
        
        # Step C: The new frontier is explicitly elements that are NOT already seen.
        new_state_mask = ~already_seen_mask
        frontier = unique_next_pool[new_state_mask]
        first_output_indices = first_output_indices[new_state_mask]

        # raw_next_pool is laid out as [frontier state][item].  Convert the
        # selected first output for each child into its parent visited index
        # and the item that created it.
        parent_frontier_rows = first_output_indices // d_arys[0].size
        next_parent_indices = frontier_indices[parent_frontier_rows]
        next_item_indices = (first_output_indices % d_arys[0].size).astype(cp.int8)
        
        # Step D: Update master visited set with the newly discovered frontier
        if frontier.size > 0:
            visited = cp.concatenate([visited, frontier])
            parent_indices = cp.concatenate([parent_indices, next_parent_indices])
            item_indices = cp.concatenate([item_indices, next_item_indices])
            frontier_indices = cp.arange(
                visited.size - frontier.size, visited.size, dtype=cp.int32
            )
        
        print(f"Gen {generation:02d} | Time: {stopwatch.get_time(): 0.2f} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
        # print(f"Gen {generation:02d} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
        
        generation += 1

    print("-" * 50)
    print(f"BFS Complete! Explored every reachable state combination.")
    print(f"Total Unique 64-bit States Discovered: {visited.size}")
    return visited, parent_indices, item_indices


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    visited, _, _ = bfs()
    state_multipliers = calculate_state_multipliers(visited)
    top_states = show_top_multiplier_states(visited, state_multipliers)
