import cupy as cp
import numpy as np
from numba import cuda
from Stopwatch import Stopwatch
import warnings
from Item import Item


NUM_ITEMS = 16
MAX_SWITCHES_PER_ITEM = 10

always_masks = np.zeros(NUM_ITEMS, dtype=np.uint64)
switch_counts = np.zeros(NUM_ITEMS, dtype=np.int32)
switch_keys = np.zeros((NUM_ITEMS, MAX_SWITCHES_PER_ITEM), dtype=np.uint64)
switch_values = np.zeros((NUM_ITEMS, MAX_SWITCHES_PER_ITEM), dtype=np.uint64)


def modify_4_arrays(items_data):
    global always_masks, switch_counts, switch_keys, switch_values

    # Populate these arrays using your `items_gem.json` data and `effect_to_bitmask` dict
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


@cuda.jit
def apply_items_kernel(frontier, output):
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
            current_mask &= ~key_mask                    # Clear the trigger bit
            current_mask |= switch_values[item_idx, j]    # Set the value bits

    # 4. Map back to the flattened 1D output array (Size: len(frontier) * 16)
    output_idx = state_idx * 16 + item_idx
    output[output_idx] = current_mask


def _grow_buffer(buf, needed_size, dtype, preserve_size=0):
    """
    Return a buffer with capacity >= needed_size.

    If the existing buffer already has enough capacity, it is returned
    unchanged (no allocation). Otherwise a new, larger buffer is allocated
    (capacity doubles, or grows to fit needed_size if that's bigger) and,
    if preserve_size > 0, the first `preserve_size` elements of the old
    buffer are copied into the new one.

    This gives amortized O(1) growth instead of reallocating/copying on
    every single generation.
    """
    if buf is not None and buf.size >= needed_size:
        return buf

    current_cap = buf.size if buf is not None else 0
    new_cap = max(needed_size, current_cap * 2 if current_cap > 0 else needed_size)

    new_buf = cp.empty(new_cap, dtype=dtype)
    if buf is not None and preserve_size > 0:
        new_buf[:preserve_size] = buf[:preserve_size]

    return new_buf


def bfs():
    global always_masks, switch_counts, switch_keys, switch_values

    stopwatch = Stopwatch()
    initial_state = 0
    frontier = cp.array([initial_state], dtype=cp.uint64)

    # --- Pre-allocated "visited" dynamic array -----------------------------
    # Instead of reallocating/copying the entire `visited` array every
    # generation (cp.concatenate), we keep one growable buffer plus a size
    # counter and only reallocate (doubling capacity) when we actually run
    # out of room. Appending becomes a plain slice-assignment (no copy of
    # existing data) in the common case.
    visited_capacity = 1024
    visited_buf = cp.empty(visited_capacity, dtype=cp.uint64)
    visited_buf[0] = initial_state
    visited_size = 1

    # --- Pre-allocated scratch buffer for kernel output ---------------------
    # Reused across generations; only grows (doubling) when the current
    # generation's frontier needs more space than we currently have.
    raw_next_pool_buf = None

    print(f"Starting real GPU BFS (fast)... Seed state: {hex(initial_state)}")
    print("-" * 50)

    generation = 1
    while frontier.size > 0:
        needed_size = frontier.size * always_masks.size

        raw_next_pool_buf = _grow_buffer(raw_next_pool_buf, needed_size, cp.uint64)
        raw_next_pool = raw_next_pool_buf[:needed_size]

        # Configure grid blocks based on the size of the frontier
        threads_per_block = (16, 16)
        blocks_per_grid_x = (frontier.size + threads_per_block[0] - 1) // threads_per_block[0]
        grid_dims = (blocks_per_grid_x, 1)

        # Launch kernel (writes every slot of raw_next_pool exactly once,
        # so no need to zero-initialize it)
        apply_items_kernel[grid_dims, threads_per_block](
            frontier,
            raw_next_pool
        )

        # Step A: Filter out duplicates generated *within* this new batch
        unique_next_pool = cp.unique(raw_next_pool)

        # Step B: Eliminate states we have already visited in previous generations.
        already_seen_mask = cp.in1d(unique_next_pool, visited_buf[:visited_size])

        # Step C: The new frontier is explicitly elements that are NOT already seen
        frontier = unique_next_pool[~already_seen_mask]

        # Step D: Append the newly discovered frontier into the visited buffer,
        # growing it (doubling) only when necessary.
        if frontier.size > 0:
            new_visited_size = visited_size + frontier.size
            visited_buf = _grow_buffer(
                visited_buf, new_visited_size, cp.uint64, preserve_size=visited_size
            )
            visited_buf[visited_size:new_visited_size] = frontier
            visited_size = new_visited_size

        print(f"Gen {generation:02d} | Time: {stopwatch.get_time(): 0.2f} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited_size}")

        generation += 1

    print("-" * 50)
    print(f"BFS Complete! Explored every reachable state combination.")
    print(f"Total Unique 64-bit States Discovered: {visited_size}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    items = Item.from_json_file()
    modify_4_arrays(items)

    bfs()