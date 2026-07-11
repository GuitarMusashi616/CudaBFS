import cupy as cp
import numpy as np
from numba import cuda
from Item import Item
from Effect import Effect

def get_4_arrays(items_data):
    # --- Data Preparation for CUDA ---
    NUM_ITEMS = len(items_data)
    MAX_SWITCHES_PER_ITEM = max(len(item.switch_effects) for item in items_data)

    # Initialize host arrays
    always_masks = np.zeros(NUM_ITEMS, dtype=np.uint64)
    switch_counts = np.zeros(NUM_ITEMS, dtype=np.int32)
    # 2D arrays of shape (NUM_ITEMS, MAX_SWITCHES_PER_ITEM)
    switch_keys = np.zeros((NUM_ITEMS, MAX_SWITCHES_PER_ITEM), dtype=np.uint64)
    switch_values = np.zeros((NUM_ITEMS, MAX_SWITCHES_PER_ITEM), dtype=np.uint64)

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

    # Move item lookup structures to the GPU device
    d_always_masks = cuda.to_device(always_masks)
    d_switch_counts = cuda.to_device(switch_counts)
    d_switch_keys = cuda.to_device(switch_keys)
    d_switch_values = cuda.to_device(switch_values)

    return d_always_masks, d_switch_counts, d_switch_keys, d_switch_values

def load_cupy_items(filename: str = 'data/items.json'):
    items = Item.from_json_file(filename)
    # d_items = get_cupy_items(items)

    return get_4_arrays(items)


if __name__ == "__main__":
    pass