import cupy as cp
import numpy as np
import pandas as pd
from numba import cuda
from Stopwatch import Stopwatch
import warnings
from Effect import Effect
from Item import Item
import streamlit as st


NUM_ITEMS = 16
MAX_SWITCHES_PER_ITEM = 10
BASE_PRICE = 35.0
BASE_COST = 30.0


def effects_to_state(effects: list[Effect]) -> int:
    """Return the state mask containing every effect in ``effects``.

    Example: ``effects_to_state([Effect.CALMING])``.
    """
    state = 0
    for effect in effects:
        state |= effect.bitmask
    return state


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
    # current_mask |= always_masks[item_idx]
    
    # Initialize the effects to remove and the effects to add
    remove_effects = 0
    add_effects = always_masks[item_idx]

    # 3. Step 2: Process the item's switches sequentially
    num_switches = switch_counts[item_idx]
    for j in range(num_switches):
        key_mask = switch_keys[item_idx, j]
        # Check if the key bit is set
        if (current_mask & key_mask) != 0:
            # Set the key_mask into remove_effects
            # 0001 key_mask
            # 0100 remove_eff
            # 0101 - OR
            remove_effects |= key_mask
            # current_mask |= ~key_mask            # Clear the trigger bit
            # current_mask |= switch_values[item_idx, j] # Set the value bits

            # 1010
            # 0110
            # 1110 - OR= res
            add_effects |= switch_values[item_idx, j] # Set the value bits
    
    # Apply remove_effects to the current_mask first
    # 1101 current_mask
    # 0111 remove_effects
    # 1000 ~remove_effects
    # 1000 result
    current_mask &= ~remove_effects

    # Apply add_effects next
    # 1010 current_mask
    # 1101 add_effects
    # 1111 res
    current_mask |= add_effects
            
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


def calculate_state_multipliers(visited, total_costs, base_price=BASE_PRICE):
    """Return multiplier, price, total cost, and profit for each state.

    ``total_costs`` is aligned with ``visited`` and is built by :func:`bfs`
    from each state's recorded parent and incoming item.  Scoring, pricing,
    and profit calculation all remain on the GPU.
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
    prices = multipliers * np.float32(base_price)
    profits = prices - total_costs
    return multipliers, prices, total_costs, profits


def show_top_profit_states(
    visited, prices, total_costs, profits, parent_indices, item_indices,
    item_counts, items, limit=500, sort_by="profit", ascending=False,
):
    """Return states ordered by a selectable numeric report column.

    Sorting occurs on the GPU; only the selected report rows are transferred to
    the host for path/effect formatting.  Per-item columns exclude the seed
    state (which has zero items) by placing its undefined values last.
    """
    count = min(limit, visited.size)
    if count == 0:
        return pd.DataFrame(columns=[
            "price", "profit", "total_cost", "price_per_item",
            "profit_per_item", "item_order", "effects",
        ])

    item_counts_float = item_counts.astype(cp.float32)
    # CuPy ufuncs do not consistently support NumPy's ``where=`` keyword.
    # Substitute a safe denominator first, then explicitly restore NaN for
    # the seed state, which has zero applied items.
    safe_item_counts = cp.where(item_counts == 0, 1.0, item_counts_float)
    price_per_item = cp.where(
        item_counts == 0, cp.nan, prices / safe_item_counts
    )
    profit_per_item = cp.where(
        item_counts == 0, cp.nan, profits / safe_item_counts
    )
    sortable_columns = {
        "price": prices,
        "profit": profits,
        "total_cost": total_costs,
        "price_per_item": price_per_item,
        "profit_per_item": profit_per_item,
    }
    if sort_by not in sortable_columns:
        valid_columns = ", ".join(sortable_columns)
        raise ValueError(f"sort_by must be one of: {valid_columns}")

    sort_values = sortable_columns[sort_by]
    # CuPy's NaN ordering is backend-dependent, so use an explicit sentinel
    # that places undefined zero-item values after all valid states.
    nan_last_value = cp.inf if ascending else -cp.inf
    sort_values = cp.where(cp.isnan(sort_values), nan_last_value, sort_values)
    sorted_indices = cp.argsort(sort_values)
    if not ascending:
        sorted_indices = sorted_indices[::-1]
    top_indices = sorted_indices[:count]

    host_indices = cp.asnumpy(top_indices)
    host_states = cp.asnumpy(visited[top_indices])
    host_prices = cp.asnumpy(prices[top_indices])
    host_costs = cp.asnumpy(total_costs[top_indices])
    host_profits = cp.asnumpy(profits[top_indices])
    host_item_counts = cp.asnumpy(item_counts[top_indices])
    effects = list(Effect)
    rows = []
    for state_index, state, price, total_cost, profit, item_count in zip(
        host_indices, host_states, host_prices, host_costs, host_profits,
        host_item_counts,
    ):
        active_effects = ", ".join(
            effect.name for effect in effects if int(state) & effect.bitmask
        )
        path = reconstruct_item_indices(state_index, parent_indices, item_indices)
        price_per_item = float(price) / item_count if item_count else np.nan
        profit_per_item = float(profit) / item_count if item_count else np.nan
        rows.append({
            "price": float(price),
            "profit": float(profit),
            "total_cost": float(total_cost),
            "price_per_item": price_per_item,
            "profit_per_item": profit_per_item,
            "item_order": " -> ".join(items[item_index].name for item_index in path),
            "effects": active_effects,
        })

    return pd.DataFrame(rows)


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


def bfs(initial_state=0, base_cost=BASE_COST, num_gens=12):
    """Explore states reachable from ``initial_state`` using ``base_cost``."""
    items = Item.from_json_file()
    d_arys = create_d_arrays(items)
    item_costs = cp.asarray([item.cost for item in items], dtype=cp.float32)

    stopwatch = Stopwatch()
    frontier = cp.array([initial_state], dtype=cp.uint64)
    visited = cp.array([initial_state], dtype=cp.uint64)

    # These are aligned with ``visited``.  A parent is stored as an index into
    # visited instead of another uint64 state, which is both smaller and makes
    # a path reconstruction a direct O(path length) walk.
    frontier_indices = cp.array([0], dtype=cp.int32)
    parent_indices = cp.array([-1], dtype=cp.int32)
    item_indices = cp.array([-1], dtype=cp.int8)
    item_counts = cp.array([0], dtype=cp.int32)
    # Costs use the same visited-array alignment as the parent/item links.
    total_costs = cp.array([base_cost], dtype=cp.float32)

    print(f"Starting real GPU BFS... Seed state: {hex(initial_state)}")
    print("-" * 50)
    
    generation = 1
    while frontier.size > 0 and generation <= num_gens:
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
        next_total_costs = total_costs[next_parent_indices] + item_costs[next_item_indices]
        next_item_counts = item_counts[next_parent_indices] + 1
        
        # Step D: Update master visited set with the newly discovered frontier
        if frontier.size > 0:
            visited = cp.concatenate([visited, frontier])
            parent_indices = cp.concatenate([parent_indices, next_parent_indices])
            item_indices = cp.concatenate([item_indices, next_item_indices])
            total_costs = cp.concatenate([total_costs, next_total_costs])
            item_counts = cp.concatenate([item_counts, next_item_counts])
            frontier_indices = cp.arange(
                visited.size - frontier.size, visited.size, dtype=cp.int32
            )
        
        print(f"Gen {generation:02d} | Time: {stopwatch.get_time(): 0.2f} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
        # print(f"Gen {generation:02d} | Unique Pool: {unique_next_pool.size:<6} | New Frontier: {frontier.size:<6} | Total Global States Mastered: {visited.size}")
        
        generation += 1

    print("-" * 50)
    print(f"BFS Complete! Explored every reachable state combination.")
    print(f"Total Unique 64-bit States Discovered: {visited.size}")
    return visited, parent_indices, item_indices, total_costs, item_counts, items


def run(
    base_price=BASE_PRICE, base_cost=BASE_COST, initial_state=0, limit=500,
    sort_by="profit", ascending=False,
):
    """Run BFS and return the state DataFrame ordered by ``sort_by``.

    ``initial_state`` can be built with :func:`effects_to_state`, e.g.
    ``run(initial_state=effects_to_state([Effect.CALMING]))``.
    """
    visited, parent_indices, item_indices, total_costs, item_counts, items = bfs(
        initial_state=initial_state, base_cost=base_cost
    )
    _, prices, total_costs, profits = calculate_state_multipliers(
        visited, total_costs, base_price=base_price
    )
    return show_top_profit_states(
        visited, prices, total_costs, profits, parent_indices, item_indices,
        item_counts, items, limit=limit, sort_by=sort_by, ascending=ascending,
    )

class Price:
    WEED = 35
    METH = 70

class Cost:
    OG_KUSH = 30
    SOUR_DIESEL = 35
    GREEN_CRACK = 40
    GRANDDADDY_PURPLE = 45
    # 60, 80, 110 for pseudo low med high 10x
    # pseudo, phosphorus, acid
    METH = 8 + 40 + 40


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    top_states = run(
        base_cost = Cost.OG_KUSH,
        base_price = Price.WEED,
        initial_state = effects_to_state([Effect.CALMING]),
        limit = 500,
        sort_by="price",
        ascending=False,
    )
    top_states.to_csv('output/og_kush_top_500_price_new.csv')

    # top_states = run(
    #     base_cost = Cost.SOUR_DIESEL,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.REFRESHING]),
    #     limit = 500,
    #     sort_by="price",
    #     ascending=False,
    # )
    # top_states.to_csv('output/sour_diesel_top_500_price.csv')

    # top_states = run(
    #     base_cost = Cost.GREEN_CRACK,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.ENERGIZING]),
    #     limit = 500,
    #     sort_by="price",
    #     ascending=False,
    # )
    # top_states.to_csv('output/green_crack_top_500_price.csv')

    # top_states = run(
    #     base_cost = Cost.GRANDDADDY_PURPLE,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.SEDATING]),
    #     limit = 500,
    #     sort_by="price",
    #     ascending=False,
    # )
    # top_states.to_csv('output/granddaddy_purple_top_500_price.csv')

    # top_states = run(
    #     base_cost = Cost.METH,
    #     base_price = Price.METH,
    #     initial_state = 0,
    #     limit = 500,
    #     sort_by="price",
    #     ascending=False,
    # )
    # top_states.to_csv('output/meth_top_500_price.csv')

    # top_states = run(
    #     base_cost = Cost.OG_KUSH,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.CALMING]),
    #     limit = 500,
    #     sort_by="profit",
    #     ascending=False,
    # )
    # top_states.to_csv('output/og_kush_top_500_profit.csv')

    # top_states = run(
    #     base_cost = Cost.SOUR_DIESEL,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.REFRESHING]),
    #     limit = 500,
    #     sort_by="profit",
    #     ascending=False,
    # )
    # top_states.to_csv('output/sour_diesel_top_500_profit.csv')

    # top_states = run(
    #     base_cost = Cost.GREEN_CRACK,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.ENERGIZING]),
    #     limit = 500,
    #     sort_by="profit",
    #     ascending=False,
    # )
    # top_states.to_csv('output/green_crack_top_500_profit.csv')

    # top_states = run(
    #     base_cost = Cost.GRANDDADDY_PURPLE,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.SEDATING]),
    #     limit = 500,
    #     sort_by="profit",
    #     ascending=False,
    # )
    # top_states.to_csv('output/granddaddy_purple_top_500_profit.csv')

    # top_states = run(
    #     base_cost = Cost.METH,
    #     base_price = Price.METH,
    #     initial_state = 0,
    #     limit = 500,
    #     sort_by="profit",
    #     ascending=False,
    # )
    # top_states.to_csv('output/meth_top_500_profit.csv')

    top_states = run(
        base_cost = Cost.OG_KUSH,
        base_price = Price.WEED,
        initial_state = effects_to_state([Effect.CALMING]),
        limit = 500,
        sort_by="profit_per_item",
        ascending=False,
    )
    top_states.to_csv('output/og_kush_top_500_profit_per_item_new.csv')

    # top_states = run(
    #     base_cost = Cost.SOUR_DIESEL,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.REFRESHING]),
    #     limit = 500,
    #     sort_by="profit_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/sour_diesel_top_500_profit_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.GREEN_CRACK,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.ENERGIZING]),
    #     limit = 500,
    #     sort_by="profit_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/green_crack_top_500_profit_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.GRANDDADDY_PURPLE,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.SEDATING]),
    #     limit = 500,
    #     sort_by="profit_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/granddaddy_purple_top_500_profit_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.METH,
    #     base_price = Price.METH,
    #     initial_state = 0,
    #     limit = 500,
    #     sort_by="profit_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/meth_top_500_profit_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.OG_KUSH,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.CALMING]),
    #     limit = 500,
    #     sort_by="price_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/og_kush_top_500_price_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.SOUR_DIESEL,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.REFRESHING]),
    #     limit = 500,
    #     sort_by="price_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/sour_diesel_top_500_price_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.GREEN_CRACK,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.ENERGIZING]),
    #     limit = 500,
    #     sort_by="price_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/green_crack_top_500_price_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.GRANDDADDY_PURPLE,
    #     base_price = Price.WEED,
    #     initial_state = effects_to_state([Effect.SEDATING]),
    #     limit = 500,
    #     sort_by="price_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/granddaddy_purple_top_500_price_per_item.csv')

    # top_states = run(
    #     base_cost = Cost.METH,
    #     base_price = Price.METH,
    #     initial_state = 0,
    #     limit = 500,
    #     sort_by="price_per_item",
    #     ascending=False,
    # )
    # top_states.to_csv('output/meth_top_500_price_per_item.csv')

    # print(top_states.to_string(index=False))
    # top_states = pd.read_csv('output/meth_top_500_profit.csv', index_col=0)
    # st.dataframe(top_states)
