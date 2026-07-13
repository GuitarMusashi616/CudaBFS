# CUDA BFS State Explorer

This project explores every combination of products available in the game Schedule I. 
It explores every effect state reachable from the item definitions in
`data/items.json` using a CUDA-accelerated breadth-first search (BFS).  It
calculates prices and profits for each discovered state, writes ranked CSV
reports to `output/`, and includes a Streamlit app for browsing or comparing
those reports.

## Prerequisites

- **Python 3.10 or newer** (the project is currently used with Python 3.11).
- A CUDA-capable **NVIDIA GPU** and a current NVIDIA GPU driver.
- A CUDA installation supported by your driver.  The GPU execution path uses
  both CuPy and Numba's CUDA support.  Install the NVIDIA CUDA Toolkit if your
  Numba/CUDA setup requires it, and ensure that `nvidia-smi` works in a
  terminal before running the program.
- `pip` (or another Python package installer).

> This is a GPU/CUDA project: running `main.py` requires a usable NVIDIA CUDA
> environment. It will not run the BFS on a CPU-only machine without code
> changes.

## Installation

From the repository root, create and activate a virtual environment (optional
but recommended):

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Windows (Command Prompt)

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
```

Install the Python packages:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` selects `cupy-cuda12x`, which is appropriate for CUDA 12.x.
If your driver/toolkit supports CUDA 11 instead, replace `cupy-cuda12x` in that
file with `cupy-cuda11x` before installing.  Do **not** install the generic
`cupy` package in addition to a CUDA-specific CuPy package.

You can verify that Python can see the GPU with:

```bash
python -c "from numba import cuda; import cupy as cp; print(cuda.gpus); print(cp.cuda.runtime.getDeviceProperties(0)['name'])"
```

## Generate the CSV reports

`main.py` invokes `run()` for all configured starting states and report sorts:

- OG Kush / Calming
- Sour Diesel / Refreshing
- Green Crack / Energizing
- Granddaddy Purple / Sedating
- Meth / no initial effect

For each state, it produces the top 500 rows sorted by price, profit, profit
per item, and price per item. Run all of them from the repository root:

```bash
python main.py
```

The generated files are saved in `output/`, for example
`output/og_kush_top_500_profit.csv`. Generation can take time because each
`run()` explores the reachable state space on the GPU. Keep the terminal open
until it prints the BFS completion message for each run.

### Customize a run

The bottom `if __name__ == "__main__":` section of `main.py` contains the
individual `run(...)` calls and their `to_csv(...)` output paths. Modify or add
calls there to change a base price/cost, starting effect, output limit, or sort
order. `run()` accepts these arguments:

```python
run(
    base_price=35.0,
    base_cost=30.0,
    initial_state=effects_to_state([Effect.CALMING]),
    limit=500,
    sort_by="profit",  # price, profit, total_cost, price_per_item, profit_per_item
    ascending=False,
)
```

For a seed with no effects, use `initial_state=0`. Save the returned DataFrame
to a CSV in `output/`, such as:

```python
top_states = run(sort_by="profit")
top_states.to_csv("output/my_custom_report.csv")
```

## View output data in Streamlit

After generating at least one CSV, launch the interactive viewer from the
repository root:

```bash
streamlit run see_data.py
```

Streamlit will display a local URL (normally `http://localhost:8501`). Open it
in a browser to select a CSV, inspect the table, download it, open it in a new
tab, or compare two files side by side. Press `Ctrl+C` in the terminal to stop
the Streamlit server.

## Project files

- `main.py` — CUDA BFS, scoring, and report generation.
- `see_data.py` — Streamlit viewer for `output/*.csv` files.
- `data/items.json` — item/effect input data.
- `output/` — generated report CSV files.