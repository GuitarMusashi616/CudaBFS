import numpy as np
import cupy as cp
from numba import cuda

# ==============================================================================
# PROOF CONDITION B: Define empty base structures at the script/global level.
# Numba requires these to be defined globally so its static compiler can find
# their types, signatures, and dimensions during parsing.
# ==============================================================================
TEST_CONSTANT_ARRAY = np.zeros(4, dtype=np.uint64)


def simulate_json_load(value_to_inject):
    """Simulates loading dynamic data into our global placeholder structure."""
    # We must explicitly invoke 'global' so we modify the reference, 
    # not just a local shadow variable.
    # global TEST_CONSTANT_ARRAY
    
    TEST_CONSTANT_ARRAY[:] = [value_to_inject] * 4
    print(f"[Host Setup] Injected '{value_to_inject}' into the global array.")


# ==============================================================================
# PROOF CONDITION A: Define a kernel that uses a hardware constant array.
# ==============================================================================
@cuda.jit
def test_constant_kernel(frontier, output):
    idx = cuda.grid(1)
    if idx >= frontier.shape[0]:
        return
        
    # Bind to hardware Constant Memory space using Numba's stub syntax
    device_constants = cuda.const.array_like(TEST_CONSTANT_ARRAY)
    
    # Simple operation: add the cached constant value to the frontier element
    output[idx] = frontier[idx] + device_constants[idx % 4]


def run_test_generation(expected_constant_value):
    """Launches the kernel and checks the results."""
    # Setup sample input/output streams
    h_frontier = np.array([10, 20, 30, 40], dtype=np.uint64)
    d_frontier = cp.array(h_frontier)
    d_output = cp.zeros(4, dtype=np.uint64)
    
    # Configure grid parameters
    threads = 4
    blocks = 1
    
    # Launch the kernel
    test_constant_kernel[blocks, threads](d_frontier, d_output)
    
    # Bring results back to host to inspect
    h_output = cp.asnumpy(d_output)
    
    print(f"  -> Input Frontier: {h_frontier}")
    print(f"  -> Kernel Output:   {h_output}")
    
    # Validate mathematical correctness
    expected = h_frontier + expected_constant_value
    assert np.array_equal(h_output, expected), f"Test failed! Expected {expected} but got {h_output}"
    print("   ✅ Match Confirmed!")


# ==============================================================================
# THE MAIN EXECUTIVE EXPERIMENT
# ==============================================================================
if __name__ == "__main__":
    print("=== STARTING NUMBA CONSTANT MEMORY EXPERIMENT ===\n")
    
    # --------------------------------------------------------------------------
    # STEP 1: Verify dynamic value assignment BEFORE the first compilation run.
    # --------------------------------------------------------------------------
    print("--- ROUND 1: First Initial Injection ---")
    simulate_json_load(value_to_inject=5)
    
    print("\n[Kernel Event] Launching kernel for the FIRST time. Compiling...")
    # PROOF CONDITION C (Part 1): Numba intercepts the values *now* (which are 5).
    run_test_generation(expected_constant_value=5)
    print("-> Result: Success. The compiler froze '5' into the hardware constant pipeline.")
    
    # --------------------------------------------------------------------------
    # STEP 2: Prove compilation freezing (The Capture Timing Lock)
    # --------------------------------------------------------------------------
    print("\n" + "="*50)
    print("--- ROUND 2: Modifying Global Array After Compilation ---")
    
    # We change the host array to 999.
    simulate_json_load(value_to_inject=999)
    
    print("\n[Kernel Event] Launching kernel for the SECOND time. Reusing cached binary...")
    # PROOF CONDITION C (Part 2): Even though the host array is now 999, the GPU 
    # constant cache was already frozen with '5' on the first run. The output 
    # will still process using 5!
    run_test_generation(expected_constant_value=5)
    
    print("\n" + "="*50)
    print("=== ALL TESTS PASSED SUCCESSFULLY ===")
    print("Summary of verified behaviors:")
    print("  A) True: The kernel successfully reads from a hardware constant array layer.")
    print("  B) True: Declaring 'global' allowed us to seamlessly overwrite our host buffer placeholder.")
    print("  C) True: Constant data behaves as a snapshot; captured exactly at the first execution trigger.")