import modal

triton_image = modal.Image.debian_slim().pip_install("triton", "torch", "numba", "numpy", "pandas")

stub = modal.Stub("aconline")

with triton_image.imports():
    import torch
    import triton
    import triton.language as tl
    import numba
    import pandas as pd
    import numpy as np

@stub.function(gpu="T4", image=triton_image)
def execute_kernel():
    import signal
    def signal_handler(signum, frame):
        raise TimeoutError("ERROR: kernel execution time exceeded limit (10s)")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(10)
    try:
        # CODE
    except TimeoutError as e:
        print(e)

@stub.local_entrypoint()
def main():
    execute_kernel.remote()