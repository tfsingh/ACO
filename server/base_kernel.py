import modal
import signal 

triton_image = modal.Image.debian_slim().pip_install("triton", "torch")

with triton_image.imports():
    import torch
    import triton
    import triton.language as tl

stub = modal.Stub("aconline")

@stub.function(gpu="T4", image=triton_image)
def execute_kernel():
    # CODE

@stub.local_entrypoint()
def main():
    def signal_handler(signum, frame):
        raise TimeoutError("Kernel execution timed out")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(15)
    try:
        execute_kernel.remote()
    except TimeoutError as e:
        print(e)
    