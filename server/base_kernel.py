import modal

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
    execute_kernel.remote()