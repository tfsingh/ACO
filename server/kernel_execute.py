import modal

triton_image = modal.Image.debian_slim().pip_install("triton")

stub = modal.Stub("given")

@stub.function(image=triton_image)
def execute_kernel(function):
    exec(function)

function = """
import triton
import triton.language as tlp

@triton.jit
def add(x, y):
  # Define output tensor
  output = tlp.zeros((x.shape[0],), dtype=x.dtype)

  # Loop through elements in parallel
  with tlp.for_() as i:
    output[i] = x[i] + y[i]

  return output

# Usage example
a = triton.arange(10)
b = triton.ones(10)
c = add(a, b)
print(c) 
"""

@stub.local_entrypoint()
def main():
    execute_kernel.remote(function)

