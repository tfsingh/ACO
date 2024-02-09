import modal
from typing import Dict
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

triton_image = modal.Image.debian_slim().pip_install("triton", "torch", "numba", "numpy", "pandas")

auth_scheme = HTTPBearer()

stub = modal.Stub("aconline")

with triton_image.imports():
    import torch
    import triton
    import triton.language as tl
    import numba
    import pandas as pd
    import numpy as np

@stub.function(gpu="T4", image=triton_image, secrets=[modal.Secret.from_name("aconline-token")])
@modal.web_endpoint(method="POST")
async def execute_kernel(item: Dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    import asyncio
    import os
    try:
        if token.credentials != os.environ["AUTH_TOKEN"]:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        with open("kernel.py", "w") as file:
            file.write(item["code"])

        process = await asyncio.create_subprocess_exec(
            'python', 'kernel.py',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=20.0)
        except asyncio.TimeoutError:
            process.kill()
            stdout, stderr = await process.communicate()
            return "ERROR: kernel execution time exceeded limit (10s)"

        stdout = stdout.decode()
        stderr = stderr.decode()

        if process.returncode == 0:
            return stdout
        else:
            return stderr

    except Exception as e:
        print(f"An error occurred: {e}")

@stub.local_entrypoint()
def main():
    execute_kernel.remote()