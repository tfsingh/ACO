# Accelerated Computing Online

### Summary
ACO is an online environment for GPU-accelerated computing, providing an alternative to Colab for learning Triton/Numba/CUDA and testing small kernels.

### Environment
The following packages are included in the Triton/Numba environment:
- Triton
- Numba
- PyTorch
- Numpy
- Pandas

Every request is executed on a NVIDIA T4 GPU, with an execution limit of 15s. We rate-limit users to 30 requests every 24 hours; we'll be monitoring traffic as time goes on and will hopefully be able to increase this limit accordingly.

### FAQs

**Why do you require sign-in?**

We do this solely for the purposes of rate limiting — we understand this may be frustrating, but gpus are expensive!

**When are you adding CUDA support?**

We hope to be adding CUDA support in the next few weeks — we might use our existing infra provider but are exploring options, including a bespoke execution sandbox.

Please feel free to email any further questions to tejfsingh@gmail.com.