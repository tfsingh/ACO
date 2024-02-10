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

Every request is executed serverlessly on a NVIDIA T4 GPU, with a kernel execution limit of 20s (does not include network/cold start times). We rate-limit users to 30 requests every 24 hours; we'll be monitoring traffic as time goes on and will hopefully be able to increase this limit accordingly. As of current, code/execution results are stored locally.

### FAQs

**Why do you require sign-in?**

We do this solely for the purposes of rate limiting — we understand this may be frustrating, but GPUs are expensive!

**When are you adding CUDA support?**

We hope to be adding CUDA support in the next few weeks — we're exploring options including our existing infra provider or a bespoke execution sandbox.

**Do you plan on charging?**

No. Executing code serverlessly is a major benefit in this regard, albeit at the cost of speed. We're also lucky in that Triton and CUDA are quite niche, and our use cases (primarily pedagogical) further limit the number of people that would benefit from something like this. If you're here, welcome — this was built for you!

Further questions can be sent to tejfsingh@gmail.com.
