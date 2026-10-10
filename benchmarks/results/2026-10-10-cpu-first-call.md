# First-call costs: Apple M5 (cpu)

2026-10-10T16:48:11 · commit d132355 · JAX 0.11.2 · load average at start [5.94, 9.1, 20.3]

| step | Voltax trace s | passes s | kernels (unique) | compile, 1 thread s | compile, 8 threads s | jax.jit trace+lower s | jax.jit compile s | cache hit µs (leaves) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mlp | 0.027 | 0.034 | 143 (94) | 0.68 | 0.17 | 0.016 | 0.06 | 8.2 (22) |
| cnn | 0.033 | 0.032 | 185 (152) | 1.54 | 0.38 | 0.030 | 0.30 | 12.2 (28) |
| transformer | 0.057 | 0.059 | 895 (255) | 2.19 | 0.59 | 0.072 | 0.30 | 42.0 (94) |
