"""First-call costs: tracing, passes, kernel compilation; and the cost of a cache hit.

    uv run python benchmarks/first_call.py

For each workload's train step this measures, from a cold kernel cache:
  * Voltax trace time and pass time,
  * compiling every kernel one at a time vs on a thread pool,
  * jax.jit's trace + compile time for the same step, for comparison,
  * the Python cost of a graph-mode cache hit (flatten args + build the key + lookup).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time

import jax

sys.path.insert(0, os.path.dirname(__file__))
import workloads  # noqa: E402
from bench import environment  # noqa: E402

from voltax.graph import kernels  # noqa: E402


def measure(w) -> dict:
    batch = w.make_batch(0)
    state, step, _ = w.voltax_step("graph")
    result = {}

    traced = step.trace(state, batch)  # trace + passes, no compile
    result["trace_s"] = round(traced.trace_seconds, 4)
    result["passes_s"] = round(traced.optimize_seconds, 4)
    result["kernels"] = len(traced.graph.ops)

    for workers in (1, 8):
        kernels.clear_cache()
        t0 = time.perf_counter()
        kernels.compiled_kernels(traced.graph.ops, max_workers=workers)
        result[f"compile_s_{workers}_thread" + ("s" if workers > 1 else "")] = round(time.perf_counter() - t0, 3)
    result["unique_kernels"] = kernels.KernelStats.compiles

    jstate, jstep, _ = w.voltax_step("jit")
    t0 = time.perf_counter()
    lowered = jstep.lower(jstate, batch)
    t1 = time.perf_counter()
    lowered.compile()
    t2 = time.perf_counter()
    result["jax_jit_trace_lower_s"] = round(t1 - t0, 4)
    result["jax_jit_compile_s"] = round(t2 - t1, 4)

    # A cache hit: everything vx.function does before running the executable.
    step(state, batch)
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(200):
            step.get_concrete_function(state, batch)
        samples.append((time.perf_counter() - t0) / 200)
    result["cache_hit_overhead_us"] = round(statistics.median(samples) * 1e6, 2)
    result["arg_leaves"] = len(jax.tree.leaves((state, batch)))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workloads", nargs="+", default=["mlp", "cnn", "transformer"])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rows = {}
    for name in args.workloads:
        rows[name] = measure(workloads.WORKLOADS[name]())
        print(name, rows[name], flush=True)

    env = environment()
    stem = args.out or os.path.join(os.path.dirname(__file__), "results", f"{dt.date.today().isoformat()}-{jax.default_backend()}-first-call")
    lines = [
        f"# First-call costs: {env['cpu']} ({env['backend']})",
        "",
        f"{env['date']} · commit {env['git_commit']} · JAX {env['jax']} · load average at start {env['load_average_at_start']}",
        "",
        "| step | Voltax trace s | passes s | kernels (unique) | compile, 1 thread s | compile, 8 threads s | jax.jit trace+lower s | jax.jit compile s | cache hit µs (leaves) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in rows.items():
        lines.append(
            f"| {name} | {r['trace_s']:.3f} | {r['passes_s']:.3f} | {r['kernels']} ({r['unique_kernels']}) | "
            f"{r['compile_s_1_thread']:.2f} | {r['compile_s_8_threads']:.2f} | {r['jax_jit_trace_lower_s']:.3f} | "
            f"{r['jax_jit_compile_s']:.2f} | {r['cache_hit_overhead_us']:.1f} ({r['arg_leaves']}) |"
        )
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".json", "w") as fh:
        json.dump({"environment": env, "rows": rows, "load_average_at_end": list(os.getloadavg())}, fh, indent=2)
    with open(stem + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
