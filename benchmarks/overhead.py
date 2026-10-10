"""Where graph mode loses to jax.jit: per-op dispatch and missing fusion.

    uv run python benchmarks/overhead.py

The function is a chain of cheap elementwise ops, `x = x * 1.0001 + 0.0001`
repeated, on a small array (dispatch cost dominates) and a large one (memory
traffic dominates). Under jax.jit XLA fuses the whole chain into one loop that
reads x once and writes once. Graph mode runs every op as its own kernel, so
each op pays a dispatch and a full read + write of the array.
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
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
from bench import environment  # noqa: E402

import voltax as vx  # noqa: E402


def chain(n_pairs):
    def f(x):
        for _ in range(n_pairs):
            x = x * 1.0001 + 0.0001
        return x

    return f


def time_calls(fn, x, calls, repeats):
    jax.block_until_ready(fn(x))
    rounds = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(calls):
            y = fn(x)
        jax.block_until_ready(y)
        rounds.append((time.perf_counter() - t0) / calls)
    return statistics.median(rounds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    env = environment()  # records the load average before any work
    rows = []
    for size in (1_000, 1_000_000):
        for n_ops in (8, 64, 512):
            f = chain(n_ops // 2)
            x = jax.random.normal(jax.random.PRNGKey(0), (size,), jnp.float32)
            calls = max(3, min(200, int(2e6 / (size * n_ops / 64 + 2000 * n_ops))))
            modes = {
                "eager": f,
                "jit": jax.jit(f),
                "graph": vx.function(f),
                "graph_xla": vx.function(f, jit_compile=True),
            }
            row = {"size": size, "ops": n_ops}
            for name, fn in modes.items():
                row[name + "_us"] = round(time_calls(fn, x, calls, args.repeats) * 1e6, 2)
            row["graph_us_per_op"] = round(row["graph_us"] / n_ops, 3)
            row["eager_us_per_op"] = round(row["eager_us"] / n_ops, 3)
            rows.append(row)
            print(row, flush=True)

    stem = args.out or os.path.join(os.path.dirname(__file__), "results", f"{dt.date.today().isoformat()}-{jax.default_backend()}-overhead")
    lines = [
        f"# Per-op overhead: {env['cpu']} ({env['backend']})",
        "",
        f"{env['date']} · commit {env['git_commit']} · JAX {env['jax']} · load average at start {env['load_average_at_start']}",
        "",
        "`x = x * 1.0001 + 0.0001` chained; time per call in microseconds (median of rounds).",
        "",
        "| array size | ops | eager | jit | graph | graph_xla | graph µs/op | eager µs/op |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['size']:,} | {r['ops']} | {r['eager_us']:,.1f} | {r['jit_us']:,.1f} | {r['graph_us']:,.1f} | "
            f"{r['graph_xla_us']:,.1f} | {r['graph_us_per_op']:.2f} | {r['eager_us_per_op']:.2f} |"
        )
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".json", "w") as fh:
        json.dump({"environment": env, "rows": rows, "load_average_at_end": list(os.getloadavg())}, fh, indent=2)
    with open(stem + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
