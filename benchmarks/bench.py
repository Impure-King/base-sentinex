"""Training-step throughput: raw JAX + Optax vs Sentinex in each run mode.

    uv run python benchmarks/bench.py                 # full run, writes benchmarks/results/
    uv run python benchmarks/bench.py --quick         # smoke test
    uv run python benchmarks/bench.py --workloads mlp cnn

Modes
  raw_jax           hand-written jnp/lax model + optax.adam, whole step in jax.jit
  sentinex_jit        Sentinex model/loss/optimizer, whole step in jax.jit
  sentinex_graph_xla  sx.function(step, jit_compile=True): Sentinex graph -> one XLA program
  sentinex_graph      sx.function(step): Sentinex graph, run op by op (one XLA kernel per node)
  sentinex_eager      no compilation of the step; every op dispatched from Python

Method
  * Same initial weights and the same sequence of synthetic batches in every mode.
  * First call is timed separately (it includes tracing and compilation).
  * Then `warmup` steps, then `repeats` timed rounds per mode. Every mode runs the
    same number of steps per round, chosen so a raw_jax round takes about
    `round_seconds`. Rounds are interleaved across modes (rotating order), so
    background load and thermal throttling on a fanless laptop hit every mode
    about equally. Many short rounds keep that interleaving fine-grained.
  * A round is timed from the first dispatch to block_until_ready on the last
    result, i.e. pipelined training-loop throughput.
  * Reported: median and min step time over rounds (and max), examples/s from
    the median, ratios to raw_jax, and the loss after the first step (identical
    weights and batch, so it must match across modes) and after the last step.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

import jax

sys.path.insert(0, os.path.dirname(__file__))
import workloads  # noqa: E402

import sentinex  # noqa: E402

MODES = ["raw_jax", "sentinex_jit", "sentinex_graph_xla", "sentinex_graph", "sentinex_eager"]
_NON_KERNEL = {"parameter", "constant", "tuple", "get-tuple-element", "bitcast"}


def count_hlo_kernels(hlo_text: str) -> int:
    """Instructions in the optimized ENTRY computation that do work (each fusion,
    dot or convolution is one kernel the CPU runs)."""
    entry = hlo_text[hlo_text.index("ENTRY") :]
    body = entry[entry.index("{") + 1 : entry.index("\n}")]
    ops = re.findall(r"=\s*\S+\s+([\w-]+)\(", body)
    return sum(op not in _NON_KERNEL for op in ops)


class Runner:
    def __init__(self, workload, mode, batches):
        self.mode, self.batches = mode, batches
        if mode == "raw_jax":
            self.state, self.step, self.loss_of = workload.raw_step()
        else:
            self.state, self.step, self.loss_of = workload.sentinex_step(mode.removeprefix("sentinex_"))
        self.steps_done = 0
        self.out = None

    def run(self, n: int) -> float:
        t0 = time.perf_counter()
        for _ in range(n):
            self.out = self.step(self.state, self.batches[self.steps_done % len(self.batches)])
            self.state = self.out[0]
            self.steps_done += 1
        jax.block_until_ready(self.out)
        return time.perf_counter() - t0

    def details(self) -> dict:
        batch = self.batches[0]
        if self.mode in ("raw_jax", "sentinex_jit"):
            text = self.step.lower(self.state, batch).compile().as_text()
            return {"xla_kernels_per_step": count_hlo_kernels(text)}
        if self.mode.startswith("sentinex_graph"):
            cf = self.step.get_concrete_function(self.state, batch)
            d = {
                "graph_ops_traced": cf.pass_stats["traced"],
                "graph_ops_optimized": len(cf.graph.ops),
                "trace_s": round(cf.trace_seconds, 4),
                "optimize_s": round(cf.optimize_seconds, 4),
                "compile_s": round(cf.compile_seconds, 4),
            }
            if cf.executable is not None:
                d["kernels_per_step"] = cf.executable.num_kernels
            else:
                d["xla_kernels_per_step"] = count_hlo_kernels(cf.xla_executable.as_text())
            return d
        return {}


def bench_workload(workload, modes, args) -> dict:
    batches = [workload.make_batch(i) for i in range(4)]
    jax.block_until_ready(batches)
    runners = {m: Runner(workload, m, batches) for m in modes}
    result = {}
    for m, r in runners.items():
        result[m] = {"first_call_s": round(r.run(1), 4), "first_step_loss": float(r.loss_of(r.out))}
    timed = [m for m in modes if m != "sentinex_eager"]
    for m in timed:
        t = runners[m].run(args.warmup) / args.warmup
        if m == "raw_jax":
            steps = args.steps or max(3, min(500, round(args.round_seconds / t)))
    rounds = {m: [] for m in modes}
    for rep in range(args.repeats):
        order = timed[rep % len(timed) :] + timed[: rep % len(timed)]
        for m in order:
            rounds[m].append(runners[m].run(steps) / steps)
    if "sentinex_eager" in modes:
        r = runners["sentinex_eager"]
        r.run(1)
        for _ in range(args.eager_repeats):
            rounds["sentinex_eager"].append(r.run(args.eager_steps) / args.eager_steps)

    raw_median, raw_min = statistics.median(rounds["raw_jax"]), min(rounds["raw_jax"])
    for m in modes:
        med = statistics.median(rounds[m])
        result[m].update(
            step_ms_median=round(med * 1e3, 4),
            step_ms_min=round(min(rounds[m]) * 1e3, 4),
            step_ms_max=round(max(rounds[m]) * 1e3, 4),
            examples_per_s=round(workload.batch_size / med, 1),
            slowdown_vs_raw_pct=round((med / raw_median - 1) * 100, 2),
            slowdown_vs_raw_min_pct=round((min(rounds[m]) / raw_min - 1) * 100, 2),
            steps_per_round=args.eager_steps if m == "sentinex_eager" else steps,
            steps_run=runners[m].steps_done,
            final_loss=float(runners[m].loss_of(runners[m].out)),
            rounds_ms=[round(t * 1e3, 4) for t in rounds[m]],
            **runners[m].details(),
        )
    return result


def environment() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            return None

    dirty = sh("git", "status", "--porcelain")
    return {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "git_commit": sh("git", "rev-parse", "--short", "HEAD"),
        "git_dirty": bool(dirty),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": getattr(jax.lib, "__version__", None) or sh(sys.executable, "-c", "import jaxlib; print(jaxlib.__version__)"),
        "sentinex": sentinex.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "platform": platform.platform(),
        "cpu": sh("sysctl", "-n", "machdep.cpu.brand_string") or platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1) or None,
        "load_average_at_start": [round(x, 2) for x in os.getloadavg()],
    }


def wait_for_idle(max_load: float, timeout_s: float):
    """Wait until the 1-minute load average is below `max_load` (other processes
    competing for cores make CPU timings meaningless)."""
    t0 = time.time()
    while os.getloadavg()[0] > max_load:
        if time.time() - t0 > timeout_s:
            print(f"warning: load still {os.getloadavg()[0]:.1f} after {timeout_s:.0f}s; running anyway", flush=True)
            return
        print(f"waiting for idle machine: 1-min load {os.getloadavg()[0]:.1f} > {max_load}", flush=True)
        time.sleep(30)


def to_markdown(report: dict) -> str:
    env, cfg = report["environment"], report["config"]
    lines = [
        f"# Sentinex benchmark: {env['cpu']} ({env['backend']})",
        "",
        f"{env['date']} · commit {env['git_commit']}{' (dirty)' if env['git_dirty'] else ''} · "
        f"JAX {env['jax']} · Python {env['python']} · {env['platform']}",
        "",
        f"{env['cpu_count']} cores, {env['memory_gb']} GB. Load average (1/5/15 min) at start "
        f"{env['load_average_at_start']}, at end {report['load_average_at_end']}.",
        "",
        f"{cfg['repeats']} interleaved timed rounds per mode after {cfg['warmup']} warmup steps; steps per "
        f"round sized so a raw_jax round takes ~{cfg['round_seconds']} s (eager: {cfg['eager_repeats']} x "
        f"{cfg['eager_steps']} steps). Adam, lr {workloads.LEARNING_RATE}, float32, synthetic data.",
        "",
    ]
    for name, w in report["workloads"].items():
        lines += [
            f"## {name}: {w['description']} ({w['params']:,} params)",
            "",
            "| mode | step ms median | min | max | examples/s | vs raw (median) | vs raw (min) | first call s | kernels/step | loss step 1 | final loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, r in w["modes"].items():
            kernels = r.get("kernels_per_step") or r.get("xla_kernels_per_step") or "–"
            vs = "–" if mode == "raw_jax" else f"{r['slowdown_vs_raw_pct']:+.1f}%"
            vs_min = "–" if mode == "raw_jax" else f"{r['slowdown_vs_raw_min_pct']:+.1f}%"
            lines.append(
                f"| {mode} | {r['step_ms_median']:.3f} | {r['step_ms_min']:.3f} | {r['step_ms_max']:.3f} | "
                f"{r['examples_per_s']:,.0f} | {vs} | {vs_min} | {r['first_call_s']:.2f} | {kernels} | "
                f"{r['first_step_loss']:.6f} | {r['final_loss']:.5f} |"
            )
        lines.append("")
    lines += [
        "kernels/step: for jit modes, XLA's fused kernels in the optimized program; for",
        "sentinex_graph, the number of one-op kernels the executor dispatches per step.",
        "loss step 1: same weights, same batch, so it must match across modes. final loss: all",
        "non-eager modes ran the same steps on the same batches; small float differences between",
        "fused and unfused kernels can grow over many Adam steps (eager ran fewer steps).",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workloads", nargs="+", default=list(workloads.WORKLOADS))
    p.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=0, help="steps per timed round (0 = size from --round-seconds)")
    p.add_argument("--round-seconds", type=float, default=0.3)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--eager-steps", type=int, default=3)
    p.add_argument("--eager-repeats", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    p.add_argument("--wait-for-idle", type=float, default=None, metavar="LOAD",
                   help="before starting, wait until the 1-minute load average is below LOAD")
    p.add_argument("--out", default=None, help="output path stem (default: benchmarks/results/<date>-<backend>)")
    args = p.parse_args()
    if "raw_jax" not in args.modes:
        args.modes = ["raw_jax", *args.modes]
    if args.quick:
        args.warmup, args.steps, args.repeats, args.eager_steps, args.eager_repeats = 2, 3, 2, 1, 1

    if args.wait_for_idle is not None:
        wait_for_idle(args.wait_for_idle, timeout_s=3 * 3600)
    report = {"environment": environment(), "config": vars(args), "workloads": {}}
    for name in args.workloads:
        w = workloads.WORKLOADS[name]()
        print(f"== {name}: {w.description}", flush=True)
        modes = bench_workload(w, args.modes, args)
        report["workloads"][name] = {
            "description": w.description,
            "batch_size": w.batch_size,
            "params": workloads.param_count(w),
            "modes": modes,
        }
        for m, r in modes.items():
            print(f"   {m:17s} {r['step_ms_median']:9.3f} ms/step  {r['examples_per_s']:>12,.0f} ex/s  "
                  f"first call {r['first_call_s']:.2f}s  loss {r['final_loss']:.5f}", flush=True)

    report["load_average_at_end"] = [round(x, 2) for x in os.getloadavg()]
    stem = args.out or os.path.join(
        os.path.dirname(__file__), "results", f"{dt.date.today().isoformat()}-{jax.default_backend()}"
    )
    if args.quick and not args.out:
        stem += "-quick"
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    with open(stem + ".json", "w") as f:
        json.dump(report, f, indent=2)
    with open(stem + ".md", "w") as f:
        f.write(to_markdown(report))
    print(f"wrote {stem}.json and {stem}.md")


if __name__ == "__main__":
    main()
