# Voltax walkthrough

This explains how Voltax works, why it is built this way, and how its graph mode
differs from `jax.jit`. It ends with the order to rebuild it in and the questions
an interviewer is likely to ask.

Read it next to the code. Every section names the file it describes. Graph mode
is about 1,100 lines across seven files in `src/voltax/graph/`.

## Contents

1. [The one idea: one op table, several ways to run it](#1-the-one-idea-one-op-table-several-ways-to-run-it)
2. [Tracing: turning Python into a graph](#2-tracing-turning-python-into-a-graph)
3. [Signatures, caching and retracing](#3-signatures-caching-and-retracing)
4. [The graph IR](#4-the-graph-ir)
5. [Graph-mode autodiff](#5-graph-mode-autodiff)
6. [Optimization passes](#6-optimization-passes)
7. [Executing a graph op by op](#7-executing-a-graph-op-by-op)
8. [Graph mode vs jaxpr → XLA](#8-graph-mode-vs-jaxpr--xla)
9. [The library on top](#9-the-library-on-top)
10. [Benchmarks: what they show](#10-benchmarks-what-they-show)
11. [Limitations](#11-limitations)
12. [Rebuilding it yourself: what to do first](#12-rebuilding-it-yourself-what-to-do-first)
13. [Interview questions to be ready for](#13-interview-questions-to-be-ready-for)

---

## 1. The one idea: one op table, several ways to run it

Files: `primitive.py`, `ops.py`.

Every array operation a model uses is a `Primitive`: a name, an `impl` written
with `jax.numpy`/`lax`, and optional VJP rules. Calling a primitive does one check:

```python
def __call__(self, *args, **params):
    for a in args:
        if type(a) is Symbol:                     # we are tracing a Voltax graph
            return a.builder.record(self, args, params)
    return self.impl(*args, **params)             # otherwise: just run it
```

That `if` is the whole trick. Layers, losses and optimizers are written with
`voltax.ops` (and ordinary `+`, `*`, `@`, indexing), so the same model code runs
in four ways:

| mode | first call | every later call |
|---|---|---|
| eager | nothing special | Python runs the model; each op is dispatched to JAX one at a time; `jax.value_and_grad` does AD op by op |
| `jax.jit` | JAX traces Python with its tracers into a jaxpr (AD included), lowers it to StableHLO, XLA optimizes and compiles one executable | C++ looks up the signature, runs the one executable |
| `vx.function` | Voltax traces Python with `Symbol`s into a Voltax `Graph` (AD included), runs graph passes, compiles one small XLA kernel per op (cached) | Python looks up the signature, runs the kernels one after another |
| `vx.function(jit_compile=True)` | same Voltax graph, then evaluated under `jax.jit` so XLA compiles it as one program | runs the one executable |

Under `jax.jit`, `impl` receives JAX tracers, so Voltax ops become part of JAX's
jaxpr. Voltax adds nothing at run time there: after compilation, Voltax-under-jit
and hand-written JAX are the same kind of XLA program (the benchmarks check this).

Graph mode is the interesting part. It is a working model of TensorFlow's
`tf.function` (without `jit_compile=True`): trace once per signature into a graph,
optimize the graph, then run it with an executor that calls one kernel per node.

---

## 2. Tracing: turning Python into a graph

Files: `graph/tracing.py`, `graph/kernels.py`.

**Symbols.** A `Symbol` stands in for an array whose value isn't known yet. It
has a shape and dtype (its `Aval`) and points at the graph node that will compute
it. Its operators (`__add__`, `__matmul__`, `__getitem__`, ...) call Voltax ops,
which see a Symbol argument and record a node instead of computing.

**The builder.** `GraphBuilder` is the trace context. `vx.function` creates one,
pushes it on a thread-local stack, creates one input node per array argument, and
calls your Python function on Symbols. `record(prim, args, params)` does four
things:

1. Converts each argument into a node input. A Symbol becomes its node. A Python
   scalar becomes a `Literal`. A concrete array (a closed-over weight, a numpy
   mask) becomes a `const` node.
2. Freezes `params` into a hashable tuple (they are part of cache keys later).
3. Works out the output shape and dtype (abstract evaluation, below).
4. Appends a `Node` and returns a new Symbol for it.

**Abstract evaluation.** To know a node's output shape, Voltax runs
`jax.eval_shape` on the primitive's own `impl` with `ShapeDtypeStruct` inputs.
Nothing is computed, and the shape rule can never disagree with the kernel,
because it *is* the kernel. The result is cached per (op, params, literals, input
avals), so each distinct op is abstractly evaluated once. A hand-written framework
would write shape rules per op instead (TensorFlow has a C++ shape function per
op). That is more code, but it doesn't depend on JAX.

**Literals vs consts.** `x * 2.0` records `mul(%x, 2.0)` with the Python float
kept as a literal. That preserves JAX's weak typing: if `x` is float16 the result
stays float16. If `2.0` became a float32 array, the result would be promoted to
float32. `tests/test_passes.py::test_literals_keep_weak_typing` checks this.

**What happens to your Python.** It runs once, at trace time. Consequences:

* Python control flow on static values (`if mode == "relu"`, loops over layers)
  is fine: it decides which ops get recorded.
* Python control flow on traced values (`if ops.sum(x) > 0:`) can't work: the
  value doesn't exist yet. `Symbol.__bool__` raises `TracingError` and tells you
  to use `ops.where`.
* Side effects (`print`, appending to a list) happen only while tracing.
* An op with no Symbol arguments runs eagerly during tracing. `ops.exp(numpy_array)`
  is just computed and becomes a const. You get constant folding for free here.
* Passing a Symbol to `jax.numpy` fails with JAX's "not a valid JAX type"
  error; inside graph mode you must use `voltax.ops`.
* A Symbol stored in a global and used after its trace finished raises
  `TracingError` (JAX calls this an "escaped tracer").

This is the same mechanism JAX uses to build jaxprs, and TensorFlow uses to build
`FuncGraph`s. The difference is that here it fits in one readable file and
produces Voltax's own IR.

---

## 3. Signatures, caching and retracing

File: `graph/function.py`.

Each `vx.function` keeps a dictionary from *signature* to `ConcreteFunction`
(one traced, optimized, compiled graph). On every call:

1. Flatten `(args, kwargs)` into leaves with `jax.tree.flatten`.
2. Build the key: the pytree structure, plus for each leaf
   * an array: `("array", shape, dtype, weak_type)`,
   * anything else: `("static", type, value)` (must be hashable).
3. Hit: run the cached executable on the array leaves. Miss: trace, optimize,
   compile, store, then run.

**What causes a retrace:**

| change between calls | `vx.function` | `jax.jit` |
|---|---|---|
| new array values, same shapes | no retrace | no retrace |
| new shape or dtype | retrace | retrace |
| new pytree structure (e.g. extra dict key) | retrace | retrace |
| new Python scalar value (`lr=0.05` → `0.01`) | **retrace** (static, as in `tf.function`) | no retrace (traced as a weakly-typed scalar) |
| same number passed as a 0-d array | no retrace | no retrace |

Treating Python scalars as static is the `tf.function` rule, and it's a classic
source of slowdowns: pass the step number as a Python int and you trace every
step. Voltax warns after 5 traces of the same function. `examples/inspect_graph.py`
section 5 shows each row of this table happening.

**Closures are captured by value.** A weight the function closes over becomes a
const node in the graph. Rebinding the Python variable later doesn't change the
cached graph. `jax.jit` behaves the same way. TensorFlow differs here: a
`tf.Variable` is captured by reference, so the graph reads its current value.

**Nesting.** Calling a `vx.function` while another is being traced just inlines
its Python body. Calling one inside `jax.jit` or `jax.grad` (arguments are JAX
tracers) replays the cached Voltax graph through JAX with `interpret`, so graph
mode composes with JAX transforms.

**Cost of a cache hit.** Flattening and building the key happens in Python, so
a hit costs microseconds per call. `jax.jit` does the same lookup in C++.

---

## 4. The graph IR

File: `graph/ir.py`.

A `Graph` is a flat list of `Node`s in topological order, with input nodes and
output nodes. Each node defines exactly one array (SSA):

```
%4 = matmul(%2, %1)                         f32[8,4]
%5 = add(%4, %0)                            f32[8,4]
%6 = relu(%5)                               f32[8,4]
%7 = max(%6) axes=(1,) keepdims=True        f32[8,1]
...
%34 = mul(0.1, %31)                         f32[4]
%35 = sub(%0, %34)                          f32[4]
```

A node is `(id, op, inputs, params, aval)`, plus `value` for consts. Inputs are
earlier nodes or literals; params are static (axes, shapes, dtypes, strides).

Design choices worth defending:

* **Node ids are creation order, and creation order is a topological order.** A
  node can only use Symbols that already exist. So no topological sort is ever
  needed, and autodiff can walk the graph backwards by id.
* **One output per node.** It keeps the executor's register file and autodiff
  simple. Where an op naturally has several outputs, such as the backward of a
  convolution (gradient for the input and for the filter), Voltax makes one node
  per output. TensorFlow does the same: `Conv2DBackpropInput` and
  `Conv2DBackpropFilter` are separate ops.
* **Static shapes.** Every aval is fully known. That is what lets each node be
  compiled ahead of time.

**Compared with a jaxpr.** A jaxpr is also typed SSA over primitives. But a
jaxpr allows multiple outputs per equation and nests sub-jaxprs (`pjit`,
`custom_jvp_call`, `scan`, `cond`), and it is a short-lived intermediate: JAX
lowers it straight to StableHLO, and XLA's HLO is where the optimization happens.
In Voltax the graph *is* the thing that gets optimized and executed. Run
`examples/inspect_graph.py` to see the same step as a Voltax graph, a jaxpr,
and XLA's optimized HLO.

---

## 5. Graph-mode autodiff

Files: `autodiff.py`, `graph/autodiff.py`, VJP rules in `ops.py`.

`vx.value_and_grad(fn)` checks whether a graph trace is active. If not, it is
`jax.value_and_grad`. If so, it differentiates symbolically on the graph, the way
`tf.gradients` does in TensorFlow graph mode:

1. **Fresh variables.** Each leaf being differentiated is wrapped in an
   `identity` node, and `fn` is called on those. This makes each leaf an
   independent variable even if two leaves are the same Symbol (tied weights),
   which matches JAX. The identities are removed later by a pass.
2. **Forward sweep.** Walk nodes from the first wrt node to the loss and mark
   which ones depend on a wrt node. Only those can carry gradient. Non-differentiable
   ops (comparisons, argmax, RNG, `stop_gradient`) stop the marking.
3. **Backward sweep.** Seed the loss with a const `1.0`. Walk nodes in reverse id
   order. For each node with a cotangent `g`, call the VJP rule of each marked
   float input and add the result into that input's cotangent (`add` nodes for
   fan-out). Reverse id order guarantees a node's cotangent is complete before
   it is used.
4. Return the cotangents of the wrt nodes (zeros for leaves the loss doesn't use).

The new gradient nodes are appended to the same graph, so the result is one graph
that computes the loss *and* the gradients. It gets optimized and executed like
anything else.

**VJP rules** are written with Voltax ops, one per input:

```python
mul.defvjp(
    lambda g, out, x, y: _unbroadcast(g * y, shape_of(x)),
    lambda g, out, x, y: _unbroadcast(g * x, shape_of(y)),
)
exp.defvjp(lambda g, out, x: g * out)          # reuses the forward output
relu.defvjp(lambda g, out, x: where(greater(x, 0.0), g, 0.0))
```

`_unbroadcast` sums a gradient back down to an input's shape: the adjoint of
numpy broadcasting. Every binary op needs it.

**Derived backward kernels.** For ops whose gradient is a different kind of
computation (convolution, pooling, gather/`take`, `getitem`, `concatenate`,
`einsum`), Voltax doesn't hand-write the math. It creates a backward primitive,
e.g. `conv2d_grad1`, whose kernel body is `jax.vjp` of the forward `impl`. The
graph still gets a real backward node per input, and the executor still runs it
as one kernel. This mirrors TensorFlow's separate backward kernels. Be upfront
about it: the *structure* of the gradient graph is Voltax's, but the inside of
those few backward kernels comes from JAX.

Because gradient nodes are ordinary nodes, gradients can be differentiated again.
`tests/test_autodiff.py::test_second_order` checks a second derivative against JAX.

**Compared with JAX.** JAX doesn't write VJP rules per primitive. It defines
forward-mode (JVP) rules, linearizes, and *transposes* the linear part to get
reverse mode. That gives forward mode for free and composes with `vmap`. The
Voltax/TensorFlow approach (direct VJP rules) is simpler to explain and is all
graph mode needs.

---

## 6. Optimization passes

File: `graph/passes.py`.

After tracing, `passes.optimize` runs four passes. Each returns a new graph.

| pass | what it does | why it matters here |
|---|---|---|
| `forward_identities` | removes the identity nodes autodiff inserted | 6 (MLP) to 30 (transformer) nodes per step |
| `fold_constants` | evaluates nodes whose inputs are all consts or literals, once, at build time | the start of the backward pass depends only on the seed: `neg`, `1/n` from the mean, reshapes and a broadcast become consts (5 nodes). Capped at 4096 elements so it never materializes a big broadcast |
| `cse` | merges nodes with the same (op, inputs, params) by hash-consing | 20 merges in the transformer: each LayerNorm computes `mean(x)` and `x - mean` twice (once for the output, once inside `var`), and the gradient of `centered * centered` produces `g * centered` twice |
| `dce` | keeps only nodes the outputs need | drops work whose result is never used; in these train steps CSE already removed it all |

Op counts for the benchmark train steps (forward + backward + Adam):

| step | traced | after identities | after folding | after CSE | after DCE |
|---|---:|---:|---:|---:|---:|
| MLP | 154 | 148 | 143 | 143 | 143 |
| CNN | 199 | 191 | 185 | 185 | 185 |
| transformer | 950 | 920 | 915 | 895 | 895 |

**Why CSE is safe:** every Voltax primitive is a pure function of its inputs.
Even random ops take their key as an input (stateless RNG), so two
`bernoulli(key, ...)` nodes with the same key really are the same value.
TensorFlow's graph optimizer has to exclude its stateful random ops from CSE.

**What's missing: fusion.** None of these passes merge several ops into one
loop. In XLA, fusion is the main optimization. Take the Adam update. Per parameter
tensor it is 15 elementwise ops: 3 for the first moment (`mul, mul, add`), 4 for
the second (`mul, mul, mul, add`), 2 `div`s for bias correction, 4 for
`m / (sqrt(v + eps_root) + eps)`, a `mul` by the learning rate and an `add` to
apply it. XLA fuses each chain into one loop that reads each array once and writes
once. In graph mode each op is a separate kernel that reads and writes the whole
array. The transformer has 30 parameter tensors, so about 450 of its 895 nodes are
the optimizer.

There is no algebraic simplification either: `v + eps_root` with `eps_root = 0.0`
stays an `add` node. (It is not quite a no-op: `-0.0 + 0.0` is `+0.0`. XLA's
simplifier knows rules like this; Voltax's passes don't.)

---

## 7. Executing a graph op by op

Files: `graph/kernels.py`, `graph/executor.py`.

**Kernels.** For each op node Voltax compiles a *kernel*: the primitive's `impl`
with its params and literals closed over, compiled alone by XLA
(`jax.jit(...).lower(input_avals).compile()`). Kernels live in a global cache
keyed by (primitive, literals, params, input avals). So the second `add` of two
`f32[512]` arrays anywhere, in any graph, after any retrace, reuses the same
executable. That is what a TensorFlow kernel registry gives you, except TensorFlow
ships its kernels precompiled and Voltax compiles them on first use.

Compiling hundreds of small programs one by one took 1.7 s for the MLP step and
4.3 s for the transformer. XLA compilation releases Python's GIL, so
`compiled_kernels` compiles missing kernels on a thread pool: 0.2 s and 0.5 s.

**The program.** `Executable` turns the graph into a flat list of instructions
`(kernel, input_slots, output_slot, dead_slots)` over a register file with one
slot per node:

```python
regs = template.copy()                  # consts pre-filled
regs[input_slots] = args
for kernel, ins, out, dead in program:
    regs[out] = kernel(*[regs[i] for i in ins])
    for s in dead:
        regs[s] = None                  # last use: drop the buffer
return [regs[s] for s in output_slots]
```

**Liveness.** `dead_slots` comes from a liveness pass: each intermediate is freed
right after the last instruction that reads it, so peak memory is closer to the
live set than to the sum of all intermediates. XLA goes further: buffer
assignment reuses memory between values that are never live at the same time,
and fusion means most intermediates never exist at all.

**Asynchronous dispatch.** Each kernel call returns immediately with a future-like
`jax.Array`. The Python loop keeps enqueueing while earlier kernels run. With large
ops, dispatch overhead hides behind compute. With tiny ops (small batches), the
loop itself is the bottleneck. `benchmarks/overhead.py` measures exactly this.

**Profiling.** `Executable.profile(args)` blocks after every kernel and reports
time per op type. On the CNN step it shows XLA:CPU's backward convolution and
max-pool backward taking about 60% of the step.

**Compared with TensorFlow's executor:** TensorFlow runs its graph from C++, and
independent ops can run concurrently on an inter-op thread pool. Voltax runs one
op at a time from Python. Both share the defining property: no cross-op fusion.

---

## 8. Graph mode vs jaxpr → XLA

| | Voltax graph mode | `jax.jit` |
|---|---|---|
| who traces | Voltax `Symbol`s + `GraphBuilder` | JAX tracers |
| IR | Voltax `Graph`: single-output SSA nodes over Voltax primitives | jaxpr → StableHLO → HLO |
| autodiff | symbolic reverse mode on the graph (VJP rules) | JVP + transpose while tracing |
| optimization | identity forwarding, const folding, CSE, DCE | XLA: hundreds of HLO passes incl. algebraic simplification, fusion, layout assignment |
| unit of execution | one small precompiled kernel per node | one executable for the whole step, made of fused kernels |
| intermediates | every node's output is written to memory | most live in registers/cache inside fused loops |
| memory planning | free after last use | buffer assignment and reuse across the whole program |
| per-step overhead | Python loop: one dispatch per node | one dispatch |
| first-call cost | trace + passes + compile every *new* kernel (cached across graphs) | trace + compile the whole program (cached per function) |
| what you can inspect | the graph, per-op timings | jaxpr, HLO before/after optimization |
| `jit_compile=True` | the Voltax graph is evaluated under `jax.jit` and becomes one XLA program | (this is the default) |

The trade-off in one sentence: graph mode removes Python from the hot loop and
lets you see and transform the graph, but without fusion every op pays a dispatch
and a full round trip through memory. XLA pays more at compile time to remove both.

`vx.function(..., jit_compile=True)` shows the graph IR is not tied to the
executor: the same optimized graph can be lowered to one XLA program, exactly like
`tf.function(jit_compile=True)`.

---

## 9. The library on top

Files: `nn/`, `losses.py`, `optim.py`, `train.py`, `data.py`.

**Layers are configuration; parameters are data.** A layer object holds only
hyperparameters. `init(key, input_shape)` returns its parameter pytree,
`output_shape(input_shape)` does shape inference so containers can chain layers,
and `apply(params, x, key=None, train=False)` is a pure function. `Sequential`,
`Residual` and plain functions (activations) compose into bigger layers;
`TransformerBlock` is just a `Sequential` of two `Residual`s. Because parameters
are plain pytrees (tuples and dicts of arrays), they flatten into graph inputs with
no special handling. This is the stax/Flax design, not Equinox's modules-as-pytrees.

**Optimizers follow Optax.** A `GradientTransformation` is a pair of pure
functions `init(params)` and `update(grads, state, params)`. `adam` is
`chain(scale_by_adam(...), scale_by_learning_rate(lr))`. The math follows Optax's
order of operations, and `tests/test_optim.py` checks sgd, momentum, Nesterov,
Adam, AdamW, RMSProp, global-norm clipping and schedules against Optax.

**Training is a pure step.** `TrainState(params, opt_state, step, key)` goes in,
a new one comes out. `make_train_step(loss_fn, optimizer)` builds that function;
`compile_step(step, mode)` wraps it for eager, `jit`, `graph` or `graph_xla`.
The RNG key is split inside the step and carried in the state, so a run is fully
determined by its seed (tested).

---

## 10. Benchmarks: what they show

Files: `benchmarks/workloads.py`, `benchmarks/bench.py`, `benchmarks/overhead.py`,
results in `benchmarks/results/` and `benchmarks/RESULTS.md`.

BENCHMARK_SECTION_PLACEHOLDER

---

## 11. Limitations

* **No control flow in graph mode.** There is no `cond`, `while_loop` or `scan`
  primitive, so data-dependent control flow has to be written with `ops.where`.
  Adding them means primitives whose params hold sub-graphs, and an executor that
  can run a sub-graph (this is how TF's `If`/`While` ops and JAX's `cond`/`scan` work).
* **Static shapes only.** Every new batch size is a new trace. `tf.function` can
  trace with unknown dimensions (`input_signature` with `None`); Voltax can't,
  because it compiles each kernel for exact shapes.
* **The executor is a Python loop**, sequential, with no inter-op parallelism and
  no buffer donation.
* **No mutable layer state** (BatchNorm running statistics). It would need
  `apply` to return updated state alongside its output.
* **Graph mode sits on JAX.** Kernels are XLA executables, shape inference uses
  `jax.eval_shape`, and some backward kernels use `jax.vjp`. It is a faithful model
  of the graph-runtime design, not an independent runtime.

---

## 12. Rebuilding it yourself: what to do first

Rebuild in this order. Each step is testable on its own, and later steps never
force you to redo earlier ones.

1. **Primitive + a few ops, eager only.** `Primitive` with `impl` and `__call__`
   that just runs `impl`. Write `add`, `mul`, `exp`, `sum`, `matmul`, `reshape`.
   Test against `jax.numpy`.
2. **Tracing and the IR.** `Aval`, `Node`, `Graph`, `Symbol` with operator
   overloads, `GraphBuilder.record` with `jax.eval_shape` for shapes, and the
   Symbol check in `Primitive.__call__`. Trace a two-layer MLP forward pass and
   print the graph. This is the most important step: get it right before moving on.
3. **An interpreter.** `interpret(graph, args)` that runs `impl` per node. Test
   "traced then interpreted" == eager for every op.
4. **`vx.function`.** Signature key, cache, retracing. Write the retrace tests
   (shape, dtype, structure, Python scalars) first.
5. **Symbolic autodiff.** `backward` with forward and backward sweeps; VJP rules
   for add/sub/mul/div with `_unbroadcast`, then sum, reshape, matmul, the unary
   ops. Test every rule against `jax.grad` (copy `tests/test_ops.py`'s structure).
   Add derived backward kernels last.
6. **The executor.** Per-node compiled kernels, the kernel cache, the register
   file, then liveness. Compare speed with the interpreter.
7. **Passes.** DCE first (easiest), then CSE, identity forwarding, constant folding.
   Print op counts before and after.
8. **Library.** Layers (Dense → Sequential → Conv/pool → attention), losses,
   Optax-style optimizers tested against Optax.
9. **Training and benchmarks.** `TrainState`, `make_train_step`, run modes, a raw
   JAX baseline with a parity test, then the timing harness.
10. **Extras.** `jit_compile=True` lowering, parallel kernel compilation,
    per-op profiling, the retrace warning.

Things that will bite you (they bit this version):

* `jax.Array.__mul__` treats any object with `__jax_array__` as an array, so
  defining it on Symbol breaks `jax_array * symbol`. Leave it out; JAX then returns
  `NotImplemented` and Python calls `Symbol.__rmul__`.
* Python literals must stay Python scalars, or float16 code silently becomes float32.
* `1`, `1.0` and `True` are equal and hash the same in Python, so cache keys must
  include the type of literals and static arguments.
* Ops on plain numpy arrays run eagerly at trace time, so a "constant folding"
  test written with numpy inputs tests nothing. Fold-able nodes mostly come from
  autodiff.
* Adam amplifies tiny float differences (it divides by `sqrt(v)` when `v` is
  tiny), so compare modes with SGD when you want parameters to match closely.
* Benchmarks on a laptop are noisy. Interleave modes, take medians, record the
  load average, and don't benchmark while other heavy jobs run.

---

## 13. Interview questions to be ready for

**Why is graph mode slower than `jax.jit`?** No fusion: every op reads and writes
whole arrays, and every op pays a dispatch. XLA fuses elementwise chains and plans
memory for the whole program. See the numbers in section 10 and
`benchmarks/overhead.py`.

**Then why do graph runtimes exist?** Tracing once removes Python from the step
(faster than eager), the graph can be inspected, optimized, serialized and run
without Python, and compile time is small and incremental (kernels are shared).

**How do you get shapes without running anything?** Abstract evaluation: propagate
avals through each op's shape rule. Voltax uses `jax.eval_shape` on the kernel.

**What triggers retracing?** A new signature: shapes, dtypes, pytree structure, or
values of static (non-array) arguments. Know the Python-scalar difference between
`tf.function` and `jax.jit`.

**Why must ops be pure?** Caching a trace, CSE and constant folding all assume the
same inputs give the same output. That's why RNG takes an explicit key.

**How does reverse-mode AD on a graph work?** Topological order; seed the output;
walk backwards applying VJP rules; sum cotangents at fan-out. Cost is a small
constant times the forward cost; memory holds forward values the rules need.

**What does XLA do that your executor doesn't?** Fusion, algebraic simplification,
layout assignment, buffer assignment/reuse, and scheduling over the whole program.

**How would you make the executor faster?** Run the loop in C++; run independent
ops in parallel; donate input buffers; and the big one, fuse elementwise chains
into single kernels. Once you do that you are writing a compiler, which is what XLA is.

**How would you add `cond` or `while`?** Primitives whose params hold sub-graphs;
trace each branch/body into its own graph; the executor runs the chosen sub-graph;
autodiff needs rules for them (for `while`, this is why JAX only reverse-differentiates
`scan`, which has a known trip count).

**Why one output per node?** Simpler register allocation and autodiff; multi-output
ops become several nodes (like TF's separate conv backprop ops).
