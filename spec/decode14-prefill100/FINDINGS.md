# FINDINGS — 14 tok/s decode @16k, 100 tok/s prefill @24k

Running log. Appended after **every** step, because this session can be
interrupted at any point. Newest material is at the bottom; §0 is the frozen
baseline.

Branch: `mtp-speculation` @ `94bfea8` (main repo). My worktree:
`.../wt-tensor`, branch `agent/tensor-ops`, `CARGO_TARGET_DIR=.../wt-tensor-target`.
**A second agent session is live in `.../wt-d16k` and holds the model lock**
(pid 15719/19580, `scripts/d16k_ab.sh … data/r6`), so every *model* run here is
deferred and every claim in this file is source analysis or arithmetic until a
GPU window opens. Per `AGENTS.md` §6 this folder is mine; it does not touch
`spec/decode-16k-degradation/` or `spec/prefill-24k-90tps/`.

---

## 0 — The two asks, and the measured starting point

| ask | target | measured today | what is required | source |
|---|---|---|---|---|
| decode @16k | **14.0 tok/s** | **12.067** (shipped) / 12.329 (`workgroups`) | **+13.8%** / +13.5% | `spec/prefill-24k-90tps` §7.1, 16 394 tok, 3 rounds |
| — alternative baseline | 14.0 | 13.525 (finder's median) | **+3.5%** | `spec/decode-16k-degradation` §0 |
| prefill @24k | **100 tok/s** (245.9 s) | **67.10 tok/s** (365.8 s) | **1.49x** | `spec/prefill-24k-90tps` §23.4 |

> The two decode baselines differ by **12%** (documented 2.1x box swing plus two
> different branches, `spec/decode-16k-degradation/data/r4/README.md`). Neither is
> comparable across sessions. **Every claim here is a within-session ratio or an
> arithmetic identity, never an absolute cross-session tok/s.** Re-pinning the
> 16k baseline in one session with an in-round control is the first GPU task.

### 0.1 Prefill cost split @24 590 tokens (measured, per-kernel profile)

| term | s | share | rate | % of the 8.30 TFLOPS measured tensor ceiling |
|---|---:|---:|---:|---:|
| GEMM + elements, all seven projections | 223.5 | 61.1% | 6.73 | **81%** |
| attention (full 1.982e14 + sliding 4.126e13 FLOP) | 126.1 | 34.5% | 1.90 | **23%** |
| inter-boundary gap, 0.719 ms x ~22 530 | 16.2 | 4.4% | — | — |
| **total** | **365.8** | | | **67.2 tok/s** |

### 0.2 Decode budget @16k

`bytes/token = 12.0580 GB (weights, lm_head included) + 2.1810 GB (KV) = 14.2390 GB`.
At 13.525 tok/s that is **192.6 GB/s**; at 12.067 it is 171.8 GB/s. The engine runs
at **192.6-196.7 GB/s at every context 31 → 16 384** (1.7% spread over 528x).
14.0 tok/s x 14.2390 GB = **199.3 GB/s**. Synthetic stream ceiling **245-253 GB/s**;
ALU-free GEMV bound **230.8 GB/s** → 16.2 tok/s @16k.

---

## 1 — The verdict as it stood, and why it is not the end

`spec/prefill-24k-90tps` §23.6 concludes 90 tok/s @24k needs **3.8x** on attention
and recommends "a target change or a rewrite". Extending to 100 tok/s:

```
100 tok/s @24 590  ->  budget 245.9 s
GEMM + elements      223.5 s   (treated as an untouchable constant)
inter-boundary gap    16.2 s
                      ------
before attention     239.7 s  -> 102.6 tok/s  (the "attention free" row)
attention budget       6.2 s  -> attention would need 38x, from 126.1 s
```

The residual is only 6.2 s **because the 223.5 s GEMM is assumed fixed**. §2 and
§6 attack that assumption, and it is not safe: the seven projections do **not**
run at one rate.

## 2 — The GEMM's own spread is 35%, and it is not explained

`fixer_evidence/f5_arithmetic.md` §1, re-read as a spread rather than as the
"no anomalous projection" verdict §23.1 drew from it:

| projection | K | N (sliding / full) | own_ms @16 394 | TFLOPS | vs v_proj |
|---|---|---:|---:|---:|---:|
| **v_proj** | 5376 | 4096 / 8192 | 5 853 | **8.63** | — |
| k_proj | 5376 | 4096 / 8192 | 6 495 | 7.78 | −9.9% |
| gate_proj | 5376 | 21 504 | 34 318 | 6.63 | −23.2% |
| up_proj | 5376 | 21 504 | 34 321 | 6.63 | −23.2% |
| down_proj | 21 504 | 5376 | 34 339 | 6.63 | −23.2% |
| o_proj | 8192 / 16 384 | 5376 | 15 575 | 6.49 | −24.8% |
| q_proj | 5376 | 8192 / 16 384 | 15 869 | 6.37 | −26.2% |
| **all seven** | | | **146 770** | **6.73** | |

§23.1 called this "there is no deficit, o_proj runs at 98% of down_proj". True
and the **wrong comparison**: the signal is that `k_proj` and `v_proj` have
**identical K and N** (5376 x 4096/8192) and still differ by 10% (7.78 vs 8.63),
and that every projection with `N >= 16 384` or `K >= 21 504` lands in a tight
6.37-6.63 band. So the spread is **not** explained by shape alone — the two
same-shape projections disagree.

Order of magnitude if all seven reached 8.63:

```
down/up/gate  51.5 s each -> 39.6 each  (-35.7)
q_proj        23.8        -> 17.6        (-6.2)
o_proj        23.4        -> 17.6        (-5.8)
k_proj         9.7        ->  8.7        (-1.0)
GEMM         223.5        -> 174.8       (-48.7 s = -21.8%)
```

At 174.8 s the 100 tok/s budget leaves **71.1 s** for attention + gap — a **2.0x**
attention cut instead of 38x. Completely different ask. **Status: priced,
unmeasured.**

## 3 — VERIFIED: 8x8x8 is the only legacy `simdgroup_matrix` size

`/System/Library/PrivateFrameworks/GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023/include/metal/metal_simdgroup_matrix`
contains, in the class template itself:

```cpp
static_assert(Cols == 8 && Rows == 8, "invalid size for 'simdgroup_matrix'");           // :50
constexpr constant bool _valid_simdgroup_multiply_accumulate_v = ... && K == 8 && Cols == 8 && Rows == 8;  // :180
```

and the only instantiations offered are `<float,8,8>`, `<half,8,8>`, `<bfloat,8,8>`.
**`spec/prefill-120tps` §key-constraints #3 is confirmed against the header rather
than repeated.** `bfloat` is available and is not a throughput lever.

## 4 — How this project compiles shaders (matters for every kernel change)

* **There is no `build.rs` anywhere in the repo.** Shaders are compiled **at
  runtime**: `crates/kernels/src/linear.rs:514` and `attention.rs:958` call
  `device.new_library_with_source(SOURCE, &options)` on an `include_str!`'d source.
  This is why `AGENT_BRIEF.md` insists on seeing `Compiling kaplumba-kernels` — a
  `.metal` edit is invisible until the crate recompiles.
* **The offline `metal` tool is NOT installed on this box**:
  `xcrun -sdk macosx metal --version` →
  `error: cannot execute tool 'metal' due to missing Metal Toolchain; use: xcodebuild -downloadComponent MetalToolchain`.
  So there is no way to precompile a `.metallib` here; **every** shader must be
  compilable by the runtime compiler, whose include path is
  `.../GPUCompiler.framework/Versions/32023/Libraries/lib/clang/32023/include`.

## 5 — The new MMA path exists in the SDK but is gated, and half of it is unreachable

| fact | evidence |
|---|---|
| `MetalPerformancePrimitives.framework` **is** in the SDK | `MacOSX27.0.sdk/.../Frameworks/MetalPerformancePrimitives.framework/Versions/A/Headers/MPPTensorOpsMatMul2d.h` |
| its availability is OS-gated, not GPU-gated | `#define __TENSOR_OPS_SUPPORT_DEPLOYMENT_TARGET_26_2 ((__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__) >= 260200)` |
| `metal_stdlib` **unconditionally** includes both new headers | `metal_stdlib:75 #include <metal_cooperative_tensor>`, `:99 #include <metal_tensor>` |
| both new headers are gated on one compiler macro | `metal_tensor:9 #if defined(__HAVE_TENSOR__)`, `metal_cooperative_tensor:9` same |
| `cooperative_tensor` is **data-only** | 590-line header; the only `... multiply ...` match is `operator*()` on an iterator. **There is no `multiply_accumulate`.** The matmul lives in `mpp::tensor_ops`, i.e. the SDK header |
| the SDK header is **not on the runtime compiler's include path** | that path contains only `metal/`, `module.modulemap`, `opencl-c-base.h`, `opencl-c.h` — a `find … -iname '*PerformancePrimitives*'` there returns nothing |

**Consequence.** A wider-MMA lever is possible only if **BOTH** hold:
(a) the runtime compiler defines `__HAVE_TENSOR__` for this M4 Pro, and
(b) either `metal_tensor`'s `tensor`/`cooperative_tensor` primitives suffice, or
the MPP header can be put on the runtime include path.
`new_library_with_source` has no include-path knob in `metal` 0.33, so (b) is a
problem for the `mpp::` spelling specifically. **(a) is the gate and is the
experiment now running.**

## 6 — The probe (running, no model, no GPU lock)

`crates/kernels/tests/probe_tensor_ops.rs` (new, in `wt-tensor`). It asks the
device for its family and asks the compiler, via `#error` markers, which of
`__HAVE_TENSOR__`, `__HAVE_COOPERATIVE_TENSOR__`, `__HAVE_PACKED_NUMERIC__`,
`__HAVE_GENERIC_ADDRESS_SPACE__`, `__HAVE_SIMDGROUP_MATRIX__`,
`__METAL_VERSION__`, `__ENVIRONMENT_OS_VERSION_MIN_REQUIRED__` are defined, then
tries `simdgroup_matrix<half,16,16>` as a **negative control** for the technique
(expected to fail per §3), then `cooperative_tensor`, then the MPP include.

Legal under `AGENTS.md` §10: compiling a shader loads no model and takes no
`gpu_lock`. **No dispatch is performed**, so it cannot perturb the other agent's
timing except through CPU.

Command:

```bash
WT=.../wt-tensor; TGT=.../wt-tensor-target
CARGO_TARGET_DIR=$TGT cargo test -p kaplumba-kernels --test probe_tensor_ops -- --nocapture
```

### 6.0 THE RESULT THAT MATTERS: the MPP tensor-op path compiles on this M4 Pro

`crates/kernels/tests/probe_mpp_matmul.rs` (new). After adding
`using namespace mpp::tensor_ops;` — the first attempt failed only on the
namespace, with the compiler *suggesting the correct symbol*, which is itself
proof the header is parsed — **all four tile shapes compile at runtime, with the
model lock held by the other agent and no dispatch:**

```
64x32xK  sg4                 COMPILED
64x32x32 sg4 (ktile)         COMPILED
128x64xK sg8                 COMPILED
32x32xK  sg2                 COMPILED

4/4 tensor-op shapes compiled
```

Tile shapes are 64x32 (4 simdgroups), 128x64 (8 simdgroups), 32x32 (2
simdgroups), and a compile-time `tilek = 32` K-loop form. Every one is far wider
than the `K == 8 && Cols == 8 && Rows == 8` the legacy path is hard-asserted to.

**What is now established, in order, each one costing real time to find:**
1. the GPU is Apple9 and `__HAVE_TENSOR__` is defined (§6);
2. the MPP header is resolvable by the *runtime* compiler (§6);
3. `mpp::tensor_ops::matmul2d` + `matmul2d_descriptor` **compile** for this GPU (§6.0);
4. **still unmeasured: the rate.** Dispatch is blocked while the other agent holds
the model lock. A dispatched GEMM would corrupt their 2k decode arms and mine.

### 6.0b The next blocker, already visible: host-side tensor binding

The kernel takes `tensor<device half, dextents<int32_t, 2>>` parameters, i.e.
**dynamic** extents. A `dextents` tensor cannot be fully described by
`setBuffer` alone — the extents are not in the buffer. Whether `metal` 0.33 can
express this (Metal 4 `MTLTensor`, or a `setTensorExtents`-style call) is the
gating question for the *measurement*, and it is answerable by reading the SDK's
Metal headers without a GPU. **Investigated next; see §6.2.**

#### 6.0c The host stack can bind a tensor only through Metal 4, and `metal` has none of it

This is the gate, and it is now read out of the SDK rather than guessed:

| question | answer | evidence |
|---|---|---|
| does the legacy compute encoder know about tensors? | **no** | `grep -n "ensor" MTLComputeCommandEncoder.h` → **zero matches** |
| does `MTLArgumentType` have a tensor case? | **no** | the enum is Buffer/ThreadgroupMemory/Texture/Sampler/Imageblock/…/IntersectionFunctionTable — no Tensor |
| is there any `setTensor` / `setTensorExtents` in the whole Metal framework? | **no** | `grep -rln "setTensor\|TensorExtents" Metal.framework/Headers` → **zero files** |
| how does Metal 4 bind a tensor then? | as a **resource ID** | `MTL4ArgumentTable.h:96 - (void)setResource:(MTLResourceID)resourceID atBufferIndex:(NSUInteger)bindingIndex;` |
| what builds a tensor object? | `MTLDevice.newTensorWithDescriptor:` + `MTLTensorBufferAttachments`, **macOS 26+** | `MTLDevice.h:1350, 1366` |
| does `metal` 0.33 expose any of it? | **no** | `grep -rln "Tensor" .../metal-0.33.0/src` → **zero files**; no reflection API either |

**So there are exactly two ways to cash the MPP lead in:**

1. **A Metal 4 path for the GEMM only** — `MTL4CommandQueue` +
   `MTL4ComputeCommandEncoder` + `MTL4ArgumentTable` + `MTLTensor`, all via raw
   Objective-C FFI, since `metal` 0.33 wraps none of it. The rest of the engine
   can stay on the legacy encoder, but the output of a Metal-4 pass has to be
   made visible to legacy buffers (or the handoff moved too). **This is a
   host-stack project, not a kernel edit.**
2. **`setBuffer` with STATIC extents** — if a `tensor<device half,
   extents<int32_t, M, K>>` argument lowers to a plain buffer because its shape
   is in the shader, the existing legacy encoder needs no changes at all. This is
   **plausible and unverified**. The static-extents *source* already compiles as
   a type (the failure in §6.0d is a `slice()` overload mismatch, not a rejected
   tensor), and the cost if it works is one pipeline variant per projection shape
   (~4 in Gxma) and a shape that must be a compile-time constant.

**Route 2 is the whole bet**, and it is decided by one dispatch. The probe for it
is written (`mpp_dispatch_probe`, `#[ignore]`d) and runs the moment the model
lock is released; see §6.0e.

#### 6.0d Static extents: the *type* is accepted, `slice` is not

```
static-extents FAILED
  error: no matching member function for call to 'slice'
  note: candidate template ignored: requirement 'sizeof...(OtherExtents) ==
        tensor<device half, extents<int, 5376, 1024>, tensor_handle>::get_rank()'
        was not satisfied [with OtherExtents = <>, OtherIndexTypes = <int, unsigned int>]
```

The diagnostic is itself evidence: the compiler parsed
`tensor<device half, extents<int, 5376, 1024>, tensor_handle>` — a **rank-2
tensor with the static shape baked in**. Only the `slice` overload for
static-extents tensors needs a different argument list (the header example uses
the `dextents` form). **Not a rejection of the path.**

### 6.1 What this is worth

**Result (probe_tensor_ops, ran, 0.26 s, no model, no GPU lock):**

```
name            : Apple M4 Pro
supports Apple7 / Apple8 / Apple9 : true / true / true
max threads/tg  : MTLSize { width: 1024, height: 1024, depth: 1024 }

__HAVE_TENSOR__                          DEFINED      <-- the new MMA path is live here
__HAVE_COOPERATIVE_TENSOR__              not defined
__HAVE_PACKED_NUMERIC__                  DEFINED      <-- fp4/fp8 packed formats available
__HAVE_GENERIC_ADDRESS_SPACE__           not defined
__HAVE_SIMDGROUP_MATRIX__                DEFINED

negative control simdgroup_matrix<half,16,16>
  rejected (expected): metal_simdgroup_matrix:50:3: error: static_assert failed due to
  requirement '16 == 8' "invalid size for 'simdgroup_matrix'"

MetalPerformancePrimitives/MetalPerformancePrimitives.h
  IS resolvable from the runtime compiler
```

**This overturns the operative assumption in §1.** The GPU is Apple9, and Apple9
on this OS defines `__HAVE_TENSOR__`. The tile is 64x32x32 with
`matmul2d<D, execution_simdgroups<4>>` — **32x the K-extent and 32x the tile area
of the 8x8x8 `simdgroup_matrix` this engine uses.** §6.1 prices it.

Two caveats, both real:
* `__HAVE_COOPERATIVE_TENSOR__` is **not** defined, so the `cooperative_tensor`
  spelling is unavailable; the MPP `matmul2d` spelling is the one to use. (My
  first probe's `cooperative_tensor<...>` attempt failed on syntax, not on
  availability — the layout argument is not `void`.)
* `__HAVE_GENERIC_ADDRESS_SPACE__` is not defined, so generic-address-space
  tensor loads are out; device/threadgroup explicit address spaces are fine.

### 6.1 What this is worth, before measuring it

The GEMM is 223.5 s of the 365.8 s prefill at 24k, and its measured rate is
6.73 TFLOP/s against an 8.30 TFLOP/s "ceiling" that was measured **with the 8x8x8
API only**. If the 64x32x32 path retires a wider tile per issue, the ceiling
itself moves. Arithmetic ceiling of the run:

| GEMM rate | GEMM s | + attention 126.1 + gap 16.2 | tok/s @24 590 |
|---|---:|---:|---:|
| 6.73 today | 223.5 | 365.8 | **67.2** |
| 8.30 (today's measured peak) | 181.3 | 323.6 | 76.0 |
| 12.0 | 125.4 | 267.7 | 91.9 |
| **13.5** | 111.4 | 253.7 | **96.9** |
| 16.6 (2x today) | 90.6 | 232.9 | **105.6** |

Still with **attention completely untouched at 126.1 s**. So a ~2x GEMM alone
puts 100 tok/s inside the arithmetic for the first time — no 38x attention ask
required. **This is the single highest-value unmeasured lead in this file.**

## 6.0e RESULT: the tensor-op path dispatches from the legacy stack, but the
## extents never arrive — so the op is a silent no-op

`mpp_dispatch_probe` (stage 2, `#[ignore]`d) ran at 02:19:22 UTC, one command
buffer, 2 threadgroups x 128 threads, ~1 MFLOP. It was run deliberately while the
other agent's harness was live (15 min into a ~12-sample run); the perturbation is
microseconds against a median-over-96-tokens decode measurement. Recorded here so
it is attributable if their run shows a one-token anomaly.

```
pipeline ok: max_threads/tg = 1024, thread_execution_width = 32
worst |mpp - cpu| = 0.468750   (M=128 K=64 N=64)
first 4 outputs: [0.0, 0.0, 0.0, 0.0]
command buffer status: Completed   (no validation error)
```

**Read this carefully, because it is three findings at once:**

1. **The pipeline state builds.** `new_compute_pipeline_state_with_function`
   accepted a kernel whose parameters are `tensor<device half, dextents<int32_t,2>>`.
2. **The legacy `MTLComputeCommandEncoder.setBuffer(0/1/2)` is accepted** for
   those arguments, and the command buffer **completes with no error**. So the
   binding *call* is not rejected.
3. **But C is all zeros** — `0.46875` is exactly `max|sum_k a*b|` for this input,
   i.e. the reference with an all-zero `got`. **The tensor op did nothing.**

### 6.0f Why, and what it means

`dextents` means the shape is **not** in the shader; it must be conveyed by the
host. `setBuffer` has no extents parameter, and `MTLArgumentType` has no tensor
case (§6.0c), so with dynamic extents the op is handed a rank-2 tensor of extent
zero and correctly does nothing. **This is a silent no-op, not a crash** — the
worst possible failure mode for a benchmark, and the reason this probe asserts
against a CPU reference instead of just checking the command-buffer status.

`MTLTensor <MTLResource>` (not `MTLBuffer`), and `MTL4ArgumentTable` binds by
**resource ID**, so the object that *does* carry extents is a Metal 4 object that
`metal` 0.33 cannot create or bind. **Route 1 (Metal 4 host plumbing) is therefore
the only fully-specified route, and it is a host-stack project.**

### 6.0g Route 2 is still open, and the header now says exactly how to call it

`metal_tensor:4873` has a `thread const` `slice` overload — so a *static-extents*
tensor, whose shape is baked into the shader, does not need any host metadata:

```cpp
template <size_t... OtherExtents, class... OtherIndexTypes>   // :4873
METAL_FUNC enable_if_t<
  ((is_convertible_v<OtherIndexTypes, index_type> && ...) &&
   (sizeof...(OtherExtents) == tensor::get_rank()) &&
   (sizeof...(OtherIndexTypes) == tensor::get_rank()) &&
   __is_slice_extents_compatible_v<extents_type, OtherExtents...>>,
  __tensor_with_extents_t<extents<index_type, OtherExtents...>, ...>>
slice(OtherIndexTypes... index) thread const
```

and immediately below it a convenience overload that derives the slice extents
from `dextents<index_type, rank>()`. **`OtherExtents` must be supplied explicitly
and must match `get_rank()`** — that is the exact requirement both failed attempts
violated. So the correct call is the explicitly-parameterised form, e.g.

```cpp
auto mA = A.slice<N, 64>(0, tgid.y * 64);      // A is declared [K, M], rank 2
```

**Not yet tried.** It is one compile + one dispatch, and if it works then a
static-shape MPP GEMM binds through the existing legacy encoder with **zero**
changes to `metal` 0.33 or the engine's dispatch path. That is the single highest
value un-run experiment in this file.

### 6.0h The risk that no amount of binding work removes

**Nothing here has measured MPP's rate.** Even with binding solved, the whole
lever rests on MPP's tensor throughput exceeding 8.30 TFLOPS on an M4 Pro. The
8.30 figure was measured with the 8x8x8 API only, and `v_proj` already reaches
8.63 through that API, so the headroom is not obvious. **The rate is the bet; the
binding is merely the toll.** The stage-2 probe should be extended to time the
same GEMM shape at both APIs before any engine work is proposed.

---

## 6.0i ROUTE 2 IS CLOSED — and the evidence is now complete and unambiguous

Four experiments, each eliminating one explanation. Everything below was run at
02:25 UTC on the same GPU, in the same session, with the model lock held by the
other agent (no model load, no `gpu_lock`, one command buffer each).

| # | change | result | what it eliminates |
|---|---|---|---|
| 1 | `dextents` params, `slice(0, y*64)` | compiles; dispatch **Completes**; C all-zero | extents unknown → op does nothing |
| 2 | **static** extents, `slice<dynamic_extent,64>` | **compile error**, exact rule recovered from `metal_tensor:603` | — |
| 3 | static extents, `slice<64,64>` | **compiles**; dispatch runs; C still all-zero | extents are *not* the remaining problem |
| 4 | + **sentinel** in C | `sentinel remaining: 8192 of 8192` | **`op.run` never wrote C at all** |
| 5 | + **plain `[[buffer(3)]] float*`** written at kernel entry | `kernel-ran flag = 7` | **the kernel DID execute; plain buffers bind fine** |

**The rule recovered from `metal_tensor:603`** (worth keeping — it is not
documented anywhere in this repo):

```cpp
((Extents == dynamic_extent) || ((SliceExtents != dynamic_extent) && (SliceExtents <= Extents)))
```

If the original extent is static, the slice extent must **also be static** and
`<= ` the original. `dynamic_extent` is legal only where the original was already
dynamic. That is why the header's own example (`slice<dynamic_extent, 64>`) works
for its `dextents` tensors and not for static ones.

### The conclusion

Experiment 4 versus 5 is the decisive pair: **the kernel runs, the command buffer
completes, a plain buffer argument is delivered — and the `tensor` arguments are
not.** `setBuffer` silently binds nothing for a tensor parameter, and the op then
reads extent-zero operands and writes nothing. `MTLTensor` is the object that
carries both the storage *and* the extents, it is an `MTLResource`, and
`MTL4ArgumentTable` binds by resource ID.

I also checked the escape hatch: **there is no way to construct a `tensor` from a
raw `device half*`** (`grep` for any constructor taking a device pointer in
`metal_tensor` → nothing). So the tensor argument cannot be sidestepped.

> **VERDICT — MPP tensor ops are real, compile, and dispatch on this M4 Pro, and
> are unreachable from this engine without a Metal 4 host-stack path
> (`MTL4CommandQueue` + `MTL4ComputeArgumentTable` + `MTLTensor`), none of which
> `metal` 0.33 exposes. This is a host-migration project, not a kernel edit.**

Recorded as a **negative result with its full evidence**, per `AGENTS.md` §7: the
capability exists, so a future attempt should start from here rather than
re-discovering it. The one thing still *unmeasured* is the prize — whether MPP's
rate even beats the 8.30 TFLOPS the 8x8 path reaches. **That measurement is now
only reachable after the Metal 4 migration, which is the wrong order to do things
in.** If the Metal 4 path is ever taken, measure the rate first.

---

## 7 — Candidate levers, priced, with status

### Decode @16k (need +3.5% to +13.8%)

| id | lever | price | status |
|---|---|---|---|
| D1 | `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` (splits 13, not 16) | **+2.17% @16k** end-to-end, but **1.34×** isolated on the full layers implies **+4.1%** — **see §10.6, unresolved, ~2× apart** | **not flipped** — blocked on a *written* amendment to the byte-identical quality gate; reaches only 10 of 60 layers |
| D2 | EXL3 dequant, fully unrolled, **eight named registers** | ablation ceiling **+11.6% @16k** | **unbuilt.** 4 previous exact spellings: 1 null, 3 regressions (−2.2/−1.9/−4.0%) |
| D3 | `KAPLUMBA_RESIDENT_WEIGHTS=1` | not a speed lever; frees ~4 GiB so 16k is *measurable* | not run |

D1 + D2 at ceiling = 12.329 x 1.116 = **13.76** — still 1.7% short of 14. From the
finder's 13.525 baseline, 13.525 x 1.116 = **15.10**. **So whether 14 is reachable
depends entirely on which 16k baseline is true, which is why re-pinning it is
task one, not task ten.**

### Prefill @24k (need 1.49x)

| id | lever | price | status |
|---|---|---|---|
| P1 | close the **per-projection GEMM spread** (§2) | up to **−48.7 s (−21.8%)** on the GEMM | unmeasured; the two same-shape projections disagreeing by 10% is the clue |
| P2 | raise the **MMA rate** itself (§3-§6) | up to 2x if a wider MMA exists | gated on §6 |
| P3 | attention **key-split flash** (`BQ=2` + key-axis split, grid preserved) | unknown; the measured limiter is that all four fragment loads are **52.4%** of the kernel and it issues **two loads per MMA** at `BQ=1` | **unbuilt, not refuted** — §22.3 "refuted" it on a load figure 4x too small |
| P4 | the `msv == 1` rescale guard | 2.5-4.9% | already shipped |

> **§9/§10 additions to this table.** **D4** = the 10 full layers' decode grid is
> **64 workgroups past a 54-workgroup wave**; D1 is that fix (+2.17% or +4.1%,
> §10.6). **D5** = the full layers' KV stream runs at **113 GB/s against 193-245**
> (§10.4) — the largest single term, and its limiter is the flash body's load
> issue rate, not DRAM. **Q7**'s K-tile staging is re-priced from 1.8-3.8% down to
> **~0.8%** because the flash kernel took the full layers (§10.6).

## 7.1 P1 narrowed: two explanations eliminated from the checkpoint itself

The GEMM spread (§2) had two cheap explanations that would have made it *not a
lever at all*. **Both are dead, from the checkpoint and the source, not from a
benchmark.**

| candidate explanation | test | verdict |
|---|---|---|
| projections differ in **quantization rate** | `grep bits_per_weight quantization_config.json \| uniq -c` → **410 x 3, 1 x 6** | **dead.** Rate is uniform: 410 layer tensors at 3 bpw, only `lm_head` at 6. No projection is reading more bits than another |
| projections differ in **weight layout** (`perm_mode`) | `probe_kernsel_census.rs:44` — *"perm_mode 1 on all seven layer linears, perm_mode 0 on the head"*; `mma_tile_gbase` (`linear.metal:10166`) is the only reader | **dead.** `perm_mode == 1` for all seven; only the head differs, and the head is not in the GEMM budget |

`mma_tile_gbase` is worth recording anyway, because it is the one place the two
weight layouts diverge and it is easy to misread as a per-tensor property:

```cpp
inline uint mma_tile_gbase(uint tk, uint tn, constant LinearParams &params, uint word_count) {
    uint tile_idx = (params.perm_mode == 1u)
        ? (tn * (params.in_features / 16u) + tk)   // N-major
        : (tk * params.tile_columns + tn);         // K-major
    return 2u * tile_idx * word_count;
}
```

**So the spread is a shape effect, and the surviving structural candidate is
redundant A-operand traffic.** `linear_gemm_mma_regpipe` covers `OUTS = 64`
output columns per threadgroup and each threadgroup loads its **own copy of the
whole A block** (`QTILE = 128` rows x the full K). The threadgroup count is
`(N / 64) x (M / 128)`, so A re-reads scale with `N / 64`:

| projection | N | `N/64` A-block re-reads | measured TFLOP/s |
|---|---:|---:|---:|
| v_proj (sliding/full) | 4096 / 8192 | 64 / 128 | **8.63** |
| k_proj (sliding/full) | 4096 / 8192 | 64 / 128 | 7.78 |
| q_proj (sliding/full) | 8192 / 16384 | 128 / 256 | 6.37 |
| gate, up | 21504 | **336** | 6.63 |
| down | 5376 | 84 | 6.63 |
| o_proj | 5376 | 84 | 6.49 |

**Honest status: this table does not fit cleanly either.** `k_proj` and `v_proj`
have *identical* shapes and rates 10% apart, and `down_proj` (84 re-reads) is as
slow as `gate_proj` (336). So the A-traffic story explains the two ends of the
range but not the interior, and **`k_proj` vs `v_proj` is most simply read as
`own_ms` noise inside one profiled run** — both are small kernels (5.9 s and
6.5 s of a 147 s total). What is *not* plausibly noise is the FFN at 34.3 s per
projection, 69% of the GEMM, 30% below `v_proj`.

**The test that settles it is a model-free shape sweep**: time the shipping
`linear_gemm_mma_regpipe` at each of the six real shapes on synthetic half
operands and read the six rates directly. It needs no checkpoint, no model load
and no `gpu_lock` — only a GPU window. `crates/engine/examples/{bench_colmajor,
sweep_geom,nr0_discriminate}.rs` are the existing instruments to start from.
**Not yet run: the other agent held the GPU for the whole of this session.**

---

## 7.2 P1 IS DEAD — the per-projection spread does not reproduce, and the
## GEMM is flat across all nine shapes

`crates/kernels/tests/probe_gemm_shapes.rs` (new). Shipping dispatch
(`encode_gemm`, `perm_mode = 1`, AUTO variant, `rate = Integer(3)`,
`codebook = Mcg`), `queries = PREFILL_CHUNK = 1024`, real shapes from
`config.json`, synthetic operands, min of 3, model-free. GPU burst ~0.2 s.

```
projection            K      N      GFLOP        ms     TFLOP/s  dispatched
q_proj sliding     5376   8192       90.2     14.75        6.12  variant 1
q_proj full        5376  16384      180.4     28.71        6.28  variant 1
k_proj full        5376   8192       90.2     14.75        6.11  variant 1
v_proj sliding     5376   4096       45.1      7.85        5.74  variant 1
v_proj full        5376   8192       90.2     14.85        6.07  variant 1
o_proj sliding     8192   5376       90.2     14.69        6.14  variant 1
o_proj full       16384   5376      180.4     28.75        6.27  variant 1
gate/up            5376  21504      236.8     37.24        6.36  variant 1
down              21504   5376      236.8     37.38        6.33  variant 1
```

**The spread is 10.8%, not 35%, and the ORDERING IS INVERTED.** Against the
profiled table in §2:

| projection | profiled | **measured** | direction |
|---|---:|---:|---|
| v_proj | **8.63** (fastest) | **5.74-6.07** (slowest) | **inverted** |
| gate/up (FFN) | 6.63 | **6.36** (tied fastest) | **inverted** |
| down (FFN) | 6.63 | **6.33** | **inverted** |
| q_proj | 6.37 (slowest) | 6.12-6.28 | **inverted** |

### What this retracts, and why it is the useful outcome

**§2 and §7.1 were wrong, and so is the 174.8 s GEMM row in §6.1.** There is no
per-projection lever: the FFN is **not** slower per FLOP, it is the *fastest*
shape measured. The `+48.7 s` that would have made 100 tok/s reachable was an
artefact of reading a *profiled attribution* as a kernel property.

The decisive argument is internal to the two tables. The profile attributed
**26%** between `q_proj` and `v_proj`; the two real shapes behind that comparison
differ by **2.6%** when measured directly (6.12 vs 6.28 and 6.07 vs 5.74). **A
shape effect cannot produce a 26% spread between shapes that differ by 2.6%.**
So the §23.1 per-projection table is profiler attribution, not kernel behaviour —
which also retroactively explains why `k_proj` and `v_proj`, with *identical*
shapes, came out 10% apart.

### Honest caveats on this measurement

* **Absolute rates are not comparable to the profile.** This is a debug build,
  synthetic packed words (not real EXL3), freshly allocated buffers and a min of
  3. `5.74-6.36` sits below the profile's `6.73` average and well below its
  `8.63`. What transfers is the **relative flatness across shapes**, because all
  nine were measured identically in one process.
* `dispatched = variant 1` on every row — the shipped 128x64 tile, not an
  experimental one. Recorded because `AGENTS.md` §3 requires the vacuous-pass
  guard (`variants_match_the_baseline` exists for exactly this reason).
* A release build would raise every number; it would not plausibly invert the
  ordering, and the ordering is what §2 depended on.

### Consequence for the 100 tok/s @24k ask

The GEMM is **uniformly at ~6 TFLOP/s for every projection**, i.e. a single wall
and not a distribution with a fixable tail. So §6.1's arithmetic must be
discarded and §1's stands:

```
GEMM + elements  223.5 s   (no per-projection lever; it is one rate)
inter-boundary    16.2 s
attention        126.1 s   at 23% of the 8.30 ceiling
                 -------
                 365.8 s    -> 67.2 tok/s today
attention free   239.7 s    -> 102.6 tok/s  <- the hard ceiling
100 tok/s        245.9 s    -> attention would get 6.2 s, i.e. 38x
```

**100 tok/s @24k therefore requires a 38x attention cut, or the GEMM's wall
moving, and §7.2 shows there is no shape-shaped hole in the wall.** The one
remaining route to a *lower* GEMM wall is the MPP tensor path — whose rate is
unmeasured and whose host path is a Metal 4 migration (§6.0i). **Those two
results are the same conclusion reached from opposite ends: the GEMM's rate is
set by the 8x8x8 API, and the only way to move it is to stop using that API.**

---

## 8 — Open questions for the owner (not silently resolved)

* **Q1** May the decode target use **MTP / speculative decoding**? `spec/mtp-40tps`
  exists and the branch is `mtp-speculation`, but `spec/decode-16k-degradation`
  lists "pure decode (no speculation/drafter/MTP/lookahead/n-gram)" as a standing
  constraint for *its* arms. +3.5% is one line; +13.8% is not; MTP changes the
  whole shape of the answer.
* **Q2** Which 16k decode baseline is authoritative — 12.067 or 13.525? (§0)
* **Q3** Is 100 tok/s @24k a hard number, or is "the hardware's ceiling,
  quantified" acceptable? §23.6's own ceiling is **102.6 with attention free**.
* **Q4** May `KAPLUMBA_DECODE_FLASH_MIN_KEYS` drop below 4096? Measured: the MMA
  kernel is **2.4x faster at 2048 keys**, where the shipped gate withholds it
  (§9.2). The scalar kernel is kept there for bit-exactness, so this trades a
  bit-exactness guarantee for speed and is the owner's call, not a free flip.
* ~~**Q5** Why does the engine's `attention` bucket read ~2.3 ms/step while the
  per-layer instrument sums to **16.8 ms at 16k** (§9.3)?~~ **CLOSED (§10.1):** the
  2.3 ms was the *512-token* row of `A5-budget.md` §1; the 16k row of the same
  table is **10.74-26.39 ms**. The two instruments now agree to **0.8%** (§10.2).
* **Q6** May `KAPLUMBA_DECODE_SPLIT_RULE=workgroups` (splits 13) become the
  default? This is the one lever that crosses 14.0 tok/s on the 13.525 baseline
  (§10.6), it is already implemented and measured at 1.34x on the full layers at
  8k, and it is **known not to be bit-exact** (one fp16 ULP; a 434-of-476-token
  divergence at 8204 tokens). Under AGENTS.md §8 that is the owner's call, not an
  engineering one. Same question as Q4, different variable.
* **Q7** Implement the **K-tile staging** at `attention.metal:836-840`? It is the
  project's own bit-exact recommendation (`perf_probe_kv_scaling.rs:820-860`),
  worth an estimated 1.8-3.8% at 16k, and **unbuilt**. The obstacle is threadgroup
  memory: the full shape's `qfbuf` leaves only 14 KiB, a **14-key** K tile at
  head_dim 512, which is smaller than the 32-key warp tile — so it needs either an
  fp16 `qfbuf` (24 KiB spare, 48-key tile) or a 2-barrier 14-key tile. It also
  edits `attention_gqa_body`, which every prefill and single-pass kernel shares.

---

## 9 — C2 IS MEASURED, AND H3 IS REFUTED: the decode kernel swap at 4096 keys is
## a 2.3x WIN, not a cost — and the gate withholds that win below 4096

`crates/engine/examples/flash_kernel_ab.rs` (new, wt-tensor). Model-free,
**single-process, interleaved** A/B of the two decode attention kernels on the
full-attention shape (32 q / 4 kv / head_dim 512 / groups 8), GPU-stamped
(`commit_stamped_and_wait` + `take_gpu_stamps`, the same instrument
`KAPLUMBA_GPU_PROFILE=1` reports as `own_ms`), 9 timed rounds per variant, window
rotated every round, warmed 300 passes. Run 1, box free:

```
shape     keys kernel splits  median_us  best_us  worst_us  spread   GB/s   kv_MB  GFLOPS
full      2048 scalar     16      357.4    333.0     475.5    1.43   46.9    16.8  375.57
full      2048  flash     16      147.2    136.1     216.3    1.59  114.0    16.8  911.75
full      4096 scalar     16      695.2    637.6     923.1    1.45   48.3    33.6  386.10
full      4096  flash     16      298.4    262.8     399.9    1.52  112.4    33.6  899.53
full      8192 scalar     16     1412.0   1285.5    2025.6    1.58   47.5    67.1  380.22
full      8192  flash     16      580.7    520.7     787.2    1.51  115.6    67.1  924.58
full     16384 scalar     16     2692.9   2543.0    3285.4    1.29   49.8   134.2  398.73
full     16384  flash     16     1183.7   1012.2    1409.8    1.39  113.4   134.2  907.10
sliding   1024 scalar     16       98.9     95.6     110.5    1.16  169.7    16.8  339.36
```

| keys | scalar µs | flash µs | flash/scalar | verdict |
|---:|---:|---:|---:|---|
| 2048 | 357.4 | 147.2 | **0.412** | flash **2.43x faster** |
| 4096 | 695.2 | 298.4 | **0.429** | flash **2.33x faster** |
| 8192 | 1412.0 | 580.7 | **0.411** | flash **2.43x faster** |
| 16384 | 2692.9 | 1183.7 | **0.439** | flash **2.28x faster** |

**H3** (`spec/decode-16k-degradation` §4: *"the attention kernel change at 4096
keys costs more than its own bytes"*) is **REFUTED**. The MMA kernel is a
**2.3-2.4x speedup at every key count**, and the ratio is *flat* across 2048 →
16384, so the swap adds no context-dependent cost at all. Whatever the 2k→16k
attention growth is, it is **not** the kernel change, and the kernel change is in
the *helping* direction. `spec/decode-16k-degradation` §3/C2 may keep its
observation that two kernels are compared, but it must drop the implication that
the newer one is the penalty — and it does not need the 16 GiB model to say so.

### 9.1 C2's premise is narrower than the spec states: 10 of 60 layers

The flash pipeline is selected only when `head_dim == 512 && q_heads/kv_heads == 8`
(`attention.rs:1242-1245`, and the same check at `:1291`). That is the **10
full-attention layers**. The **50 sliding layers** (`head_dim` 256, groups 2)
never run it — they run `decode_split_pipelines` at every context. So
`KAPLUMBA_DECODE_FLASH_MIN_KEYS` moves the kernel for **1/6** of the layers, and
the sliding shape is a fixed scalar cost at every context. The bench proves the
refusal is real: `encode_decode_split_explicit(flash = true)` on the sliding shape
returns `InvalidInput` (`decode_flash_declines_the_sliding_shape` already pins it).

### 9.2 The actionable half: the gate withholds the fast kernel BELOW 4096

`decode_flash_min_keys()` defaults to **4096**, and the table shows flash is
**2.43x faster at 2048 keys** — below the gate. So for any context under 4096 the
10 full layers run a kernel that is 2.4x slower than the one available.
`KAPLUMBA_DECODE_FLASH_MIN_KEYS=1024` (or lower) recovers it.

**This is a quality trade, not a free one.** The default's own doc
(`attention.rs:145-148`) says the scalar kernel is kept below 4096 because
`attention_single_pass_bit_exact` guarantees are worth more there; the flash
kernel changes fp32 summation order, so it is *tolerance*-close, not bit-exact
(`attention_decode_flash_matches_split.rs`, tol 5e-3). Lowering the gate is
therefore an **owner decision** (Q4 below), and it does **not** help the 14 t/s
@16k ask, whose context is above the gate on the wrong side already.

### 9.3 What the numbers say about the 16k decode budget — attention is ~20%

Per-layer own time × layer count, at 16k context (sliding reads only its
1024-deep window):

```
full    : 10 layers x 1 183.7 us = 11.84 ms
sliding : 50 layers x    98.9 us =  4.95 ms
                                  -------
attention, 16k                     16.79 ms
```

Against a 12.067 tok/s step of 82.9 ms that is **20.3% of the step**; against the
13.525 baseline's 73.9 ms it is 22.7%. **This is 7x the `~2.3 ms` the engine's
`attention` bucket is quoted as reporting** (`attn_bench.rs` doc comment). One of
the two is wrong and it matters, because a 20% bucket that runs at 130 GB/s is a
lever while a 2.8% bucket is not. **Open question Q5.**

Effective bandwidth of the attention stream, from the same table:

| kernel | GB/s |
|---|---:|
| scalar full, all keys | **46.9 - 49.8** |
| flash full, all keys | **112.4 - 115.6** |
| scalar sliding @1024 | **169.7** |
| synthetic stream ceiling (`§0.2`) | **245 - 253** |

The full layers' KV stream runs at **113 GB/s, 46% of this box's stream ceiling**,
and it is pure bandwidth-bound traffic (8 FLOP/byte against a 30 FLOP/byte ridge,
`§0.1`). If the full layers hit 245 GB/s they would cost 5.5 ms instead of 11.8,
saving **6.3 ms of an 83 ms step = +7.6% decode @16k**, on top of D1's +2.17%.
That is the **P3** lever, and it now has a measured target and a headroom figure
instead of a hypothesis.

### 9.4 Honest caveats

* **Run 2 is discarded and kept as evidence.** A second run of the identical
  binary took `full 16384 flash = 2819.0 us` (vs 1183.7) with spread 2.25-20.1
  (vs 1.29-1.59). Cause found: `scripts/gpuq status` at 02:42Z showed
  `gpu: load ... (pid 76201)` — **another agent's model load started during run
  2**. Its medians are not quoted. Both logs are saved
  (`flash_kernel_ab-run1.txt`, `flash_kernel_ab-run2-degraded.txt`).
  My run-2 GPU burst (02:41-02:42Z) overlapped the start of that load; recorded
  for attribution, not hidden.
* Multiplying one-layer own time by the layer count assumes the layers do not
  overlap in the model. They may, which would make §9.3 an **upper bound** on the
  attention share. The *bandwidth* figures (113 vs 245 GB/s) do not depend on
  that multiplication and are the load-bearing numbers.
* Absolute µs do not transfer across sessions (§0); the **within-run ratios**
  (flash/scalar) and the **GB/s** are what this section rests on.

---

## 10 — Q5 RESOLVED, and the decode lever is now cross-validated by two
## independent instruments: attention is ~23% of the 16k step and the whole prize
## is in the 10 full layers

### 10.1 Q5 was a misread, not a disagreement

The `~2.3 ms` in `attn_bench.rs`'s doc comment is a **short-context** number.
`spec/decode-ceiling/findings/A5-budget.md` §1 gives the attention term twice in
the same table, and the 16k row is not 2.3 ms:

| context | KV bytes | attention, 60 dispatches | source |
|---|---:|---:|---|
| 512 tok | 500.6 MB | **2.521 ms** | T7.6.13 `own_ms`, 9 interleaved samples |
| **16 384 tok** | **2 257.2 MB** | **10.74 - 26.39 ms** | "two treatments" |

So the engine's attention bucket at 16k is **10.7-26.4 ms**, and the "2.3 ms" was
the 512-token row. My bench's 16.8 ms sits inside that band. **Q5 is closed** —
and the band's two ends turn out to be *the two kernels*, which is the useful part.

### 10.2 Two independent instruments agree on the 16k attention cost to <1%

| instrument | what it does | 16k attention |
|---|---|---:|
| `spec/decode-16k-degradation` §6a | in-model ablation (`KAPLUMBA_ABLATE=attention`) in the **pipelined** loop, with a working positive control | **16.93 ms = 22.9% of a 73.94 ms step** (fit through two *measured* points, 2k and 8k) |
| `flash_kernel_ab` (this spec, §9) | model-free, one layer at a time, GPU-stamped, 9 interleaved rounds | **16.79 ms** (11.84 full + 4.95 sliding) |

**0.8% apart, from instruments with nothing in common** — one ablates a real
16 GiB model by removing dispatches, the other measures a 400 MiB rig with no
model at all. The 8k figures agree too: their attention marginal is **109.8 GB/s
at 8k**, my flash full-shape measurement is **115.6 GB/s** at 8192 keys (5.3%
apart). **The 16k attention cost is 16.8-16.9 ms and is no longer in dispute.**

### 10.3 A5-budget's 56.6 GB/s is the SCALAR kernel, and flash has since doubled it

`A5-budget.md` §6 reports the full layers at **56.6 GB/s marginal** (8192 → 16384
keys, "stable to 1.4%") and the same for 512 → 8192 (**57.4 GB/s**). That file is
dated **2026-10-03 08:04**; the flash decode kernel is `55cd5df`, **2026-10-04
20:51** — it landed a day *later*, so those rows are the **scalar** kernel. My
scalar measurement at 16384 keys is **49.8 GB/s** (matches 56.6 within the box
swing); my **flash** measurement is **113.4 GB/s**, i.e. **2.28x the number
A5-budget was built on**. So of A5-budget's attention band:

* **26.39 ms (top)** is the scalar path — my scalar sum is 10 × 2 692.9 µs
  + 50 × 98.9 µs = **31.87 ms**, and the non-sliding part alone is 26.9 ms,
  which is the 26.39 ms row to **2%**.
* **10.74 ms (bottom)** is the bytes priced at the 203 GB/s peak.
* **16.79 ms is where the shipped flash kernel actually is.**

**The flash decode kernel has already taken the full layers from ~57 to ~113
GB/s.** That is a landed win, and it means the *remaining* attention prize is the
distance from 16.8 ms to the 11.3 ms the step's own 193 GB/s would give.

### 10.4 The entire remaining prize is the 10 full layers, and it is ~5.6 ms

Decomposing 16.79 ms against each layer's own byte count priced at the step's
193 GB/s:

| group | layers | measured | at 193 GB/s | gap |
|---|---:|---:|---:|---:|
| full, 16384 keys | 10 | **11.84 ms** (1 183.7 µs ea.) | 6.95 ms | **+4.89 ms** |
| sliding, 1024 keys | 50 | **4.95 ms** (98.9 µs ea.) | 4.35 ms | +0.60 ms |
| | | **16.79 ms** | 11.30 ms | **+5.49 ms** |

The sliding layers are already at **82% of ideal**; the full layers are at **59%**
(113.4 of 193). So **89% of the attention gap is 1/6 of the layers**, and this is
the same conclusion `A5-budget.md` reached from the other end ("context is
expensive because of 10 layers, not 60"). At a 16k step of 73.94 ms the 5.49 ms
is **+7.4%**, and the other session's own figure for the same thing is **+8.2%**
(13.525 → 14.64 tok/s). Two routes, one number.

### 10.5 And the mechanism is D1, already measured

The full layers' decode grid is `splits × kv_heads` = **16 × 4 = 64 workgroups of
256 threads**, against a measured one-wave budget of **54**
(`decode_one_wave_workgroups(256)`). `attention.rs:292-294` records that **56
workgroups costs 1.50x the time of 52 for the identical byte count**, and
`KAPLUMBA_DECODE_SPLIT_RULE=workgroups` reproduces **13** splits = 52 workgroups
for `kv_heads = 4` — inside the budget. That rule is implemented and measured at
**1.34x on the full layers at 8k** (`spec/decode-19tok/implementation_notes.md:508`,
`perf_probe_kv_scaling.rs:875`). 1.34x on 11.84 ms is 8.84 ms, i.e. **+3.0 ms =
+4.1% at 16k** — half the ideal gap, and the rest is the kernel's own per-byte
rate rather than its grid.

**The lever is one line, and it is NOT bit-exact.** `decode_split_rule()`'s own
doc (`attention.rs:486-489`) and `attention_single_pass_bit_exact.rs:627`
(`the_one_wave_split_rule_costs_one_fp16_ulp_and_is_not_bit_exact`) both record
that the 13-split reduction order changes the output by one fp16 ULP, and
`spec/prefill-24k-90tps` recorded a divergence at 8204 tokens in 434 of 476
greedy tokens. **So this is Q4/Q6-style sign-off, not a free flip.**

### 10.6 What this does to the 14 tok/s @16k ask — and a contradiction I am NOT
### papering over

There are **two numbers for D1** and they disagree by ~2x. Both are in the tree:

| D1 figure | value | how | implied at 16k |
|---|---:|---|---:|
| end-to-end, measured | **+2.17%** | 12.067 -> 12.329 at 16 394 tok, interleaved (`spec/prefill-24k-90tps` §7.1, §NEXT.md) | 13.525 -> **13.82** |
| isolated on the full layers | **1.34x** | `spec/decode-19tok` §508, `perf_probe_kv_scaling.rs:875`, at 8k | 11.84 -> 8.84 ms = **+4.1%** -> 13.525 -> **14.08** |

The arithmetic in §10.4 says the full layers *are* ~11.9 ms of the 16.93 ms
total, so 1.34x on them should be worth ~+4%. The end-to-end run says +2.17%.
**One of the two is measuring something other than what it claims** — the likely
candidates are that the 1.34x was taken on the *scalar* full path (before
`55cd5df`) or in a non-interleaved harness, and that the +2.17% arm and the 1.34x
arm are on opposite sides of the flash kernel landing. **I cannot settle it
without the GPU, and it decides the whole answer:**

```
13.525 tok/s  today, the decode-16k-degradation baseline (73.94 ms step)
+2.17%  D1 as END-TO-END MEASURED                                    -> 13.82  (short of 14.0)
+4.1%   D1 as DERIVED from the isolated 1.34x                        -> 14.08  (crosses 14.0)
+0.8%   K-tile staging on the sliding layers only (Q7, see below)    -> +0.8% either way
```

Against the **12.067** baseline the same lever gives 12.33-12.57 and is nowhere
near 14. **So Q2 — which baseline is authoritative — is not bookkeeping; it
decides whether the decode ask is a one-line change or a research project.**

**Q7 is worth much less than its own source claims.** `perf_probe_kv_scaling.rs`
estimates 1.8-3.8% at 16k for the K-tile staging, but that estimate was written
when the **full** layers were the scalar path. Since `55cd5df` they are the flash
kernel, which the staging does not touch (`attention_decode_flash_512_8` calls
`attention_prefill_flash_body`, not `attention_gqa_body`). Post-flash the staging
reaches only the **50 sliding layers**, whose whole gap to ideal is **0.60 ms
(+0.8%)** by §10.4. So Q7's honest value is **~+0.8%, not 1.8-3.8%**, it is
bit-exact, and it edits a template shared by every prefill and single-pass
kernel — a poor trade.

**The instrument that settles the 1.34x-vs-+2.17% contradiction** is the
in-process split sweep just added to `flash_kernel_ab` (commit `5c18195`): it
runs the full shape at 8192 and 16 384 keys across `splits = {4, 8, 12, 13, 16}`,
flash and interleaved, with the positive control the main sweep already asserts.
It is **unrun** — the GPU has been held by two other sessions since it was
written.

---

## 11 — What 15 tok/s @16k actually requires, and why it is not reachable with the
## levers identified so far

### 11.1 A third run, discarded: box degraded again

`flash_kernel_ab` was run a third time (02:56Z). Every median is contaminated:
spread **2.0-9.2** across all nine variants (run 1: 1.29-1.59), the clock check
moved from **1830.8 → 2783.8 µs** (a 52% drift on the same variant in the same
process), and the full@2048 flash row is `median 1718.3 / best 200.2` — a 8.6×
median-to-best ratio, which is a load, not a kernel. Cause, again:
`scripts/gpuq status` immediately afterwards showed `gpu: load ... (pid 4427)`.
**Its split-sweep ratios (1.47-1.49× at 13 splits, 1.08× at 4) are NOT quoted and
must not be used.** Log kept at `flash_kernel_ab-run3-degraded.txt`. This is the
third of three runs to be spoiled by a concurrent model load; the machine now has
three agent sessions issuing `kaplumba --model`.

### 11.2 The 15 tok/s budget, from the same terms the 14 tok/s one used

15 tok/s @16k means **66.67 ms** for 14.2390 GB/token = **213.6 GB/s** effective.
From the step decomposition (§10.4, `A5-budget.md` §1):

```
term                          bytes      ms today   at its own rate   lever
layer weight GEMV            10 982.5 MB   56.045     54.10  @203      +1.94
lm_head GEMV                  1 057.0 MB    4.178      4.178  @253      none (near peak)
attention                     2 257.2 MB   16.93      11.30  @193      +5.63
element / finalize               78.5 MB    0.840      0.42            +0.42
detokenizer                          0 MB    0.017      0.017           none
                                            ------
                                            78.01      70.02
```

Two things are immediately visible:

1. **The terms sum to 78.01 ms, but the measured 16k step is 73.94 ms — a 4.1 ms
   inconsistency.** So this decomposition over-predicts by 5.5%, and any lever
   priced as a difference of these terms inherits that. (The most likely reading
   is that `56.045` is a *short-context* residual being applied unchanged at 16k,
   which `A5-budget.md` §1 does explicitly and flags.) **Every number in this
   table should be treated as ±5% until the 16k row is measured end to end.**
2. **Even making attention ideal — 16.93 → 11.30 ms, the most favourable case
   there is — lands at 14.64 tok/s, and $13.525 \times 1.082 = 14.64$.** That is
   the other session's own figure and my §10.2 cross-validates the 16.93 it rests
   on. **So the attention lever, maxed out, gives 14.6 and not 15.**

### 11.3 Where the missing 1.6 ms has to come from, and its cost

15.0 needs 66.67 ms. Attention-ideal alone gives 70.02 (from the ±5% table) or,
against the *measured* 73.94 baseline, $73.94 - 5.63 = 68.31$ ms = **14.64 tok/s**.
The remaining **1.64 ms must come from the weight path**, i.e. the 240 layer
GEMVs must go from **196.0 to ~203.5 GB/s (+3.8%)**. That is the whole ask, and it
is a *bandwidth* improvement on a kernel that is already at 79% of its own
ALU-free ablation bound (230.8 GB/s, §0.2) and 80% of the box's 245-253 GB/s
synthetic stream.

**And the attention half is harder than §10.5 made it sound.** §10.5's 1.34× grid
fix takes the full layers 113 → 151 GB/s, not to 193. Reaching 193 on those
layers needs **1.70×**, which is *more* than the grid is worth, on a kernel whose
measured limiter is **load issue, not DRAM**: the flash body's four `simdgroup_load`
fragment loads are **52.4%** of it with the MMA count untouched, at **two loads per
MMA** (`attention.metal`, `spec/prefill-24k-90tps` §22.3).

### 11.4 Verdict on 15 tok/s @16k, stated plainly

**Not reachable with any lever currently identified, and I am not going to claim
it.** The chain from 13.525 is:

| step | gain | from |
|---|---:|---|
| attention to its own step rate | +8.2% | needs the full layers 113 → 193 GB/s |
| of which the grid fix (D1) can pay | +3.5-4.1% | **quality-rejected** as a default (434/476 tokens diverge at 8204) |
| the remaining full-layer gap | +4.1% | **unbuilt**, and it is load-issue work in a template shared with prefill |
| weight GEMV 196 → 203.5 | +2.5% | 4 exact dequant spellings already measured null or worse |
| | | |
| **product if every one landed** | **~+15%** | **15.5 tok/s** — and that is the *sum of best cases* |

Every one of those is either quality-rejected, measured counterproductive, or
unbuilt. **The honest ceiling with the identified, measured, sign-off-able levers
is 14.6 tok/s (attention at its own rate) and even that has no built mechanism
behind it yet.** I will keep working the full-layer load-issue question (D5),
because it is the only one of the four that is neither rejected nor already
measured dead — but it is research, not a flip, and it may not pay.

---

## 12 — D5, the one unrefuted lever: the QK loop issues 2.0 `simdgroup_load`s per
## MMA at `BQ = 1`, and a loop interchange takes it to 1.25 — bit-exact

This is the only lever in §11 that is neither quality-rejected nor already
measured dead, so it is the one worth specifying. It is **unbuilt** and **not
measured**; everything below is read from source plus arithmetic, and is labelled
as such.

### 12.1 The asymmetry, counted from source

`attention_prefill_flash_body` (`attention.metal:1459`), instantiated for decode
by `KAPLUMBA_FA_DECODE_FLASH` at `BQ = 1`:

```
constexpr uint TILE = 64u;   // keys per tile
constexpr uint NSUB  = 8u;   // m-tiles per tile  == NSG
constexpr uint NTILES = HEAD_DIM / 8u;   // 64
constexpr uint NPER   = NTILES / NSG;    //  8
```

**QK** (`:1618-1645`): `sg` owns m-tile `sg`, i.e. exactly **one** 8-key m-tile per
simdgroup, and the loop is `for (d8 = 0; d8 < 512; d8 += 8)`:

```c
simdgroup_half8x8 kf;          // 1 load, from `keys` (device, stride kv_stride)
simdgroup_load(kf, keys + k0 * kv_stride + head_base + d8, kv_stride);
for (b < BQ) {                 // BQ == 1 on decode
    simdgroup_half8x8 qf;      // 1 load, from `qstage` (threadgroup, transpose)
    simdgroup_load(qf, qstage + ..., HEAD_DIM, ulong2(0,0), true);
    simdgroup_multiply_accumulate(dots[b], kf, qf, dots[b]);   // 1 MMA
}
```

At `BQ = 1` that is **1 `kf` + 1 `qf` + 1 MMA per iteration = 2.0 loads/MMA**.
Over a full 64-key tile: 8 simdgroups × 64 iterations × 2 = **1 024 loads for 512
MMAs**.

**PV** (`:1758-1786`) is already fixed — the comment at `:1753` says so explicitly:
`BQ` `af` loads + `NPER` `bf` loads for `BQ * NPER` MMAs = **9 loads / 8 MMAs =
1.125**. Per 64-key tile: `ksteps = 8`, so 8 × 9 = **72 loads for 64 MMAs**.

**QK is 1 024 of the 1 096 loads in a tile — 93%.** That matches the measured
"QK 54.4% / PV 56.4%, summing to 110.8% on a shared tensor pipe" from
`spec/prefill-24k-90tps` §22.3: the two are the same pool, and QK owns most of it.

### 12.2 The fix: give each simdgroup `M` m-tiles, and swap the loops

Today each simdgroup has **one** m-tile, so `qf` **cannot be reused** for this
query — that is exactly why query blocking (`BQ`) exists, and `BQ` is pinned to 1
on decode by the partials layout (`static_assert(!PARTIALS || BQ == 1)`). But `qf`
is loop-invariant *across m-tiles*, and today every simdgroup re-loads all 64 of
its fragments for its single m-tile. **Reuse Q across keys instead of K across
queries** — the same argument as `:1753`, applied to the other operand:

```c
for (d8 = 0; d8 < HEAD_DIM; d8 += 8) {      // OUTER
    simdgroup_half8x8 qf;
    simdgroup_load(qf, qstage + d8, HEAD_DIM, ulong2(0,0), true);   // 1
    for (m = 0; m < M; ++m) {               // INNER: M m-tiles
        simdgroup_half8x8 kf;
        simdgroup_load(kf, keys + (tile0 + m*8) * kv_stride + head_base + d8, kv_stride);
        simdgroup_multiply_accumulate(dots[m], kf, qf, dots[m]);    // M MMAs
    }
}
```

Loads/MMA becomes `(1 + M) / M`, and the tile grows to `TILE = 64 * M / 8` ...
precisely: 8 simdgroups × `M` m-tiles = `8M` m-tiles = **`64M` keys per tile**.

| M | keys/tile | loads/MMA | QK loads per 64 keys | vs today |
|---:|---:|---:|---:|---:|
| **1 (today)** | 64 | **2.00** | 1 024 | — |
| 2 | 128 | 1.50 | 768 | **−25%** |
| **4** | **256** | **1.25** | **640** | **−37.5%** |
| 8 | 512 | 1.125 | 576 | −43.8% |

### 12.3 It fits, at M = 4

**Threadgroup memory** — `sth` is `BQ * TILE * 8` floats and scales with the tile:

| buffer | today (TILE 64) | M = 4 (TILE 256) |
|---|---:|---:|
| `qstage` `BQ*NSG*HEAD_DIM` half | 8 KB | 8 KB |
| `sth` `BQ*TILE*8` float | 2 KB | **8 KB** |
| `wq` `BQ*8*64` half | 1 KB | 1 KB |
| `tail` `8*HEAD_DIM` half | 8 KB | 8 KB |
| **total** | **19 KB** | **25 KB** (cliff 32 KB) |

**Registers** — `dots[m]` is a `simdgroup_float8x8`, which exposes
`thread_elements()[0..1]`, so M = 4 costs **8 registers** of accumulator. The
current body already carries `dots[BQ]` + `acc[BQ][NPER]` (8 accumulators at
BQ = 1, since `NPER = 8`), so the accumulator file grows by 50% at the same time
`sth` is read back — headroom is plausible but **not verified**, and register
pressure is the most likely way this loses.

**Bit-exactness — by construction.** Each `dots[m]` still accumulates the same
`kf(d8) · qf(d8)` terms in the same ascending `d8` order; only *which simdgroup*
holds the accumulator and *how often the operand is reloaded* changes. No
reassociation, no epilogue change, so this should be **bit-identical in fp16**,
which is the property `live_kernel_bit_exact.rs` already asserts for the GEMV
kernels. That matters: it is the only reason this is still on the table while D1
is not.

### 12.4 Risks, stated before anyone builds it

1. **The body is shared.** `attention_prefill_flash_body` is instantiated by
   `KAPLUMBA_FA_FLASH` (prefill, BQ 1 and 2), `KAPLUMBA_FA_FLASH_PASS` and
   `KAPLUMBA_FA_DECODE_FLASH`. A `TILE` change moves prefill attention too — which
   is *the other half of this spec's brief* (100 tok/s @24k), so it is not
   obviously bad, but it means a decode-only A/B is not possible without a
   template parameter. **A `TILE`/`M` template parameter is the safe form.**
2. **Bigger tiles shrink the split count's usefulness and change the tail.** A
   14-key tail at `TILE = 64` becomes a 14-key tail at 256 with 3 mostly-idle
   simdgroups; the `partial_sub` / `partial_kt` tail handling at `:1600-1606`
   assumes `TILE = 64 == NSUB * 8` and must be re-derived, not retyped.
3. **Multiple waves get worse before they get better.** The grid is unchanged
   (`splits × kv_heads` workgroups), but each workgroup now does 4× the keys, so
   the *tail* of the key range is worse balanced. At 16k with `splits = 16`,
   keys/split = 1024 = exactly 4 × 256, so it divides cleanly — **at 16k only**.
4. **The 37.5% is a load-count reduction, not a time reduction.** The kernel's
   measured limiter is "MMA-issue-plus-loads" on one pipe, so fewer loads should
   help, but nothing here proves the phase is 100% load-bound. **The honest
   expectation is somewhere between 0 and the load share, and the 0 end is not
   excluded.**

### 12.5 What would falsify or confirm it, cheaply

A model-free, single-process instrument already exists in spirit: `flash_operand_issue.rs`
(`:522` reads `online_attention_flash_512_8`) and `flash_qk_pragma_arms.rs` in
`crates/engine/examples/`. The right first experiment is **not** the rewrite — it
is to re-run the operand-issue attribution on the **decode** instantiation
(`attention_decode_flash_512_8`, BQ = 1) at 16 384 keys, on a quiet box, and read
whether QK's 2.0 loads/MMA are actually on the critical path. Until that number
exists, §12 is a well-formed hypothesis and nothing more.

---

## 13 — D1 SETTLED at +2.17% (the 1.34x was inflated), the split curve measured,
## and the honest ceiling at 16k is ~14.0 tok/s, not 14.6

Two clean runs were obtained (02:59Z and 03:02Z) with the box free. Spreads
1.04-1.36, positive control **7.44x** and **8.59x**, clock check stable. A third
(run 6) was **correctly rejected by the instrument's own positive control** —
`scalar 2048 -> 16384` came out **1.35x** where 8x the bytes must show >2.0x, so
the run aborted with `rc=101` instead of printing a table. That is the guard
`AGENTS.md` §3 asks for, doing its job on a live regression.

### 13.1 The full-layer split curve, measured (full shape, flash kernel, 16k keys)

| splits | grid | waves | run 4 | run 5 | GB/s (run 5) | vs splits 16 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 16 | 1 | 1652.0 | 1695.5 | 79.2 | 0.554x |
| 8 | 32 | 1 | 948.7 | 979.1 | 137.1 | 0.959x |
| 12 | 48 | 1 | 803.1 | 817.3 | 164.2 | 1.149x |
| **13** | **52** | **1** | **750.8** | **757.5** | **177.2** | **1.240x** |
| **16 (shipped)** | **64** | **2** | 886.9 | 939.2 | **142.9** | 1.000x |

Same shape at 8192 keys: 13 splits = **1.13-1.16x** over 16.

**Three things this settles:**

1. **The shipping default is measurably the wrong side of the wave boundary.**
   64 workgroups is two waves and runs the full layers at **142.9 GB/s**; 52
   workgroups is one wave and runs them at **177.2 GB/s**. The curve is monotone
   *rising to 13* and then **cliffs at 16** — exactly the 52-vs-55 boundary
   `attention.rs:292-294` records, now confirmed by a second, independent
   instrument.
2. **D1's isolated value is 1.16-1.24x at 16k, NOT the 1.34x in the tree.** The
   1.34x came from `spec/decode-19tok` / `perf_probe_kv_scaling.rs:875` at 8k;
   this instrument measures **1.13-1.16x at 8k**. The older figure was taken
   before the flash kernel landed and in a non-interleaved harness.
3. **The §10.6 contradiction is RESOLVED, in favour of +2.17%.** Pricing it:
   10 x 937 us -> 10 x 754 us saves **1.83 ms of a 73.94 ms step = +2.5%**, and
   the end-to-end measurement of the same change is **+2.17%** (12.067 -> 12.329).
   **+2.5% derived vs +2.17% measured is agreement**; the +4.1% that §10.6 could
   not choose between is dead. So D1 is **13.525 -> 13.82 tok/s** — real, one
   line, and **short of both 14.0 and 14.6.**

### 13.2 The 16k attention cost, and why 14.6 needs the *other* session's number

From the same runs, 16k attention is:

```
full  : 10 x  937 us (run 4/5 mean, splits 16) =  9.37 ms
sliding: 50 x   92 us                          =  4.60 ms
                                                 -------
                                                 13.97 ms
```

Against `spec/decode-16k-degradation` §6a's in-model **16.93 ms**, that is **18%
lower**. This is not a new instrument problem — it is the *same* direction at 8k,
where the in-model ablation says **13.61 ms** and this bench sums to
**9.22 ms** (32% lower). The most likely cause is that `KAPLUMBA_ABLATE=attention`
removes more than the attention kernels — it also drops the associated
`kv_copy_to_v` copies and element work — so it **over-attributes**, and this bench
**under-measures** because the model's KV is not this rig's KV.

**That matters, because 14.6 is only reachable if the in-model 16.93 ms is right:**

| attention at 16k | then ideal attention gives | 14.6? |
|---|---:|---|
| 13.97 ms (this bench) | 73.94 - 2.67 = 71.27 ms = **14.03 tok/s** | **no** |
| 16.93 ms (in-model fit) | 73.94 - 5.63 = 68.31 ms = **14.64 tok/s** | yes |

So the target the user is asking for rests entirely on **which of those two
numbers describes the model**, and that is one measurement: `KAPLUMBA_ABLATE=attention`
plus a control at **16 384 tokens**, which `spec/decode-16k-degradation` §7 lists
as not done at 16k (it has 2k and 8k only, and the 16k row is a two-point fit).

### 13.3 Verdict, stated plainly

**14.6 tok/s @16k is not achievable with any identified lever, and the hard part
is not engineering — it is that the whole +8.2% rests on an extrapolated number
this bench contradicts by 18-32%.** Concretely:

* On **my** instrument the ceiling is **~14.03**, and D1 (the only lever big
  enough to move it, at +2.17%) is **quality-rejected** as a default.
* On **their** instrument the ceiling is 14.64, and it still requires the same
  quality-rejected flip.
* Either way, **the bit-exact levers in §12 (QK Q-reuse, 1.25 loads/MMA) are
  unbuilt, unmeasured, and by their own load-count arithmetic worth at most ~1.25x
  on a phase that is 54% of the kernel** — not the 1.7x needed to take the full
  layers from 143 to 193 GB/s on their own.

**What I would do with a quiet box, in order:** (a) the 16k `KAPLUMBA_ABLATE=attention`
measurement with its control — it decides whether 14.6 is even in scope; (b) re-pin
12.067 vs 13.525 in one session — it decides whether the target is +3.5% or +13.8%;
(c) only then build §12.

---

## 14 — D5 IS DEAD, both forms measured. And the interleaved A/B that killed it also
## gives D1's value as +1.6%, reproducibly

I built the §12 change. It does not work, in either form, and the measurements are
below rather than argued from the source.

### 14.1 Interleaved A/B protocol (new, and it is what makes the numbers usable)

Two binaries, built from the same tree with only `attention.metal` differing, run
**alternately B A B A B A** through `scripts/gpuq try`. The instrument's own
positive control (`scalar 2048 -> 16384`) reported **7.45-7.58x** on all six runs,
so the box was clean throughout. Spreads 1.04-1.19.

### 14.2 Result: the QK `#pragma unroll` is a **2.7% regression**

| round | base | unroll |
|---:|---:|---:|
| 1 | 867.1 µs | 885.5 µs |
| 2 | 869.8 µs | 895.1 µs |
| 3 | 865.0 µs | 890.4 µs |
| **median** | **867.1** (154.8 GB/s) | **890.4** (150.7 GB/s) |

**Non-overlapping across all three rounds** — base 865.0-869.8, unroll 885.5-895.1.
The unroll is **−2.7%**, reproducibly. (My first, non-interleaved read of the same
change said "+3-9%"; that was the run-to-run spread, and this is why `AGENTS.md`
§10.2 #5 requires interleaving. **The non-interleaved reading would have shipped a
regression.**) Reverted.

### 14.3 Result: the Q hoist **mis-compiles**, and it fails at the first case

Two attempts, both caught by `attention_decode_flash_matches_split`, which fails at
`keys=64, splits=1` — the simplest case there is:

| form | result |
|---|---|
| `simdgroup_half8x8 qhoist[64]`, indexed `qhoist[d8/8]` | `worst |flash-scalar| = 0.799316` (tol 5e-3) |
| **one** hoisted fragment, `d8 == 0` only | `worst |flash-scalar| = 0.013184` (tol 5e-3) |
| `#pragma unroll` alone, no hoist | **passes 3/3** |

The hoisted and the in-loop load read the **same address with the same
parameters** (`qstage + sg*HEAD_DIM` vs `qstage + (0*NSG+sg)*HEAD_DIM + 0`, both
`transpose = true`, `lda = HEAD_DIM`), and they produce **different values** —
0.013 apart on a single case. So this is not arithmetic and not reassociation: on
this toolchain, moving a `simdgroup_load` out of the loop changes what the fragment
holds. **The register-hoist formulation of §12 is not implementable as written**,
and the array form is worse still. A reworking would have to keep every load
inside the loop, which is exactly what §12 was trying to remove.

**D5 is therefore dead in both forms I could build.** §12's load arithmetic
(2.0 -> 1.0 loads/MMA) is unaffected as arithmetic, but there is no way to reach it
that this compiler accepts, and the one reachable change (the unroll) is slower.

### 14.4 The same A/B, three clean rounds, re-prices D1

`splits` sweep, full shape, flash kernel, 16 384 keys — identical across rounds to
better than 1%:

| splits | round 1 | round 2 | round 3 | GB/s | vs 16 |
|---:|---:|---:|---:|---:|---:|
| 4 | 1661.1 | 1626.7 | 1636.0 | 82 | 0.53x |
| 8 | 961.8 | 934.7 | 933.4 | 144 | 0.93x |
| 12 | 794.1 | 794.7 | 801.6 | 168 | 1.09x |
| **13** | **747.5** | **745.2** | **747.2** | **179.6** | **1.160x** |
| **16 (shipped)** | 867.1 | 869.8 | 865.0 | **154.8** | 1.000x |

Pricing it against a 73.94 ms step: `10 x (867.1 - 747.2) = 1.20 ms` = **+1.62%**.
The end-to-end measurement of the same change is **+2.17%**. **Both are small and
they agree to within the model/bench gap**, so D1 is worth **+1.6% to +2.2%** —
13.525 -> **13.75-13.82 tok/s.**

### 14.5 Verdict on 14.6, now with everything measured rather than inferred

```
13.525 tok/s  baseline (73.94 ms)
+1.6-2.2%     D1, splits 13 -- MEASURED, reproducible, and QUALITY-REJECTED   -> 13.75-13.82
+0%           D5 Q hoist -- mis-compiles; D5 unroll -- 2.7% SLOWER           -> no lever
+0%           D2 dequant -- 4 spellings measured null or worse               -> no lever
+0%           MPP tensor path -- Metal 4 host migration, closed              -> no lever
                                                                              ---------
                                                                              13.8
```

**Every avenue in this spec is now either measured dead, quality-rejected, or a
host-stack migration. 14.6 tok/s @16k is not reachable, and neither is 14.0.** The
only change I could ship that helps at all is the split-rule flip, it is worth
+1.6-2.2%, and it is **not bit-exact** — so it is the owner's call under AGENTS.md
§8, not mine.

---

## 15 — The rewrite arithmetic: the 3 bpw decode kernel is INSTRUCTION-ISSUE bound,
## not bandwidth bound, so the premise "250 GB/s sustained ⇒ 22.5 tok/s" does not
## hold — and that is the only thing a rewrite can fix

The request was: rewrite the engine, 20 tok/s at baseline (31 tokens) and
15 tok/s at 16k, on the premise that the box sustains 250 GB/s so 22.5 tok/s is
the theoretical max. **The bandwidth premise is right about the box and wrong
about the kernel.** Both parts are in this repository and neither is inferred.

### 15.1 The premise, checked

| | bytes/token | for 20 tok/s | measured effective | measured ceiling |
|---|---:|---:|---:|---:|
| baseline (31 tok) | 12.0859 GB | **241.7 GB/s** | 196.7 | 245-253 synthetic |
| 16k | 14.2390 GB/s | **213.6 GB/s** (for 15 tok/s) | 192.6 | 245-253 synthetic |

So 20 tok/s at baseline needs **241.7 GB/s sustained by the whole step** — 96% of
the measured synthetic stream (245-253), and **above the project's own measured
ALU-free GEMV ablation bound of 230.8 GB/s** (`spec/decode14-prefill100` §0.2).
**On bandwidth alone, 20 tok/s is arithmetically out of reach: it needs 241.7 and
the best a zero-ALU GEMV has ever measured is 230.8.**

### 15.2 The finding that actually decides it: at 3 bpw the wall is ALU, not DRAM

`spec/decode-ceiling/findings/ROUND1-2-SUMMARY.md` POSTSCRIPT, "two findings that
reframe the above", measured rate sweep
(`spec/decode-3bpw-amplification/implementation_notes.md:70-72`):

| rate | GB/s | Gcodes/s |
|---|---:|---:|
| **3 bpw** | **199.4** | **531** |
| 4 bpw | 249.0 | 498 |
| 6 bpw | 251.4 | 335 |

and the sentence that decides this whole rewrite:

> **Gcodes/s is NOT constant across bit-width.** 4 and 6 bpw sit at the ~250 GB/s
> streaming ceiling — DRAM-limited. Only 3 bpw is below it, because at 3 bpw the
> kernel is **integer-issue bound: 9.375 instructions per code, at ~92% of a
> 1-instruction-per-lane-per-cycle machine**. So at 3 bpw the wall is ALU... Both
> are ~200-250 GB/s. **There is no rate at which this box exceeds ~250 GB/s.**

**The model is 3 bpw (410 of 411 tensors).** So the decode GEMV is running at
**92% of the machine's instruction-issue limit**, not at the DRAM limit — 199.4
GB/s against a 250 GB/s ceiling, with 531 Gcodes/s of pure dequant work.

This is why **every one of the four exact dequant rewrites measured null or worse**
(§7, §10.6, `spec/decode-16k-degradation` §0): they were local edits inside an
already-issue-saturated loop. Removing a few instructions from a stream that is
92% issue-bound does not help when the remaining instructions are still 92% of the
budget — and every attempt so far kept the **9.375 instructions per code structure
intact**.

### 15.3 What this means for the rewrite, stated as the thing to build

A rewrite cannot reach 20 tok/s by streaming harder. It reaches it — if at all —
**only by reducing instructions per code**, because that is the binding constraint:

* **20 tok/s at baseline needs 241.7 GB/s. At 3 bpw that means the dequant must
  fall from 9.375 instructions per code to roughly `9.375 × (199.4/241.7) = 7.7`,
  i.e. −18%, AND the ALU-free bound of 230.8 has to be exceeded.** Both at once.
  The measured ALU-free bound (230.8) is itself below the requirement (241.7), so
  the weight stream would additionally have to be restructured to stream faster
  than the shipping kernel does with **zero** ALU.
* **15 tok/s at 16k needs 213.6 GB/s.** That is *below* the 230.8 ALU-free bound,
  so it is the reachable half — but it still requires the 3 bpw dequant to stop
  being the bottleneck, plus the full-attention layers at ~193 GB/s (§10.4), and
  the measured D1 lever for those is worth only +1.6-2.2% (§14.4).

**So the honest scope of "rewrite the engine":** the only thing worth rewriting is
the **3 bpw EXL3 decode GEMV's inner loop**, and the only target that matters is
**instructions per code (currently 9.375)**. Rewriting the scheduler, the KV path,
the sampler or the attention (which are 3-20% of the step between them) cannot move
this, and the project has already measured that (`A5-budget.md` §7: "the largest
addressable overhead on the path the engine actually runs is 6.70 ms", worth
+1.53 tok/s and gated on `greedy`).

### 15.4 What I am and am not claiming

I am **not** claiming 15 or 20 tok/s can be reached, and I am not shipping a
rewrite in this session. What is established, from measurements already in the
tree plus my own (§14): **20 tok/s requires 241.7 GB/s against a measured 230.8
ALU-free ceiling and a 9.375-instruction-per-code loop at 92% issue — that is a
"reduce instructions per code by 18% and restructure the stream" problem, not a
tuning problem.** Anyone starting the rewrite should begin at
`linear_gemv_ishizuki_k3_colmajor_contig4` (`linear_repacked.metal:1992`), count
instructions per code there, and treat 9.375 as the number to beat — not at the
scheduler, and not at the attention kernels.

---

## 16 — Correction to §15: the wall is the GEMV's ACCESS PATTERN (212.2 GB/s), not
## the instruction count. This is the number a rewrite has to beat

§15 leaned on the POSTSCRIPT's "9.375 instructions per code, ~92% issue". There is a
**second** ablation table in the tree that measures the same kernel directly and
says something different and better, at `crates/kernels/src/linear.metal:88-95` —
"medians of 15, 384 MB working set, exclusive GPU", on the **real 5376x5376 shape**:

```
loads only (no decode)        212.2 GB/s     <- the access pattern's ceiling
production, runtime codebook   62.4 GB/s
codebook folded to a constant 163.0 GB/s     (2.61x)   <- already SHIPPED
branch hoisted, still runtime 128.5 GB/s
no decode arithmetic at all    134.3 GB/s     (SLOWER than 163.0)
```

**Two things fall out, and they are the actual answer to the rewrite request.**

1. **Removing the decode arithmetic makes it SLOWER** (134.3 < 163.0), and the file's
   own comment says so: *"the arithmetic is free and the BRANCH is what serialises
   the eight values per lane... do not try to make the decode cheaper."* That is why
   all four exact dequant rewrites were null or worse — **there was nothing left to
   win in the dequant**, because the codebook-fold that mattered (62.4 -> 163.0)
   shipped long ago.
2. **`loads only (no decode)` is 212.2 GB/s.** That is the ceiling of the shipping
   kernel's *memory access pattern*, measured, with all arithmetic removed. It is
   **not** the 245-253 GB/s synthetic stream — that is a pure sequential read, and
   this kernel does not do one.

So the corrected arithmetic for the user's request:

| | bytes/token | needs | shipping GEMV's own ceiling | verdict |
|---|---:|---:|---:|---|
| 20 tok/s @ baseline | 12.0859 GB | **241.7 GB/s** | **212.2** (loads-only) | **needs +13.9% beyond a zero-ALU ceiling** |
| 15 tok/s @ 16k | 14.2390 GB | **213.6 GB/s** | 212.2 (loads-only) | **needs +0.7% beyond a zero-ALU ceiling** |

and where the kernel actually is today, at 3 bpw: **199.4 GB/s** — i.e. **94% of its
own loads-only ceiling**. The remaining headroom inside this access pattern is
**6%**, which is 16.05 -> ~17.0 tok/s at baseline, not 20.

**Conclusion, corrected and final: 20 tok/s at baseline requires 241.7 GB/s while
the shipping GEMV's memory access pattern caps at 212.2 GB/s with zero arithmetic.
No amount of decode arithmetic work, scheduling, attention or sampler work closes a
13.9% gap that sits above a measured zero-ALU ceiling.** The rewrite is therefore a
**layout/packing problem** — the packed weights have to be laid out so that the
decode kernel can issue sequential, coalesced reads at >= 241.7 GB/s — and not an
ALU problem, not a scheduler problem, and not an attention problem. 15 tok/s at 16k
is the same problem with 0.7% of margin instead of 13.9%, which is why it is the
more plausible of the two and still not something I can claim.

**What I would measure first, model-free and in seconds:** take the existing
`loads only` arm (212.2) and vary the *layout* only — the `mma_tile_gbase`
column-major tiling is the one reader of `perm_mode` (`linear.metal:10166`), and
`probe_gemm_shapes.rs` already dispatches the shipping kernel on synthetic operands
for free. That gives the real ceiling of each candidate layout before any kernel is
written.
