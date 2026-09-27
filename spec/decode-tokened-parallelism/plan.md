# Plan: decode token-1 parallelism

## Where the time goes (measured, not estimated)

`KAPLUMBA_PROFILE_PHASES=1`, decode step, 134-token context:

```
embed 0.17 ms | layers 783 ms | lm_head 9.3 ms
layers: cpu_encode 0.00 ms, gpu_wait 783 ms
```

Linear GEMVs account for ~220 ms. The rest is two serial kernels.

## Fix 1: parallel RMS reduction

Current (`elements.metal:92`), one thread per row:

```
if (gid >= params.rows) return;
for i in 0..count: sum += v*v          // serial
for i in 0..count: out[i] = v*inv*w[i] // serial
```

Add a second kernel `rms_norm_weighted_tg` with one **threadgroup** per row:

```
kernel void rms_norm_weighted_tg(..., uint lid [[thread_position_in_threadgroup]],
                                 uint tgid [[threadgroup_position_in_grid]],
                                 uint tgdim [[threads_per_threadgroup]]) {
  if (tgid >= params.rows) return;
  threadgroup float partial[256];
  uint base = tgid * params.width;
  float sum = 0.0f;
  // chunked: each thread accumulates a strided subset, which for width 5376 is
  // 21 elements per thread at 256 threads
  for (uint i = lid; i < params.width; i += tgdim)
    sum += float(input[base + i]) * float(input[base + i]);
  partial[lid] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = tgdim / 2; stride > 0; stride >>= 1) {
    if (lid < stride) partial[lid] += partial[lid + stride];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  float inverse = rsqrt(partial[0] / float(params.width) + params.eps);
  for (uint i = lid; i < params.width; i += tgdim)
    output[base + i] = half(float(input[base + i]) * inverse * float(weight[i]));
}
```

Dispatch choice in `encode_rms_norm`: `rows < 32` uses the threadgroup kernel
with `rows` groups of 256; otherwise the existing per-row kernel. Prefill
(`rows = tokens`, typically > 32) keeps today's path, so it cannot regress.

`rms_norm_unweighted` gets the same treatment.

## Fix 2: split-K attention

Current (`attention.metal:14`): `queries * q_heads` threads, each with
`float accum[512]`, scanning every key serially. For decode that is 8 threads.

Two-pass flash-decoding:

Pass A — `attention_partial`: grid `(q_heads, key_splits)`. Each threadgroup
handles a slice of keys for one (query, head) and writes
`partial_max[split]`, `partial_sum[split]`, `partial_acc[split * head_dim]`.

```
for each key k in this split (strided over the slice):
    score = dot(q, k) * scale
    if key_positions[k] stale or out of window: skip     // existing window check
    m_new = max(m, score)
    correction = exp(m - m_new)
    l = l * correction + exp(score - m_new)
    acc = acc * correction + exp(score - m_new) * v
```

Pass B — `attention_combine`: one threadgroup per (query, head), reduce the
splits:

```
m_global = max_s partial_max[s]
l_total  = sum_s partial_sum[s] * exp(partial_max[s] - m_global)
out      = sum_s partial_acc[s] * exp(partial_max[s] - m_global) / l_total
```

This also removes the 2 KB per-thread `accum[512]`, which is what caps
occupancy. No fp32 atomics; the partials go through a small scratch buffer sized
`q_heads * splits * (head_dim + 2)`.

Dispatch `splits` from the key count, e.g. `splits = clamp(key_count / 64, 1, 32)`.
Keep the existing serial kernel for `queries * q_heads >= 1024` so prefill, where
there are already plenty of threads, is untouched.

## Sequencing

T1 (time attention) then T2/T3 (norms, the measured 179 ms) then T4/T5
(attention). Re-measure the layer stack after the norms, since that alone should
take 783 ms to ~600 ms, and the attribution will be much clearer before
touching attention.

## Requirement mapping

| Requirement | Task |
|---|---|
| R1 parallel norm reduction | T2 |
| R1 both norm variants | T3 |
| R2 split-K attention | T4 |
| R3 drop per-thread `accum[512]` | T5 |
| R4 dispatch on rows/queries | T2, T3, T4 |
| R5 per-kernel costs | T1, T6 |
| R6 float tolerance | T10 |
