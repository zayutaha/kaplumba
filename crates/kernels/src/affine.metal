// spec/mtp-40tps T5b: GEMV over MLX affine-4bit packed weights.
//
// W[n, k] = dequant(nibble(n, k)) = nibble * scales[n, k/group] + biases[n, k/group]
// y[n]    = sum_k W[n, k] * x[k]
//
// One threadgroup per output row. This is a reference-correctness path, not a
// tuned kernel: the drafter is 0.28 GB resident, so an unoptimised GEMV still
// costs ~1.4 ms/step at our measured ~200 GB/s.

using namespace metal;

kernel void affine4_gemv(
    const device uint *w [[buffer(0)]],
    const device half *scales [[buffer(1)]],
    const device half *biases [[buffer(2)]],
    const device half *x [[buffer(3)]],
    device half *y [[buffer(4)]],
    constant uint &n_rows [[buffer(5)]],
    constant uint &k_dim [[buffer(6)]],
    constant uint &group_size [[buffer(7)]],
    uint3 tg_position [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]) {
  uint n = tg_position.x;
  if (n >= n_rows) {
    return;
  }
  uint words_per_row = k_dim / 8;
  const device uint *row = w + n * words_per_row;

  float acc = 0.0f;
  for (uint k = tid.x; k < k_dim; k += 256) {
    uint word = row[k / 8];
    uint nibble = (word >> (4 * (k % 8))) & 0xF;
    uint g = k / group_size;
    float scale = float(scales[n * (k_dim / group_size) + g]);
    float bias = float(biases[n * (k_dim / group_size) + g]);
    float w_det = float(nibble) * scale + bias;
    acc += w_det * float(x[k]);
  }

  threadgroup float partials[256];
  partials[tid.x] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 128; stride > 0; stride >>= 1) {
    if (tid.x < stride) {
      partials[tid.x] += partials[tid.x + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (tid.x == 0) {
    y[n] = half(partials[0]);
  }
}
