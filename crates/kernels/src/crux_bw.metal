#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// `spec/decode-20tok-crux` streaming kernels.
//
// Three axes are kept deliberately SEPARATE, because `spec/decode-19tok`
// reports that "bandwidth is flat from 8 to 32 bytes in flight per thread" and
// that sweep only reached 20,480 threads while the device allows 81,920 — so
// the two were confounded and the conclusion ("request-rate wall") may be an
// artefact of never moving the occupancy axis.
//
//   axis A  threads in the grid        (occupancy)          -> stream_u*
//   axis B  bytes per load instruction (load width)         -> stream_u1/u2/u4
//   axis C  independent loads in flight per thread (unroll) -> stream_u4_u1..u8
//
// The three are *not* interchangeable, and that is the point:
//
//  * A raises the number of memory requests issued concurrently, at constant
//    requests-per-byte. If bandwidth is flat in A, the part is not request-limited.
//  * B changes requests-per-byte at constant bytes and constant threads. If
//    bandwidth is flat in B, requests-per-byte is not the limit either.
//  * C raises requests-per-thread without changing either, which separates
//    "ILP inside a thread" from "more threads".
//
// All arms visit the same bytes in the same grid-stride order and produce the
// same checksum, so an arm that read a different range cannot be compared
// against another. `crux_bandwidth.rs` asserts that equality as a gate before
// printing any number.
// ---------------------------------------------------------------------------

// --- axis B: load width, one load in flight -------------------------------
// 4 B / 8 B / 16 B per load, identical bytes, identical visit order.
kernel void stream_u1(const device uint *data [[buffer(0)]],
                      device uint *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint total [[threads_per_grid]]) {
    uint s = 0u;
    for (uint i = gid; i < count; i += total) s += data[i];
    out[gid] = s;
}

kernel void stream_u2(const device uint2 *data [[buffer(0)]],
                      device uint *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint total [[threads_per_grid]]) {
    uint2 s = uint2(0u);
    for (uint i = gid; i < count; i += total) s += data[i];
    out[gid] = s.x + s.y;
}

kernel void stream_u4(const device uint4 *data [[buffer(0)]],
                      device uint *out [[buffer(1)]],
                      constant uint &count [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint total [[threads_per_grid]]) {
    uint4 s = uint4(0u);
    for (uint i = gid; i < count; i += total) s += data[i];
    out[gid] = s.x + s.y + s.z + s.w;
}

// --- axis C: UN independent uint4 loads in flight per thread ----------------
// Same total bytes as `stream_u4` over the same range: thread `gid` consumes
// elements `gid, gid+total, gid+2*total, ...` in U batches of U consecutive
// strided positions. The batch's U loads are independent and are summed into
// U separate accumulators, so the compiler may issue them together. That is the
// only difference from `stream_u4` and it moves no extra bytes.
//
// The tail is the exact remainder, handled scalar-wise in a *separate* pass over
// the same elements, so every element is counted exactly once and the checksum
// is order-independent (u32 addition is associative modulo 2^32).
#define STREAM_U4_UNROLL(U, ACCS)                                                   \
    kernel void stream_u4_u##U(const device uint4 *data [[buffer(0)]],              \
                               device uint *out [[buffer(1)]],                      \
                               constant uint &count [[buffer(2)]],                  \
                               uint gid [[thread_position_in_grid]],                \
                               uint total [[threads_per_grid]]) {                   \
        uint4 acc[U];                                                               \
        for (uint k = 0; k < U; ++k) acc[k] = uint4(0u);                            \
        uint i = gid;                                                               \
        for (; i + (U - 1) * total < count; i += U * total) {                       \
            uint4 v[U];                                                             \
            for (uint k = 0; k < U; ++k) v[k] = data[i + k * total];                \
            for (uint k = 0; k < U; ++k) acc[k] += v[k];                            \
        }                                                                           \
        uint4 t = uint4(0u);                                                        \
        for (; i < count; i += total) t += data[i];                                 \
        uint4 s = t;                                                                \
        for (uint k = 0; k < U; ++k) s += acc[k];                                   \
        out[gid] = s.x + s.y + s.z + s.w;                                           \
    }

STREAM_U4_UNROLL(2,)
STREAM_U4_UNROLL(4,)
STREAM_U4_UNROLL(8,)
STREAM_U4_UNROLL(16,)

// --- axis B+C: unrolled uint1 and uint2 ------------------------------------
// `stream_u1_u4` reads 4 independent u32 per iteration: 16 B in flight but
// FOUR requests. Comparing it against `stream_u4` (16 B in flight, ONE request)
// at identical bytes and identical thread count separates "bytes in flight"
// from "requests in flight" — which is the whole question, because
// `decode-19tok` reported them as the same axis.
#define STREAM_U1_UNROLL(U)                                                        \
    kernel void stream_u1_u##U(const device uint *data [[buffer(0)]],             \
                               device uint *out [[buffer(1)]],                     \
                               constant uint &count [[buffer(2)]],                 \
                               uint gid [[thread_position_in_grid]],               \
                               uint total [[threads_per_grid]]) {                  \
        uint acc[U];                                                               \
        for (uint k = 0; k < U; ++k) acc[k] = 0u;                                  \
        uint i = gid;                                                              \
        for (; i + (U - 1) * total < count; i += U * total) {                      \
            uint v[U];                                                             \
            for (uint k = 0; k < U; ++k) v[k] = data[i + k * total];               \
            for (uint k = 0; k < U; ++k) acc[k] += v[k];                           \
        }                                                                          \
        uint t = 0u;                                                               \
        for (; i < count; i += total) t += data[i];                                \
        for (uint k = 0; k < U; ++k) t += acc[k];                                  \
        out[gid] = t;                                                              \
    }

STREAM_U1_UNROLL(4)
STREAM_U1_UNROLL(8)

// --- S: independent ADDRESS STREAMS ----------------------------------------
// One threadgroup per stream, each stream reading its own disjoint slice, so the
// TLB / DRAM page-buffer sees `S` concurrent streams instead of 1. `decode-19tok`
// R3 asks for this sweep from 1 to 64 and it is the one axis that has never been
// run in this repo. Grid must be a multiple of `streams`.
//
// Slice `s` of `S` starts at `(count / S) * s`, so slices are equal-sized and
// the union is exactly `count`. The per-thread offset within its slice is the
// threadgroup id minus `s * tgs_per_stream`, so the visit order inside a slice
// is the ordinary grid-stride order and the checksum equals the 1-stream one.
kernel void stream_u4_s(const device uint4 *data [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant uint &count [[buffer(2)]],
                        constant uint &streams [[buffer(3)]],
                        constant uint &tgs_per_stream [[buffer(4)]],
                        uint gid [[thread_position_in_grid]],
                        uint tgid [[threadgroup_position_in_grid]],
                        uint tpos [[thread_position_in_threadgroup]],
                        uint total [[threads_per_grid]]) {
    uint s = tgid / tgs_per_stream;
    uint tg_size = total / (tgs_per_stream * streams); // threads per threadgroup
    uint local_tg = tgid - s * tgs_per_stream;
    uint local_gid = local_tg * tg_size + tpos;
    uint slice = count / streams;
    uint base = s * slice;
    uint4 acc = uint4(0u);
    for (uint i = local_gid; i < slice; i += tgs_per_stream * tg_size)
        acc += data[base + i];
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}
