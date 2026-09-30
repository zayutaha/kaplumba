// PROBE: is `linear_gemv_ishizuki_k3_colmajor_contig4` bound by LOAD ISSUE or by
// the DECODE ALU?  Read this file before trusting a number from it.
//
// WHY A NEW FILE, AND WHY SEPARATE FROM `linear_repacked.metal`
//
// Another agent is A/B-ing SG=4/128 and unroll factors on the *production* kernel
// in `linear_repacked.metal`. This file adds no kernel to that file, changes no
// production dispatch, and is NOT concatenated into `LinearKernels::new`
// (`crates/kernels/src/linear.rs:23`). It sits outside `crates/kernels/src/`, so
// nothing in the shipping library even sees it. The harness
// (`crates/engine/examples/probe_contig4_load_alu.rs`) compiles its OWN Metal
// library from
//     1. a preamble carrying the verbatim `decode_state*` bodies,
//     2. `linear_repacked.metal` **unmodified**, via include_str!,
//     3. this file,
// so `linear_gemv_ishizuki_k3_colmajor_contig4` -- production -- is in the same
// library, compiled by the same compiler, and measured as arm (a).
//
// THE QUESTION
//
// Two measurements in the spec appear to contradict each other:
//   * removing the dequant arithmetic (`_nod`) made it 26.7% SLOWER, which reads
//     as "the decode's dependent chains supply the ILP that hides load latency";
//   * 4 bpw -- same codebook, 1.333x fewer lane-loads per byte, 33% MORE bytes
//     -- is 22% FASTER, which reads as "the kernel is bound by load *issue*".
// From those, a ~46% load-issue / ~54% decode-ALU partition was *inferred*.
//
// The missing arm is: keep every dependent ALU op, delete the loads. That is (b).
//
// WHY THE OBVIOUS PROBE IS VOID  (read before repeating it)
//
// Commit `49814d4`'s `linear_gemv_nold_k3` -- the source of the "no weight loads,
// 1.87x, the decode is free" line in `spec/decode-20tok/tasks.md:404` -- fed the
// decode from loop-invariant 64-bit *literals*:
//
//     const ulong w0 = 0x9E3779B97F4A7C15ul;   // never varies with tk
//     v[i] = decode_state(s & 0xffffu, CB);
//
// `v[i]` is then loop-invariant, so LICM hoists the entire decode out of the tile
// loop and turns `acc0 += x0 * v[0]` into a multiply by a constant. That probe
// measured "loads gone AND decode gone" and reported the difference as the cost
// of the loads. Its conclusion ("the decode is entirely hidden") is not supported
// by that instrument: the decode was not in the loop it timed.
//
// This file fixes exactly that, in the two ways that matter:
//
//   1. Arm (b)'s synthetic words depend on the loop induction variable `tk`, so
//      nothing in its decode is loop-invariant and LICM *cannot* hoist it. That
//      is a compiler invariant, not a hope -- and it is checked: arm (e) is the
//      same kernel with `tk` removed again, i.e. it IS the hoisted/decode-free
//      variant, so if (b) had been hoisted then T(b) == T(e) and the partition
//      would come out as a flat contradiction rather than a number.
//   2. Arm (e) is that control. So T(e) = loop floor, T(b) - T(e) = a *measured*
//      decode cost, and T(a) - T(b) = the cost of the loads. Both differences are
//      real if the decode is live in both, which the harness verifies.
//
// WHY EVERY ARM IS A LITERAL COPY, NOT A TEMPLATE
//
// The first version of this file factored the arms into one `template<int ARM>`
// function and passed the reduction array in as a `threadgroup float *`, because
// MSL forbids a `threadgroup` variable in a non-kernel function. Arm (r) came out
// **bitwise identical to production but 12.6% slower**, so the parameterisation
// itself was perturbing the schedule and every arm would have inherited that. So
// each arm below is a full top-level kernel holding its own
// `threadgroup float red[SG][16]`, and each differs from
// `linear_gemv_ishizuki_k3_colmajor_contig4` (`linear_repacked.metal:1101`) only
// in the lines marked `PROBE-DELTA`. The harness's FIDELITY gate compares (r)
// against (a) bitwise before it will report any timing at all.
//
// The `half2` bitcast form of `decode_state_mcg` (notes 19.2) is what production
// uses; `decode_state_mcg` is transcribed verbatim in the harness preamble, so
// (a) and (r) run the real one.

// ---------------------------------------------------------------------------
// Arm definitions
// ---------------------------------------------------------------------------
//   (a) linear_gemv_ishizuki_k3_colmajor_contig4   production, in linear_repacked
//   (r) probe_contig4_replica        PROBE-DELTA: none. Fidelity gate.
//   (b) probe_contig4_noload         PROBE-DELTA: 8 weight loads -> ALU.
//   (e) probe_contig4_alufloor       PROBE-DELTA: words made loop-invariant.
//   (f) probe_contig4_noshift        PROBE-DELTA: (b) + constant shift amount.
//   (g) probe_contig4_shiftonly      PROBE-DELTA: (b) + decode -> one AND.
//   (c) probe_contig4_nodecode       PROBE-DELTA: decode -> one AND, loads kept.
//   (d) probe_contig4_l1weight       PROBE-DELTA: column folded to 4.
//
// (f) and (g) decompose the decode cost. `uint(A0 >> (64u - rad))` with `rad`
// varying per element is a 64-bit barrel shift; `decode_state_mcg` is the
// mul/xor/half2-add body after it. Whichever of the two dominates is where the
// next hour of work belongs, and that is not a question the two spec
// measurements can answer.

static inline float nod_stub(uint s) { return float(s & 1u); }

// One mul + one xor. Cheap (16 ops per 4-tile iteration against ~300) and
// trivially reproducible on the host, which the CPU cross-check relies on.
static inline uint probe_word(uint seed, uint tile) {
    return seed ^ (tile * 0x9E3779B9u);
}

#define PROBE_ARGS                                                                   \
    const device ushort *packed [[buffer(0)]],                                       \
    const device half *x [[buffer(1)]],                                              \
    device float *y [[buffer(2)]],                                                   \
    constant const uint *shape [[buffer(3)]],                                        \
    uint lid [[thread_position_in_threadgroup]],                                     \
    uint lane [[thread_index_in_simdgroup]],                                        \
    uint sg [[simdgroup_index_in_threadgroup]],                                      \
    uint tgid [[threadgroup_position_in_grid]],                                      \
    uint ntg [[threadgroups_per_grid]]

// Seeds ride in buffer(3) after the shape: { K, N, seed_a, seed_b }. Same binding
// list and same dispatch for every arm, production included. Never wired to the
// engine.

// ===========================================================================
// (r) replica -- MUST stay byte-for-byte the production body.
// ===========================================================================
kernel void probe_contig4_replica(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const device uint *t0 = column + (tk + 0u) * stride;
            const device uint *t1 = column + (tk + 1u) * stride;
            const device uint *t2 = column + (tk + 2u) * stride;
            const device uint *t3 = column + (tk + 3u) * stride;

            const ulong w0a = t0[j0], w0b = t0[j1];
            const ulong w1a = t1[j0], w1b = t1[j1];
            const ulong w2a = t2[j0], w2b = t2[j1];
            const ulong w3a = t3[j0], w3b = t3[j1];

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device uint *t0 = column + tk * stride;
            const ulong a0 = t0[j0], a1 = t0[j1];
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// (b) NO weight loads, FULL decode.  <- the arm that had never been run.
//
// PROBE-DELTA vs replica, exactly four lines: the eight `w?a/w?b` initialisers
// become ALU on (seed, tile index). `tile` is `tk + q`, the induction variable,
// so the whole `A? -> shift -> decode_state_mcg -> FMA` chain stays inside the
// loop and cannot be hoisted. Everything after that -- the 64-bit shifts, the
// mask, all 32 decodes, all 32 FMAs, the reduction -- is untouched.
// ===========================================================================
kernel void probe_contig4_noload(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    const uint SEED_A = shape[2];   // PROBE-DELTA
    const uint SEED_B = shape[3];   // PROBE-DELTA

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const ulong w0a = probe_word(SEED_A, tk + 0u), w0b = probe_word(SEED_B, tk + 0u);  // PROBE-DELTA
            const ulong w1a = probe_word(SEED_A, tk + 1u), w1b = probe_word(SEED_B, tk + 1u);  // PROBE-DELTA
            const ulong w2a = probe_word(SEED_A, tk + 2u), w2b = probe_word(SEED_B, tk + 2u);  // PROBE-DELTA
            const ulong w3a = probe_word(SEED_A, tk + 3u), w3b = probe_word(SEED_B, tk + 3u);  // PROBE-DELTA

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const ulong a0 = probe_word(SEED_A, tk), a1 = probe_word(SEED_B, tk);   // PROBE-DELTA
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// (e) NO weight loads, decode LOOP-INVARIANT.  The `49814d4` probe's shape,
// kept deliberately as the control that measures what "decode deleted" costs.
//
// PROBE-DELTA vs (b): `tk + q` -> `0u`. This is the arm LICM can hoist, so
// T(e) is the floor: the loop, the x loads, the 64-bit shifts, the mask, the
// 32 FMAs and the reduction, with the 32 decodes reduced to constants.
// If T(b) were also hoisted, T(b) would equal this.
// ===========================================================================
kernel void probe_contig4_alufloor(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    const uint SEED_A = shape[2];   // PROBE-DELTA
    const uint SEED_B = shape[3];   // PROBE-DELTA

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const ulong w0a = probe_word(SEED_A, 0u), w0b = probe_word(SEED_B, 0u);   // PROBE-DELTA
            const ulong w1a = probe_word(SEED_A, 0u), w1b = probe_word(SEED_B, 0u);   // PROBE-DELTA
            const ulong w2a = probe_word(SEED_A, 0u), w2b = probe_word(SEED_B, 0u);   // PROBE-DELTA
            const ulong w3a = probe_word(SEED_A, 0u), w3b = probe_word(SEED_B, 0u);   // PROBE-DELTA

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const ulong a0 = probe_word(SEED_A, 0u), a1 = probe_word(SEED_B, 0u);   // PROBE-DELTA
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// (c) weight loads KEPT, decode replaced by one AND.
//
// PROBE-DELTA vs replica: the 32 `decode_state_mcg` calls become `nod_stub`. The
// 64-bit shifts and the mask stay, so the loaded words are still consumed and the
// loads cannot be eliminated -- which is the whole point, and what the `_nod`
// kernels in `linear.metal` also do. This is the replication of the `_nod`
// result on `contig4`.
// ===========================================================================
kernel void probe_contig4_nodecode(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const device uint *t0 = column + (tk + 0u) * stride;
            const device uint *t1 = column + (tk + 1u) * stride;
            const device uint *t2 = column + (tk + 2u) * stride;
            const device uint *t3 = column + (tk + 3u) * stride;

            const ulong w0a = t0[j0], w0b = t0[j1];
            const ulong w1a = t1[j0], w1b = t1[j1];
            const ulong w2a = t2[j0], w2b = t2[j1];
            const ulong w3a = t3[j0], w3b = t3[j1];

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = nod_stub(uint(A0 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
                v1[i] = nod_stub(uint(A1 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
                v2[i] = nod_stub(uint(A2 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
                v3[i] = nod_stub(uint(A3 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device uint *t0 = column + tk * stride;
            const ulong a0 = t0[j0], a1 = t0[j1];
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = nod_stub(s & 0xffffu);   // PROBE-DELTA
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// (f) NO loads, decode KEPT, but the 64-bit VARIABLE shift replaced by a
// constant shift.
//
// Why: `uint(A0 >> (64u - rad))` with `rad = i*KA + 16` unrolled to 16,19,22,25,
// ...37 is a barrel shift of a 64-bit value by a different amount per element.
// That is not one instruction on a 32-bit ALU; it is a shift of each half plus a
// select, per element, 32 times per 4-tile iteration. This arm keeps
// `decode_state_mcg` fully live (still fed `tk`-varying words) and only pins the
// shift amount, so T(b) - T(f) is the *variable shift* cost alone and
// T(f) - T(e) is the `decode_state_mcg` body cost alone.
//
// If the barrel shift dominates, the fix is a layout change that makes `rad`
// compile-time (which is what `crates/engine/src/exl3.rs:687` `extract_ishizuki_lane_codes`
// was aiming at), not a cheaper decode.
// ===========================================================================
kernel void probe_contig4_noshift(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    const uint SEED_A = shape[2];   // PROBE-DELTA
    const uint SEED_B = shape[3];   // PROBE-DELTA

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const ulong w0a = probe_word(SEED_A, tk + 0u), w0b = probe_word(SEED_B, tk + 0u);  // PROBE-DELTA
            const ulong w1a = probe_word(SEED_A, tk + 1u), w1b = probe_word(SEED_B, tk + 1u);  // PROBE-DELTA
            const ulong w2a = probe_word(SEED_A, tk + 2u), w2b = probe_word(SEED_B, tk + 2u);  // PROBE-DELTA
            const ulong w3a = probe_word(SEED_A, tk + 3u), w3b = probe_word(SEED_B, tk + 3u);  // PROBE-DELTA

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = 48u;   // PROBE-DELTA: constant, not `i * KA + 16u`
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const ulong a0 = probe_word(SEED_A, tk), a1 = probe_word(SEED_B, tk);   // PROBE-DELTA
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = 48u;   // PROBE-DELTA
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// (g) NO loads, decode body replaced by nod_stub, VARIABLE shift kept.
//
// The complement of (f): T(b) - T(g) is the `decode_state_mcg` body cost with the
// barrel shift untouched. (f) and (g) must sum to T(b) - T(e) or the additive model
// is wrong.
// ===========================================================================
kernel void probe_contig4_shiftonly(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    const uint SEED_A = shape[2];   // PROBE-DELTA
    const uint SEED_B = shape[3];   // PROBE-DELTA

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const device uint *column = (const device uint *)packed + tn * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const ulong w0a = probe_word(SEED_A, tk + 0u), w0b = probe_word(SEED_B, tk + 0u);  // PROBE-DELTA
            const ulong w1a = probe_word(SEED_A, tk + 1u), w1b = probe_word(SEED_B, tk + 1u);  // PROBE-DELTA
            const ulong w2a = probe_word(SEED_A, tk + 2u), w2b = probe_word(SEED_B, tk + 2u);  // PROBE-DELTA
            const ulong w3a = probe_word(SEED_A, tk + 3u), w3b = probe_word(SEED_B, tk + 3u);  // PROBE-DELTA

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = nod_stub(uint(A0 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
                v1[i] = nod_stub(uint(A1 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
                v2[i] = nod_stub(uint(A2 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
                v3[i] = nod_stub(uint(A3 >> (64u - rad)) & 0xffffu);   // PROBE-DELTA
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const ulong a0 = probe_word(SEED_A, tk), a1 = probe_word(SEED_B, tk);   // PROBE-DELTA
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = nod_stub(s & 0xffffu);   // PROBE-DELTA
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ===========================================================================
// (d) weight loads KEPT, but from a ~4-column window instead of tiles_n.
//
// PROBE-DELTA vs replica: `tn` -> `tn & 3`. Identical load count, identical
// address arithmetic, identical everything -- but the footprint is 129 KB at
// gate_up (L2-resident) instead of 2.6 GB. This separates "cost of issuing the
// loads" from "cost of them missing cache", which arms (b) and (c) cannot.
// ===========================================================================
kernel void probe_contig4_l1weight(PROBE_ARGS) {
    constexpr uint KA = 3u;
    constexpr int CB = 1;
    constexpr uint RING = 256u * KA;
    constexpr uint WORDS = RING / 32u;
    const uint K = shape[0];
    const uint N = shape[1];
    const uint tiles_n = N / 16u;
    const uint tiles_k = K / 16u;
    constexpr int SG = 8;
    threadgroup float red[SG][16];
    const uint r0 = (lane % 4u) * 2u;

    const uint lo0 = (lane * 8u + 1u) * 3u + RING - 16u;
    const uint sh0 = lo0 % 32u;
    const uint hi0 = lo0 / 32u;
    const uint j0 = hi0 >= WORDS ? hi0 - WORDS : hi0;
    const uint j1 = j0 + 1u >= WORDS ? j0 + 1u - WORDS : j0 + 1u;

    for (uint tn = tgid; tn < tiles_n; tn += ntg) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        const uint col = tn & 3u;   // PROBE-DELTA
        const device uint *column = (const device uint *)packed + col * (tiles_k * WORDS);
        const uint stride = WORDS;

        const uint step = SG * 4u;
        const uint k_unroll_end = (tiles_k / step) * step;

        uint tk = sg * 4u;
        for (; tk < k_unroll_end; tk += step) {
            const device uint *t0 = column + (tk + 0u) * stride;
            const device uint *t1 = column + (tk + 1u) * stride;
            const device uint *t2 = column + (tk + 2u) * stride;
            const device uint *t3 = column + (tk + 3u) * stride;

            const ulong w0a = t0[j0], w0b = t0[j1];
            const ulong w1a = t1[j0], w1b = t1[j1];
            const ulong w2a = t2[j0], w2b = t2[j1];
            const ulong w3a = t3[j0], w3b = t3[j1];

            const ulong A0 = ((w0a << 32) | w0b) << sh0;
            const ulong A1 = ((w1a << 32) | w1b) << sh0;
            const ulong A2 = ((w2a << 32) | w2b) << sh0;
            const ulong A3 = ((w3a << 32) | w3b) << sh0;

            const device half *xr0 = x + (tk + 0u) * 16u + r0;
            const float p00 = float(xr0[0]), p01 = float(xr0[1]), p02 = float(xr0[8]), p03 = float(xr0[9]);
            const device half *xr1 = x + (tk + 1u) * 16u + r0;
            const float p10 = float(xr1[0]), p11 = float(xr1[1]), p12 = float(xr1[8]), p13 = float(xr1[9]);
            const device half *xr2 = x + (tk + 2u) * 16u + r0;
            const float p20 = float(xr2[0]), p21 = float(xr2[1]), p22 = float(xr2[8]), p23 = float(xr2[9]);
            const device half *xr3 = x + (tk + 3u) * 16u + r0;
            const float p30 = float(xr3[0]), p31 = float(xr3[1]), p32 = float(xr3[8]), p33 = float(xr3[9]);

            float v0[8], v1[8], v2[8], v3[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                v0[i] = decode_state_mcg(uint(A0 >> (64u - rad)) & 0xffffu);
                v1[i] = decode_state_mcg(uint(A1 >> (64u - rad)) & 0xffffu);
                v2[i] = decode_state_mcg(uint(A2 >> (64u - rad)) & 0xffffu);
                v3[i] = decode_state_mcg(uint(A3 >> (64u - rad)) & 0xffffu);
            }

            acc0 += p00 * v0[0] + p01 * v0[1] + p02 * v0[2] + p03 * v0[3];
            acc1 += p00 * v0[4] + p01 * v0[5] + p02 * v0[6] + p03 * v0[7];
            acc0 += p10 * v1[0] + p11 * v1[1] + p12 * v1[2] + p13 * v1[3];
            acc1 += p10 * v1[4] + p11 * v1[5] + p12 * v1[6] + p13 * v1[7];
            acc0 += p20 * v2[0] + p21 * v2[1] + p22 * v2[2] + p23 * v2[3];
            acc1 += p20 * v2[4] + p21 * v2[5] + p22 * v2[6] + p23 * v2[7];
            acc0 += p30 * v3[0] + p31 * v3[1] + p32 * v3[2] + p33 * v3[3];
            acc1 += p30 * v3[4] + p31 * v3[5] + p32 * v3[6] + p33 * v3[7];
        }

        for (tk = k_unroll_end + sg; tk < tiles_k; tk += SG) {
            const device uint *t0 = column + tk * stride;
            const ulong a0 = t0[j0], a1 = t0[j1];
            const ulong A0 = ((a0 << 32) | a1) << sh0;
            const device half *xr0 = x + tk * 16u + r0;
            const float p0 = float(xr0[0]), p1 = float(xr0[1]), p2 = float(xr0[8]), p3 = float(xr0[9]);
            float v[8];
            #pragma unroll
            for (uint i = 0; i < 8u; ++i) {
                const uint rad = i * KA + 16u;
                const uint s = uint(A0 >> (64u - rad));
                v[i] = decode_state_mcg(s & 0xffffu);
            }
            acc0 += p0 * v[0] + p1 * v[1] + p2 * v[2] + p3 * v[3];
            acc1 += p0 * v[4] + p1 * v[5] + p2 * v[6] + p3 * v[7];
        }

        acc0 += simd_shuffle_xor(acc0, 1u);
        acc0 += simd_shuffle_xor(acc0, 2u);
        acc1 += simd_shuffle_xor(acc1, 1u);
        acc1 += simd_shuffle_xor(acc1, 2u);
        if (lane % 4u == 0u) {
            red[sg][lane / 4u] = acc0;
            red[sg][lane / 4u + 8u] = acc1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid < 16u) {
            float sum = 0.0f;
            for (int g = 0; g < SG; ++g) {
                sum += red[g][lid];
            }
            y[tn * 16u + lid] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}