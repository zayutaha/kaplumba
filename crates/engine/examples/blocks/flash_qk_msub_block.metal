for (uint base = range_begin; base < range_end; base += TILEQ) {
        const uint live_q = min(TILEQ, range_end - base);
        // The one m-tile of this span that straddles `range_end`, or NSUBQ when
        // none does. Only the *last* live m-tile can straddle, so a single
        // staging buffer serves the whole span.
        const uint psub = (live_q % 8u != 0u) ? (live_q - 1u) / 8u : NSUBQ;
        if (psub < NSUBQ) {
            // Guard: the previous span's PV may still be reading `tail`.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint k0 = base + psub * 8u;
            for (uint i = tid; i < 8u * uint(HEAD_DIM); i += 256u) {
                const uint row = i / uint(HEAD_DIM);
                const uint col = i % uint(HEAD_DIM);
                tail[i] = (k0 + row < range_end)
                    ? keys[(k0 + row) * kv_stride + head_base + col]
                    : half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // ---- QK over the whole TILEQ span ------------------------------------
        //
        // Simdgroup `sg` owns MSUB m-tiles (`sg`, `sg + NSUB`, ...) and the
        // `d8` loop is the OUTER loop, so one `qf` fragment -- loop-invariant
        // across the whole key range, and therefore the fragment the kernel
        // reloads most pointlessly -- is loaded once and feeds MSUB MMAs. That
        // takes the QK fragment loads per MMA from `1 + 1/1 = 2` at MSUB = 1 to
        // `1 + 1/MSUB`.
        //
        // This needs MSUB live accumulators (2 registers/lane each, so 8 at
        // MSUB = 4, against the 16 that `acc[BQ][NPER]` already holds) and it
        // does NOT hoist Q into registers -- that was `AblationScaffold`'s arm 7
        // at -92%, 64 registers/lane of `qreg`.
        //
        // `sth` is TILEQ keys wide because every m-tile is published before the
        // first softmax reads any of them.
        if (sg <= psub) {
            simdgroup_float8x8 dots[BQ][MSUB];
            bool alive[MSUB];
#pragma unroll
            for (uint u = 0; u < uint(MSUB); ++u) {
                // `k0` depends only on `u`, so `alive` is loop-invariant in `d8`.
                // The `staged` m-tile reads `tail`, the rest the device; keys past
                // `range_end` drop out here.
                const uint mt = u * uint(NSUB) + sg;
                const uint k0 = base + mt * 8u;
                alive[u] = (mt == psub) || (k0 + 8u <= range_end);
            }
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
#pragma unroll
                for (uint u = 0; u < uint(MSUB); ++u) {
                    dots[b][u] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                }
            }
            for (int d8 = 0; d8 < HEAD_DIM; d8 += 8) {
#pragma unroll
                for (uint b = 0; b < uint(BQ); ++b) {
                    if (q0 + b >= params.queries) {
                        continue;
                    }
                    // ABL bit 2: the Q fragment load. Replaced by a filled
                    // constant, which keeps the instruction count of the MMA and
                    // changes only where its operand comes from.
                    simdgroup_half8x8 qf;
                    if ((ABL & 2u) != 0u) {
                        qf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
                    } else {
                        simdgroup_load(qf,
                                       qstage + (b * uint(NSG)) * uint(HEAD_DIM) + d8,
                                       HEAD_DIM, ulong2(0, 0), true);
                    }
                    // One K element per owned m-tile, reused by every query --
                    // the reuse query blocking buys, and now by every `u` too.
#pragma unroll
                    for (uint u = 0; u < uint(MSUB); ++u) {
                        if (!alive[u]) {
                            continue;
                        }
                        const uint mt = u * uint(NSUB) + sg;
                        simdgroup_half8x8 kf;
                        if ((ABL & 2u) != 0u) {
                            kf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
                        } else if (mt == psub) {
                            simdgroup_load(kf, tail + d8, HEAD_DIM);
                        } else {
                            simdgroup_load(kf,
                                           keys + (base + mt * 8u) * kv_stride
                                               + head_base + d8,
                                           kv_stride);
                        }
                        // ABL bit 1: the QK MMA itself.
                        if ((ABL & 1u) != 0u) {
                            simdgroup_multiply_accumulate(dots[b][u], kf, qf, dots[b][u]);
                        }
                    }
                }
            }
            // Publish [query][key in span][head]. Keys past `range_end` are
            // forced to NEG by the `live` test below, so a stale or zero slot
            // is harmless.
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
                if (q0 + b >= params.queries) {
                    continue;
                }
#pragma unroll
                for (uint u = 0; u < uint(MSUB); ++u) {
                    if (!alive[u]) {
                        continue;
                    }
                    sth[(b * TILEQ + u * TILE + sg * 8u + fm) * 8u + fn] =
                        dots[b][u].thread_elements()[0];
                    sth[(b * TILEQ + u * TILE + sg * 8u + fm) * 8u + fn + 1u] =
                        dots[b][u].thread_elements()[1];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- softmax + PV, one TILE-key half at a time, in the shipping order --
        //
        // `hbase < range_end` is a THREADGROUP-UNIFORM predicate (`base`, `u` and
        // `range_end` are the same in all 256 threads), so guarding on it rather
        // than `break`-ing keeps `u` a compile-time constant under `#pragma
        // unroll` -- which is what lets the half-separating barrier below
        // disappear at MSUB == 1. The guards inside are uniform too.
#pragma unroll
        for (uint u = 0; u < uint(MSUB); ++u) {
        const uint hbase = base + u * TILE;
        if (hbase < range_end) {
        const uint live_in_tile = min(TILE, range_end - hbase);
        // Key sub-tile that straddles `range_end`, or NSUB when none does.
        const uint partial_kt =
            (live_in_tile % 8u != 0u) ? live_in_tile / 8u : NSUB;

        // ---- online softmax: simdgroup `sg` owns head `sg`, 2 keys per lane,
        //      independently for each of the BQ blocked queries ----
        const uint kt0 = 2u * lane;
        const uint k0 = hbase + kt0;
        const uint k1 = k0 + 1u;
        const bool in0 = (k0 < range_end);
        const bool in1 = (k1 < range_end);
        const uint p0 = in0 ? key_positions[k0] : 0u;
        const uint p1 = in1 ? key_positions[k1] : 0u;
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            if (q0 + b >= params.queries) {
                continue;
            }
            const uint q_abs = query_positions[q0 + b];
            const bool live0 = in0 && (p0 <= q_abs)
                && (params.window == 0u || (q_abs - p0) < params.window);
            const bool live1 = in1 && (p1 <= q_abs)
                && (params.window == 0u || (q_abs - p1) < params.window);
            const float sc0 =
                live0 ? sth[(b * TILEQ + u * TILE + kt0) * 8u + sg] * params.scale : NEG;
            const float sc1 =
                live1 ? sth[(b * TILEQ + u * TILE + kt0 + 1u) * 8u + sg] * params.scale : NEG;
            // Same shape as the scalar body: fold the running max in before the
            // cross-lane reduction, and `NEG` (not -inf) so a fully dead tile
            // keeps `m_old - m` finite.
            const float m_new = simd_max(max(max(sc0, sc1), m[b]));
            const float ms = exp(m[b] - m_new);
            const float w0 = live0 ? exp(sc0 - m_new) : 0.0f;
            const float w1 = live1 ? exp(sc1 - m_new) : 0.0f;
            s[b] = s[b] * ms + simd_sum(w0 + w1);
            m[b] = m_new;
            wq[(b * uint(NSG) + sg) * TILE + kt0] = half(w0);
            wq[(b * uint(NSG) + sg) * TILE + kt0 + 1u] = half(w1);
            if (lane == 0u) {
                msb[b * uint(NSG) + sg] = ms;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- PV straddle staging (V rows past `range_end` zeroed) ----
        if (partial_kt < NSUB) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const uint v0 = hbase + partial_kt * 8u;
            for (uint i = tid; i < 8u * uint(HEAD_DIM); i += 256u) {
                const uint row = i / uint(HEAD_DIM);
                const uint col = i % uint(HEAD_DIM);
                tail[i] = (v0 + row < range_end)
                    ? values[(v0 + row) * kv_stride + head_base + col]
                    : half(0.0f);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // ---- PV: acc[q][head][dim] = ms * acc + P[head][key] * V[key][dim].
        // The V element `bf` is loaded once and reused by every blocked query,
        // which halves the V device traffic.
        //
        // The rescale is skipped for any (query, head) whose max did not move in
        // this tile: `ms = exp(m_old - m_new)` is then exactly 1, and
        // `x * 1 == x` in IEEE-754, so the guard is bit-exact rather than an
        // approximation. Ported from oMLX v0.7.0
        // (`omlx/patches/mimo_v2/sdpa_flash.py:299-300`, *"A row whose max did not
        // move has factor 1: skipping its rescale is exact"*).
        //
        // `fm` is the fragment row, i.e. the head, and the softmax wrote `ms` for
        // exactly that head -- so this is one branch per (query, head) per tile,
        // guarding `NPER * 2` multiplies. On the first tile of a dispatch every
        // head moves (m starts at NEG), so the branch costs nothing there; on later
        // tiles only the heads that found a new maximum take it. With `BQ = 1` the
        // loop is one iteration, so this is a plain `if` on a threadgroup value
        // all lanes of a simdgroup already computed.
#pragma unroll
        for (uint b = 0; b < uint(BQ); ++b) {
            const float msv = msb[b * uint(NSG) + fm];
            if (msv == 1.0f) {
                continue;
            }
#pragma unroll
            for (uint j = 0; j < NPER; ++j) {
                acc[b][j].thread_elements()[0] *= msv;
                acc[b][j].thread_elements()[1] *= msv;
            }
        }
        // Loop order matters more than anything else in this kernel: a
        // `simdgroup_load` is a real instruction, so the MMAs per load is what
        // caps throughput. Preloading all `BQ` P fragments once per k-step and
        // reusing them across the `NPER` n-tiles costs `BQ + NPER` loads for
        // `BQ * NPER` MMAs -- 9/8 at BQ=1 and 10/16 at BQ=2, versus the
        // `BQ * NPER` P loads a per-(n-tile) load would cost. The same argument
        // is why the Q fragment in the QK loop is loaded once per `d8` for all
        // `MSUB` owned m-tiles rather than once per m-tile. (Measured cost of
        // getting this wrong: the T4 bench scored the flash body at ~0.5
        // MMA/load, 42% below the MMA arm.)
        const uint ksteps = (live_in_tile + 7u) / 8u;
        for (uint kt = 0; kt < ksteps; ++kt) {
            simdgroup_half8x8 af[BQ];
#pragma unroll
            for (uint b = 0; b < uint(BQ); ++b) {
                // ABL bit 8: the P fragment load.
                if ((ABL & 8u) != 0u) {
                    af[b] = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
                } else {
                    simdgroup_load(af[b], wq + (b * uint(NSG)) * TILE + kt * 8u, TILE);
                }
            }
            const bool staged = (kt == partial_kt);
#pragma unroll
            for (uint j = 0; j < NPER; ++j) {
                const uint nt = sg * uint(NPER) + j; // KAPLUMBA_NT_BLOCK
                // ABL bit 8: the V fragment load.
                simdgroup_half8x8 bf;
                if ((ABL & 8u) != 0u) {
                    bf = make_filled_simdgroup_matrix<half, 8, 8>(half(0.5f));
                } else if (staged) {
                    simdgroup_load(bf, tail + nt * 8u, HEAD_DIM);
                } else {
                    simdgroup_load(bf, values + (hbase + kt * 8u) * kv_stride
                                       + head_base + nt * 8u, kv_stride);
                }
#pragma unroll
                for (uint b = 0; b < uint(BQ); ++b) {
                    if (q0 + b >= params.queries) {
                        continue;
                    }
                    // ABL bit 4: the PV MMA itself.
                    if ((ABL & 4u) != 0u) {
                        simdgroup_multiply_accumulate(acc[b][j], af[b], bf, acc[b][j]);
                    }
                }
            }
        }
        // `wq` and `msb` are rewritten at the SAME addresses by the next half's
        // softmax, which this half's PV has just read, so the halves need a
        // barrier between them. Shipping needs none here: its next-iteration
        // barrier after the QK publish is what separates one tile's PV from the
        // next tile's softmax. `u` and `MSUB` are compile-time, so this
        // disappears entirely at MSUB == 1.
        if (u + 1u < uint(MSUB)) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        }
        }
    }