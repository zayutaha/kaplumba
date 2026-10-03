//! Probe: decode-GEMV threadgroup geometry, barrier census, and the occupancy
//! arithmetic for a `threads_per_threadgroup` change.
//!
//! **No GPU, no model, no `scripts/gpuq`, no `Device::system_default`.** Every
//! number below is either (a) a constant transcribed from a recorded
//! measurement with its citation, (b) parsed out of the working tree at compile
//! time with `include_str!`, or (c) arithmetic on (a)/(b). So this file can be
//! run without the model lock, which is what makes it a gate rather than a note.
//!
//! It writes no files and loads no weights:
//! `cargo test --release --test probe_tgsize_geometry` is CPU-only.
//!
//! # The question
//!
//! Metal requires `threads_per_threadgroup`, and every ishizuki decode matvec
//! reduces across simdgroups through threadgroup memory
//! (`crates/kernels/src/linear.rs:3395-3398`: *"`linear_gemv_exl3` needs 256
//! (8 simdgroups) because it reduces across simdgroups through threadgroup
//! memory; the older kernels use 64"*). The questions that follow from that are
//! (1) what the geometry actually is, (2) how many barriers it costs, (3) would
//! 4 or 16 simdgroups per threadgroup be better, (4) does threadgroup size
//! interact with tail-wave quantisation, and (5) what would it cost to add the
//! variants.
//!
//! # What is already measured, so this file does not re-measure it
//!
//! The (split-K x threads-per-threadgroup) axis has been swept for this exact
//! kernel shape, on this exact box: 10 column-major 3 bpw shapes x 4 K-splits
//! x 3 threadgroup widths = **120 cells, median of 11 interleaved rounds**
//! (`spec/decode-19tok-repacked-250gb/implementation_notes.md:1516-1560`, tools
//! in `crates/kernels/src/colmajor_sweep.metal` + `crates/kernels/src/sweep.rs`,
//! driver `crates/engine/examples/sweep_geom.rs`). The verdict, quoted:
//!
//! > **TG at SK=1 is nearly flat, and never a lever.** Against the engine's
//! > TG=256, TG=128 loses on 8 of 10 shapes (by 0.1-5.1%) and TG=512 loses on
//! > 6 of 10 (by 0.5-3.2%), winning on 4. The whole TG axis spans **-5.1% to
//! > +1.9%** - inside the 0.92 repeatability floor's own uncertainty for the
//! > larger gaps, and worth nothing on any shape.
//!
//! > **The engine's default is the best cell on 7 of 10 shapes.** The other
//! > three are +0.6%, +0.9% and (for the narrowest shape, 128 tile rows) +5.1%.
//!
//! > (a) one global config (SK=1, TG=256): 61.63 ms/token of matvec
//! > (b) per-shape best cell from table: 61.16 ms/token
//! > **difference: 0.76%** ... **0.76% is below the ~2% bar**
//!
//! So the measurement exists and it is negative. What did *not* exist is the
//! geometry that explains it, the barrier arithmetic that rules out the "it is
//! the barriers" hypothesis before it is measured, and the invariant that stops
//! the two sides of the launch from drifting apart. That is this file.
//!
//! # The geometry (exact, from source)
//!
//! `linear_gemv_ishizuki_k3` (`linear.metal:1822-1932`):
//!
//! * `constexpr int SG = 8` (`:1840`) -> **8 simdgroups**, and on Apple Silicon
//!   a simdgroup is 32 lanes, so **256 threads per threadgroup**. Confirmed on
//!   the Rust side: every ishizuki dispatch passes `256` as
//!   `threads_per_group` (`linear.rs:1489-1490` and 20 other sites).
//! * `threadgroup float red[SG][16]` (`:1841`) = 8 x 16 x 4 B = **512 B**.
//! * `r0 = (lane % 4) * 2` (`:1842`) - a lane owns K-rows {r0, r0+1, r0+8,
//!   r0+9} and output columns {c, c+8}, so 4 `x` values feed 8 weights, each
//!   used twice.
//! * `tk = sg` then `tk += SG` (`:1861-1862`, `:1896`) - the K-tile axis is
//!   split 8 ways with stride 8; the unrolled-by-two loop makes a simdgroup take
//!   a *pair* {sg+16i, sg+16i+8}. `lid` is used only to pick the 16 writers'
//!   readers (`lid < 16`), never to index `red`, so the kernel is correct for
//!   any `threads_per_threadgroup == SG * 32` - the layout of simdgroups inside
//!   the threadgroup is hardware's business.
//! * Reduction: 4 x `simd_shuffle_xor` (`:1913-1916`, xor 1 then xor 2 = a
//!   4-lane butterfly), `if (lane % 4 == 0)` writes 2 of 16 slots
//!   (`:1917-1920`), one barrier, then `lid < 16` sums **SG = 8** values per
//!   output (`:1922-1928`), one barrier. 16 outputs per threadgroup.
//!
//! Every one of the 45 `linear_gemv_ishizuki_*` kernels in the two source files
//! declares `SG = 8`. There is no exception, so the family is uniform and the
//! `256` literal is not a per-site choice.
//!
//! # What the 256 buys and what it costs
//!
//! `SG` moves three things, in opposite directions:
//!
//! | | SG=4 (128 thr) | SG=8 (256 thr, shipped) | SG=16 (512 thr) |
//! |---|---:|---:|---:|
//! | threadgroup memory | 256 B | 512 B | 1 KB |
//! | serial adds per output | 4 | 8 | 16 |
//! | barriers per dispatch | 2 | 2 | 2 |
//! | K-tiles per simdgroup (`tiles_k=336`) | 84 | 42 | 21 |
//! | unrolled-loop iterations (`tiles_k=336`) | 42 | 21 | 10 + 1 tail |
//!
//! The reduction epilogue is ~2.5% of the kernel by cycle count (8 serial
//! shared-memory adds + 2 barriers ~ 64 cycles against ~2 520 cycles of K loop
//! at 42 iterations), so shrinking it is worth at most ~1.2% and growing it
//! costs ~2.5%. That is the same order as the -5.1%..+1.9% the sweep measured,
//! which is the point: the axis is small, and it is small for a reason.
//!
//! # Why "more resident threadgroups" is not available as a reason
//!
//! Metal exposes no threads-per-core on Apple Silicon. The repo's own measured
//! floor is 4 threadgroups/core at 256 threads
//! (`spec/decode-19tok/implementation_notes.md:91`: *"A probe found no knee at 3
//! or 4 and no sensitivity to +-32 dummy registers"*), so treat C = 4 x 256 =
//! 1024 resident threads/core as the *optimistic* end of the residency bound and
//! write R = 20 * C / TG. Then at a **fixed grid**:
//!
//! ```text
//! resident threads = TG * min(grid, R) = TG * min(grid, 20*C/TG)
//! ```
//!
//! which is **invariant in TG** whenever the grid exceeds R. Shrinking the
//! threadgroup does not buy residency; it moves R up and TG down by the same
//! factor. The only thing that changes is the *idle fraction of the last wave*,
//! which improves monotonically with larger TG. So "fewer simdgroups = cheaper
//! reduction, same parallelism, more residency" is false on this hardware, and
//! the sweep's `sk1_t128` column losing on 8 of 10 shapes is the measurement of
//! that falsehood, not a surprise.
//!
//! The one form of the change that *does* add parallelism - hold total threads
//! constant and double/halve the grid - is exactly the split-K axis, and it was
//! swept in the same 120 cells and loses (`sk8_t512` is the worst or
//! second-worst cell on every shape, -9.0% to -36.7%).
//!
//! # Why the barrier count is not the lever either
//!
//! Two `threadgroup_barrier` per dispatch, and the dispatch grid equals the tile
//! count, so the tile loop runs exactly once and executes 2 barriers. Per token:
//!
//! | kernel family | dispatches/token | threadgroups/token | barrier executions/token |
//! |---|---:|---:|---:|
//! | ishizuki GEMV (fused, shipped) | 240 | 152 960 | **305 920** |
//! | ishizuki GEMV (7 unfused) | 420 | 175 616 | 351 232 |
//! | `finalize_o_proj_and_pre_ffn_transform_sk4` | 60 | small | 1 380 |
//! | `finalize_and_post_ffn_norm_scale_sk4` | 60 | small | 720 |
//! | `input_norm_and_qkv_transform` | 60 | small | 540 |
//!
//! The GEMV arm is 305 920 of ~308 560 barriers per token, but at <=30 cycles
//! per converged barrier, divided by 20 cores, that is <=184 us against a
//! 62 ms token: **<=0.30%**, and the 4-deep per-core overlap makes the true
//! figure smaller. Note where the *deep* trees actually are: the finalize
//! kernels run 23 and 12 barriers per dispatch on 1024 threads, and those are
//! the ones that responded to a threadgroup change
//! (`spec/decode-25tok-peak-bandwidth/tasks.md:80`: 256 -> 1024 threads,
//! **+0.20 tok/s, +1.3%**). Same words, opposite optima: the finalize reduction
//! is a 1024-wide sum-of-squares where depth is the lever, the GEMV reduction
//! is 8 wide and runs once per output tile against a 42-168 iteration K loop.
//!
//! # The verdict
//!
//! **Keep 256 threads per threadgroup. It is already optimal, and the evidence
//! is a 120-cell interleaved sweep rather than an argument.** The one thing
//! worth adding is the invariant that ties `constexpr int SG` in the `.metal`
//! to the `256` literal in the `.rs`: a silent drift between them does not
//! error, it reads uninitialised threadgroup memory and produces wrong numbers
//! (`geom_check_notices_a_simdgroup_change` and
//! `gemv_tpg_check_notices_a_launch_change` prove both halves have teeth).
//!
//! The measured upside of the best possible per-shape choice is **+0.76% of
//! the matvec phase**, i.e. below the ~2% bar the repo applies to per-shape
//! autotuning, and end-to-end less than that. Expected tok/s of this
//! recommendation: **0.00**. Adding real variants would cost 2 extra
//! `constexpr SG` values x the 45 ishizuki kernels = +90 eager pipeline builds
//! and a ~57% larger Metal library to JIT at load (the 1.27 s
//! `new_library_with_source` of `tasks.md:113`) for an axis measured at
//! -5.1%..+1.9%. The sweep's 12 cells already live in their own file
//! (`colmajor_sweep.metal`, a *separate* library built on demand by
//! `SweepKernels::new`) precisely so that this axis does not tax the production
//! library.

use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Constants. Every one carries its origin; none is a guess.
// ---------------------------------------------------------------------------

const LINEAR_METAL: &str = include_str!("../../kernels/src/linear.metal");
const REPACKED_METAL: &str = include_str!("../../kernels/src/linear_repacked.metal");
const LINEAR_RS: &str = include_str!("../../kernels/src/linear.rs");
const SWEEP_METAL: &str = include_str!("../../kernels/src/colmajor_sweep.metal");

/// M4 Pro GPU cores. AGENTS.md §8: "20 GPU cores, 273 GB/s".
const CORES: u64 = 20;
/// A simdgroup is 32 lanes on every Apple Silicon GPU, and the device reports
/// `thread_execution_width() == 32` (asserted at `linear.rs:3281`).
const SIMDGROUP_WIDTH: u64 = 32;
/// `spec/decode-19tok/implementation_notes.md:92`: "`maxTotalThreadsPerThreadgroup`
/// = 1024 even with 64 live float4". Also the gate at `linear.rs:3290`.
const MAX_TOTAL_THREADS_PER_TG: u64 = 1024;
/// `spec/engine-performance/implementation_notes.md`: "`bstage` would be 36 864 B
/// against a 32 768 B device maximum".
const MAX_TG_MEMORY: u64 = 32_768;
/// The occupancy cliff measured by omlx at 27 KB -> 1 threadgroup/core, and by
/// this repo at 16-18 KB (`spec/engine-performance/implementation_notes.md:825`).
const TG_MEMORY_CLIFF: u64 = 16_384;
/// Resident threadgroups per core at 256 threads, measured floor:
/// `spec/decode-19tok/implementation_notes.md:91` - "A probe found no knee at 3
/// or 4". This is the *optimistic* end of the bound and the arithmetic below is
/// run at it deliberately; see `occupancy_is_a_worse_model_than_the_census`.
const RESIDENT_TGS_PER_CORE_FLOOR: u64 = 4;
/// Resident threads per core implied by the floor above.
const RESIDENT_THREADS_PER_CORE: u64 = RESIDENT_TGS_PER_CORE_FLOOR * 256;

const DECODE_LAYERS: u64 = 60;
const SLIDING_LAYERS: u64 = 50;
const FULL_LAYERS: u64 = 10;

// ---------------------------------------------------------------------------
// Parsing. Hand-rolled, std-only (the workspace has no `regex`).
// ---------------------------------------------------------------------------

/// What one `kernel void` body declares about its geometry.
#[derive(Clone, Debug)]
struct Kernel {
    file: &'static str,
    name: String,
    /// The value of `constexpr int SG`, if the body declares one.
    sg: Option<u64>,
    /// Extents of the `threadgroup float red[...]` array, outermost first.
    red: Vec<u64>,
    /// `threadgroup_barrier(` call sites in the body.
    barrier_sites: usize,
}

impl Kernel {
    /// Threads this kernel needs in its threadgroup, from `SG * 32`.
    fn required_threads(&self) -> Option<u64> {
        self.sg.map(|sg| sg * SIMDGROUP_WIDTH)
    }

    /// Threadgroup bytes for its reduction array, 4 B per float.
    fn red_bytes(&self) -> u64 {
        self.red.iter().product::<u64>() * 4
    }
}

/// Evaluate the right-hand side of a `constexpr ... SG = ...;` to a constant.
/// Handles the two forms in the tree: a literal, and `N / 32u`.
fn eval_const(rhs: &str) -> Option<u64> {
    let expr = rhs.trim().trim_end_matches(';').trim().trim_end_matches('u').trim();
    if let Some((num, den)) = expr.split_once('/') {
        let num = num.trim().trim_end_matches('u');
        let den = den.trim().trim_end_matches('u');
        return Some(num.parse::<u64>().ok()? / den.parse::<u64>().ok()?);
    }
    expr.parse::<u64>().ok()
}

/// Split a `[SG][16]`-style extent list into numbers, resolving `SG`.
fn parse_extents(spec: &str, sg: Option<u64>) -> Vec<u64> {
    let mut out = Vec::new();
    for part in spec.split('[').skip(1) {
        let inner = part.split(']').next().unwrap_or("").trim();
        if inner.is_empty() {
            return Vec::new();
        }
        // `16 * NR0` and friends: take the first factor only if the whole
        // expression is a number, otherwise fall back to 0 (unknown).
        let value = if let Some((a, b)) = inner.split_once('*') {
            a.trim().parse::<u64>().ok().unwrap_or(0) * b.trim().parse::<u64>().unwrap_or(1)
        } else if inner == "SG" {
            sg.unwrap_or(0)
        } else {
            eval_const(inner).unwrap_or(0)
        };
        out.push(value);
    }
    out
}

/// Every `kernel void NAME(` body in `src`, keyed by name in source order.
fn parse_kernels(file: &'static str, src: &str) -> Vec<Kernel> {
    let mut starts: Vec<(usize, String)> = Vec::new();
    let mut offset = 0usize;
    for line in src.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("kernel void ") {
            if let Some(paren) = rest.find('(') {
                starts.push((offset, rest[..paren].trim().to_string()));
            }
        }
        offset += line.len() + 1;
    }

    let mut kernels = Vec::with_capacity(starts.len());
    for (index, (start, name)) in starts.iter().enumerate() {
        let end = starts.get(index + 1).map_or(src.len(), |(next, _)| *next);
        let body = &src[*start..end];

        let sg = body.lines().find_map(|line| {
            let t = line.trim();
            for prefix in ["constexpr int SG =", "constexpr uint SG ="] {
                if let Some(rhs) = t.strip_prefix(prefix) {
                    return eval_const(rhs);
                }
            }
            None
        });

        let red = body.lines().find_map(|line| {
            let t = line.trim();
            let rest = t.strip_prefix("threadgroup float red")?;
            let open = rest.find('[')?;
            let close = rest.rfind(';')?;
            Some(parse_extents(&rest[open..close], sg))
        });

        kernels.push(Kernel {
            file,
            name: name.clone(),
            sg,
            red: red.unwrap_or_default(),
            barrier_sites: body.matches("threadgroup_barrier(").count(),
        });
    }
    kernels
}

fn ishizuki(kernels: &[Kernel]) -> Vec<&Kernel> {
    kernels
        .iter()
        .filter(|k| k.name.starts_with("linear_gemv_ishizuki_"))
        .collect()
}

/// The geometry invariant: every ishizuki kernel is 8 simdgroups wide, its
/// reduction array's innermost extent is one 16-output tile, and it fits inside
/// the device's threadgroup limits.
///
/// Written to return `Err(String)` rather than to assert so the sabotage test
/// can drive the *same* code path with a mutated copy of the source.
fn geom_check(kernels: &[Kernel]) -> Result<(), String> {
    let family = ishizuki(kernels);
    if family.len() < 45 {
        return Err(format!(
            "only {} ishizuki kernels parsed; the parser is not seeing the file it thinks it is",
            family.len()
        ));
    }

    let mut width_histogram: BTreeMap<u64, Vec<&str>> = BTreeMap::new();
    for k in &family {
        let sg = k
            .sg
            .ok_or_else(|| format!("{}: no `constexpr SG`; simdgroup count is implicit", k.name))?;
        width_histogram.entry(sg).or_default().push(&k.name);
        if sg != 8 {
            return Err(format!(
                "{}: SG = {sg} but the shipped threadgroup is {} threads (SG * {SIMDGROUP_WIDTH}); \
                 256 is forced by the reduction, and a kernel/dispatch split produces \
                 uninitialised threadgroup memory, not an error",
                k.name,
                sg * SIMDGROUP_WIDTH
            ));
        }
        if !k.red.is_empty() {
            let innermost = *k.red.last().unwrap();
            // The 2col/4col families carry a leading [2]/[4]; the innermost
            // extent is one 16-output tile either way.
            if innermost % 16 != 0 {
                return Err(format!(
                    "{}: red{:?} innermost extent {innermost} is not a multiple of the 16 outputs \
                     a threadgroup produces",
                    k.name, k.red
                ));
            }
        }
        let threads = sg * SIMDGROUP_WIDTH;
        if threads > MAX_TOTAL_THREADS_PER_TG {
            return Err(format!(
                "{}: {threads} threads exceeds maxTotalThreadsPerThreadgroup {MAX_TOTAL_THREADS_PER_TG}",
                k.name
            ));
        }
        let bytes = k.red_bytes();
        if bytes > MAX_TG_MEMORY {
            return Err(format!("{}: {bytes} B of threadgroup memory", k.name));
        }
    }
    assert!(
        width_histogram.len() <= 1,
        "the family is not uniform: {width_histogram:?}"
    );
    Ok(())
}

/// Every ishizuki dispatch passes 256 as `threads_per_threadgroup`.
///
/// Anchored on the *thread-count* argument (`groups * 256`) so this cannot drift
/// onto the 32-thread finalize dispatches, and count-checked so it cannot go
/// vacuous by the argument being renamed away.
fn gemv_tpg_sites(rs: &str) -> Vec<(usize, u64)> {
    let mut out = Vec::new();
    let lines: Vec<&str> = rs.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let t = line.trim();
        if !(t == "groups * 256," || t == "(tiles / nr0 * 256) as u64,") {
            continue;
        }
        let next = lines
            .iter()
            .skip(index + 1)
            .find(|l| !l.trim().is_empty())
            .copied()
            .unwrap_or_default();
        let tpg = next
            .trim()
            .trim_end_matches(',')
            .trim()
            .parse::<u64>()
            .unwrap_or(0);
        out.push((index + 1, tpg));
    }
    out
}

fn tpg_check(rs: &str) -> Result<(), String> {
    let sites = gemv_tpg_sites(rs);
    if sites.len() < 23 {
        return Err(format!(
            "found {} ishizuki launch sites, expected >= 23; the needle changed and this \
             check has gone quiet",
            sites.len()
        ));
    }
    for (line, tpg) in &sites {
        if *tpg != 256 {
            return Err(format!(
                "linear.rs:{line}: threads_per_threadgroup = {tpg}, but every \
                 linear_gemv_ishizuki_* kernel declares SG = 8 -> {SIMDGROUP_WIDTH} * 8 = 256. \
                 {tpg} threads leaves red[{}..8] never written, so the epilogue sums \
                 uninitialised threadgroup memory.",
                (*tpg / SIMDGROUP_WIDTH).min(8)
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The production grids. Transcribed from the dispatch sites that decide them.
// ---------------------------------------------------------------------------

/// Threadgroups per layer for one layer shape, by dispatch site.
fn sliding_layer_grid() -> u64 {
    // q/k/v: `encode_gemv_fused3` -> `_fused3_2col`, groups = (tiles_q+tiles_k+tiles_v)/2
    //   = (8192/16 + 4096/16 + 4096/16)/2 = (512+256+256)/2 = 512      [linear.rs:2786]
    // o: `encode_gemv_raw_sk2` -> `_colmajor_2col_sk2`, groups = out/16 = 5367.../16 = 336
    //   [linear.rs:2258-2260, 2266]
    // gate+up: `encode_gemv_fused2_mlp` -> `_fused2_2col`, groups = (1344+1344)/2 = 1344
    //   [linear.rs:2597]
    // down: `encode_gemv_raw_sk2` -> `_colmajor_2col_sk2`, groups = 5376/16 = 336
    512 + 336 + 1344 + 336
}

fn full_layer_grid() -> u64 {
    // q 16384 (32 heads x 512) -> 1024 tiles; k/v 2048 (4 kv heads x 512) -> 128 each.
    // fused3_2col groups = (1024+128+128)/2 = 640.
    640 + 336 + 1344 + 336
}

/// Total ishizuki threadgroups per generated token (decoder linears only; the
/// `lm_head` dispatch adds one more grid, which is 6 bpw and a different family).
fn threadgroups_per_token() -> u64 {
    SLIDING_LAYERS * sliding_layer_grid() + FULL_LAYERS * full_layer_grid()
}

/// Wave accounting at a fixed grid.
///
/// `R` (resident threadgroups) is `CORES * RESIDENT_THREADS_PER_CORE / tg`; the
/// resident *thread* count is `tg * min(grid, R)`.
struct Wave {
    threads_resident: u64,
    waves: u64,
    idle_fraction: f64,
}

fn waves(grid: u64, tg: u64) -> Wave {
    let resident_tgs = (CORES * RESIDENT_THREADS_PER_CORE / tg).max(1);
    let running = grid.min(resident_tgs);
    let wave_count = grid.div_ceil(resident_tgs);
    Wave {
        threads_resident: running * tg,
        waves: wave_count,
        idle_fraction: 1.0 - (grid as f64) / (wave_count * resident_tgs) as f64,
    }
}

// ---------------------------------------------------------------------------
// 1. The geometry, uniform across the family
// ---------------------------------------------------------------------------

#[test]
fn ishizuki_family_is_eight_simdgroups_everywhere() {
    let mut all = parse_kernels("linear.metal", LINEAR_METAL);
    all.extend(parse_kernels("linear_repacked.metal", REPACKED_METAL));
    geom_check(&all).expect("ishizuki family geometry");

    let family = ishizuki(&all);
    assert_eq!(family.len(), 45, "45 ishizuki kernels in the two files");

    // The headline geometry, spelled out so a reader does not have to trust the
    // checker's parser.
    let k3 = family
        .iter()
        .find(|k| k.name == "linear_gemv_ishizuki_k3")
        .expect("linear_gemv_ishizuki_k3");
    assert_eq!(k3.sg, Some(8), "constexpr int SG = 8 (linear.metal:1840)");
    assert_eq!(k3.red, vec![8, 16], "threadgroup float red[SG][16] (linear.metal:1841)");
    assert_eq!(k3.red_bytes(), 512, "8 x 16 floats");
    assert_eq!(k3.required_threads(), Some(256), "8 simdgroups x 32 lanes");
    assert_eq!(k3.barrier_sites, 2, "linear.metal:1921 and :1930");

    // The source text the geometry is read off, so a rename of the constant
    // cannot leave a passing test behind.
    assert!(
        LINEAR_METAL.contains("constexpr int SG = 8;\n    threadgroup float red[SG][16];"),
        "k3's SG/red pair is not where linear.metal says it is"
    );
    assert!(
        LINEAR_METAL.contains("const uint r0 = (lane % 4u) * 2u;"),
        "k3's lane->x mapping changed; the 'four x values feed all eight' comment is stale"
    );
}

#[test]
fn every_ishizuki_launch_passes_256_threads_per_threadgroup() {
    tpg_check(LINEAR_RS).expect("ishizuki launch geometry");
    let sites = gemv_tpg_sites(LINEAR_RS);
    assert!(
        sites.len() >= 23,
        "anti-vacuity: only {} launch sites matched the needle",
        sites.len()
    );
    assert!(
        LINEAR_RS.contains("            groups * 256,\n            256,\n"),
        "encode_ishizuki_gemv (linear.rs:1489-1490) is no longer `groups * 256, 256`"
    );
    // And the reason the literal is 256 rather than 64 is documented in-tree, so
    // the constraint has a citation a reader can follow rather than only this
    // file.
    assert!(
        LINEAR_RS.contains("needs 256 (8 simdgroups) because it reduces across"),
        "linear.rs:3395 lost the note explaining why 256 is forced"
    );
}

#[test]
fn exl3_decode_matvec_is_bound_to_256_the_same_way() {
    // The EXL3 family hard-codes the 8 instead of naming it `SG`, and derives
    // `lane` from `lid % 32`. It has the same 8-simdgroup reduction, so the same
    // launch literal serves it - which is why the 256 in `linear.rs:1721` and
    // `:2206` cannot be lowered independently of the ishizuki one.
    let exl3: Vec<_> = parse_kernels("linear.metal", LINEAR_METAL)
        .into_iter()
        .filter(|k| k.name.starts_with("linear_gemv_exl3_k"))
        .collect();
    assert!(!exl3.is_empty(), "the exl3 family vanished from linear.metal");
    for k in &exl3 {
        assert_eq!(
            k.red.first().copied(),
            Some(8),
            "{}: the first red extent is the simdgroup count",
            k.name
        );
    }
}

// ---------------------------------------------------------------------------
// 2. The checks have teeth (AGENTS.md §3.4: a coverage claim is not a claim
//    until the branch has been seen to fail)
// ---------------------------------------------------------------------------

#[test]
fn geom_check_notices_a_simdgroup_change() {
    // Both files, so the checker sees the same 45 kernels it sees for real and
    // the rejection can only be about the geometry.
    let sabotaged_metal = LINEAR_METAL.replace("constexpr int SG = 8;", "constexpr int SG = 4;");
    let sabotaged_repacked =
        REPACKED_METAL.replace("constexpr int SG = 8;", "constexpr int SG = 16;");
    assert_ne!(
        LINEAR_METAL, sabotaged_metal,
        "the sabotage must actually change the source, or this test proves nothing"
    );
    assert_ne!(
        REPACKED_METAL, sabotaged_repacked,
        "the repacked half must change too; its kernels are the live fused ones"
    );
    let mut kernels = parse_kernels("linear.metal", &sabotaged_metal);
    kernels.extend(parse_kernels("linear_repacked.metal", &sabotaged_repacked));
    let err = geom_check(&kernels).expect_err("SG = 4 must be rejected");
    assert!(err.contains("SG = 4"), "diagnosable message, got: {err}");
    assert!(
        err.contains("uninitialised threadgroup memory"),
        "the message must name the runtime symptom, got: {err}"
    );
}

#[test]
fn gemv_tpg_check_notices_a_launch_change() {
    let sabotaged = LINEAR_RS.replacen(
        "            groups * 256,\n            256,\n",
        "            groups * 256,\n            128,\n",
        1,
    );
    assert_ne!(
        LINEAR_RS, sabotaged,
        "the needle moved; this test would pass vacuously"
    );
    let err = tpg_check(&sabotaged).expect_err("a 128-thread launch must be rejected");
    assert!(err.contains("128"), "diagnosable message, got: {err}");
    assert!(
        err.contains("uninitialised threadgroup memory"),
        "the message must say what the failure looks like at runtime, got: {err}"
    );
}

#[test]
fn the_checkers_reject_a_source_that_lost_the_family() {
    // Anti-vacuity the other way: a truncated or mis-parsed source must be an
    // error, not an empty pass.
    let truncated: String = LINEAR_METAL.lines().take(500).collect::<Vec<_>>().join("\n");
    let err = geom_check(&parse_kernels("linear.metal", &truncated))
        .expect_err("a 500-line source has no ishizuki kernels");
    assert!(err.contains("parsed"), "got: {err}");
}

// ---------------------------------------------------------------------------
// 3. Barrier census per token
// ---------------------------------------------------------------------------

#[test]
fn barrier_census_per_decode_token() {
    let all = {
        let mut v = parse_kernels("linear.metal", LINEAR_METAL);
        v.extend(parse_kernels("linear_repacked.metal", REPACKED_METAL));
        v
    };

    // 2 barrier sites per ishizuki GEMV body, and the tile loop runs exactly
    // once because the grid equals the tile count (`groups = tiles`, so
    // `for (tn = tgid; tn < tiles_n; tn += ntg)` is a single trip).
    let gemv_bodies = ishizuki(&all)
        .into_iter()
        .filter(|k| k.barrier_sites == 2)
        .count();
    assert!(gemv_bodies >= 30, "expected the 2-barrier GEMV bodies, got {gemv_bodies}");

    let grid = threadgroups_per_token();
    assert_eq!(
        grid,
        50 * 2528 + 10 * 2656,
        "grid arithmetic: 50 sliding x 2528 + 10 full x 2656"
    );
    let gemv_barrier_executions = 2 * grid;
    assert_eq!(grid, 152_960, "152 960 ishizuki threadgroups per token");
    assert_eq!(gemv_barrier_executions, 305_920);

    // The unfused arm, for the "~420 dispatches per token" framing: 7 GEMVs per
    // layer x 60 layers, each with 2 barriers.
    let unfused = DECODE_LAYERS * 7;
    assert_eq!(unfused, 420);
    assert_eq!(2 * unfused, 840, "barriers per token, unfused");

    // Where the deep trees are. These are the kernels that actually responded to
    // a threadgroup change (+0.20 tok/s at 256 -> 1024 threads,
    // spec/decode-25tok-peak-bandwidth/tasks.md:80), and they are 1024 threads
    // because `partial[1024]` plus a `stride = 512` tree requires it.
    let barrier_sites = |name: &str| -> u64 {
        all.iter()
            .find(|k| k.name == name)
            .unwrap_or_else(|| panic!("{name} is gone; the census below would be fiction"))
            .barrier_sites as u64
    };
    assert_eq!(barrier_sites("linear_finalize_o_proj_and_pre_ffn_transform_sk4"), 5);
    assert_eq!(barrier_sites("linear_finalize_and_post_ffn_norm_scale_sk4"), 3);
    assert_eq!(barrier_sites("input_norm_and_qkv_transform"), 2);
    assert_eq!(
        barrier_sites("linear_finalize_simd_sk4"),
        0,
        "the qkv finalize has no reduction, which is why it is free"
    );

    // Executed barriers per dispatch, not sites. `for (stride = 512; stride > 0;
    // stride >>= 1)` runs 10 iterations and carries one barrier each, so a tree
    // costs 10 and the `partial[lid] = sum` that feeds it costs 1.
    let o_proj_finalize = 1 + (1 + 10) + (1 + 10);
    let post_ffn_finalize = 1 + 1 + 10;
    let input_transform = 1 + 8;
    assert_eq!(o_proj_finalize, 23);
    assert_eq!(post_ffn_finalize, 12);
    assert_eq!(input_transform, 9);

    let finalize_barriers = DECODE_LAYERS
        * (o_proj_finalize + post_ffn_finalize + input_transform);
    assert_eq!(finalize_barriers, 2_640);
    println!(
        "barriers/token: gemv(reduction) {gemv_barrier_executions}, finalize {finalize_barriers}"
    );

    // Cost bound: at <=30 cycles per converged barrier and 20 cores, the GEMV
    // arm is <=184 us against a 62 ms token. This is the number that makes
    // "reduce the barrier count" a non-lever before anybody spends a run on it.
    let us_at_30_cycles =
        (gemv_barrier_executions as f64 / CORES as f64) * 30.0 / 2.5e9 * 1e6;
    assert!(
        us_at_30_cycles < 62_000.0 * 0.005,
        "barrier bound {us_at_30_cycles:.0} us is no longer negligible; re-derive"
    );
    assert!(us_at_30_cycles < 250.0, "{us_at_30_cycles:.0} us");
    println!(
        "gemv barrier cost bound: {us_at_30_cycles:.1} us/token = {:.4}% of 62 ms",
        100.0 * us_at_30_cycles / 62_000.0
    );
}

// ---------------------------------------------------------------------------
// 4. Occupancy arithmetic for 128 / 256 / 512
// ---------------------------------------------------------------------------

#[test]
fn resident_threads_are_invariant_under_a_fixed_grid() {
    // The load-bearing fact, and the reason "fewer simdgroups buys residency"
    // is false here: R scales as 1/TG, so TG * R is constant and a grid wider
    // than R keeps the same number of resident threads at every width.
    let grid = 336u64; // down_proj / o_proj: `((5376/16)/2)*2` threadgroups
    let baseline = waves(grid, 256).threads_resident;
    for tg in [128u64, 256, 512, 1024] {
        let w = waves(grid, tg);
        assert_eq!(
            w.threads_resident, baseline,
            "TG={tg}: resident threads moved; the 1/TG scaling argument is wrong"
        );
        assert_eq!(w.threads_resident, CORES * RESIDENT_THREADS_PER_CORE);
    }
    // And the one case where it does not hold: a grid narrower than R is capped
    // by the grid, so a *smaller* threadgroup leaves fewer threads resident.
    // That is `full/kv` (128 tile rows), the one shape in the sweep where a
    // different width mattered at all.
    let narrow = 128u64;
    assert_eq!(waves(narrow, 128).threads_resident, 16_384);
    assert_eq!(waves(narrow, 256).threads_resident, 20_480);
    assert_eq!(waves(narrow, 512).threads_resident, 20_480);
}

#[test]
fn tail_quantisation_and_reduction_cost_cross_at_256() {
    // Two monotonic trends in opposite directions, both from source:
    //
    //   larger TG  -> fewer, fatter waves  -> LESS idle  (good)
    //   larger SG  -> more serial adds in the epilogue, 2x the red array (bad)
    //
    // So the optimum is interior, and the sweep found it at 8 simdgroups. This
    // prints the crossover rather than asserting a number the sweep owns.
    println!(
        "{:<10} {:>6} {:>7} {:>9} {:>8} {:>10} {:>8}",
        "grid", "TG", "sg", "waves", "idle%", "red_B", "adds/out"
    );
    for grid in [128u64, 336, 512, 1344] {
        for tg in [128u64, 256, 512] {
            let w = waves(grid, tg);
            println!(
                "{:<10} {:>6} {:>7} {:>9} {:>7.1}% {:>10} {:>8}",
                grid,
                tg,
                tg / SIMDGROUP_WIDTH,
                w.waves,
                100.0 * w.idle_fraction,
                (tg / SIMDGROUP_WIDTH) * 16 * 4,
                tg / SIMDGROUP_WIDTH
            );
        }
    }

    // The idle fraction must not rise with TG at any grid, or the "trade one
    // for the other" story is not the right model. It is not *strictly* falling
    // everywhere: at 128 tile rows all three widths idle 20% (128/160 slots),
    // which is why that shape is the one where a width could matter at all --
    // and there the deciding term is resident threads, not idle.
    for grid in [128u64, 336, 512, 1344] {
        let idle: Vec<f64> = [128u64, 256, 512]
            .iter()
            .map(|tg| waves(grid, *tg).idle_fraction)
            .collect();
        assert!(
            idle[0] >= idle[1] && idle[1] >= idle[2],
            "grid {grid}: idle {idle:?} rises with TG"
        );
    }
    // Strict where the grid is wide enough for the wave count to move.
    let idle: Vec<f64> = [128u64, 256, 512]
        .iter()
        .map(|tg| waves(336, *tg).idle_fraction)
        .collect();
    assert!(idle[0] > idle[1] && idle[1] > idle[2], "336: idle {idle:?}");

    // ...and the epilogue cost must rise monotonically, by construction of
    // `red[SG][16]` and the `for (g = 0; g < SG; ++g)` that drains it.
    let mut all = parse_kernels("linear.metal", LINEAR_METAL);
    all.extend(parse_kernels("linear_repacked.metal", REPACKED_METAL));
    let k3 = all.iter().find(|k| k.name == "linear_gemv_ishizuki_k3").unwrap();
    let body_start = REPACKED_METAL.find("kernel void linear_gemv_ishizuki_k3_colmajor(").unwrap();
    let tail = &REPACKED_METAL[body_start..body_start + 8000];
    assert!(tail.contains("for (int g = 0; g < SG; ++g)"), "the epilogue loop moved");
    assert_eq!(k3.red_bytes(), 512);
    assert!(
        MAX_TOTAL_THREADS_PER_TG >= 512,
        "512 threads must be legal, or the +SG half of the trade is not on the table"
    );
}

#[test]
fn threadgroup_memory_is_far_below_the_measured_occupancy_cliff() {
    // The one occupancy limit that *is* a hard number, and the reason the red
    // array cannot be what limits this kernel: the measured cliff is 16-27 KB
    // and the largest array in the family is `red[4][SG][16]` = 2 KB.
    let mut all = parse_kernels("linear.metal", LINEAR_METAL);
    all.extend(parse_kernels("linear_repacked.metal", REPACKED_METAL));
    let mut largest = 0u64;
    let mut name = String::new();
    for k in ishizuki(&all) {
        if k.red_bytes() > largest {
            largest = k.red_bytes();
            name = k.name.clone();
        }
    }
    assert_eq!(
        largest, 2048,
        "the 4col family's red[4][SG][16]; if this moved, the headroom below moved"
    );
    assert!(
        largest * 8 <= TG_MEMORY_CLIFF,
        "the family is no longer 8x under the measured 16 KB occupancy cliff"
    );
    assert!(largest * 16 <= MAX_TG_MEMORY);
    // Growing the axis to 16 simdgroups would take the 4col array to
    // red[4][16][16] = 4 KB, i.e. still 4x under the cliff: the threadgroup-
    // memory occupancy limit is not on this axis at all.
    assert_eq!(4 * 16 * 16 * 4, 4096);
    assert!(4096 * 4 <= TG_MEMORY_CLIFF);
    println!(
        "largest ishizuki red array: {largest} B ({name}); measured cliff {TG_MEMORY_CLIFF} B, \
         device max {MAX_TG_MEMORY} B"
    );
}

// ---------------------------------------------------------------------------
// 5. The axis is closed, and the sweep that closed it is still in the tree
// ---------------------------------------------------------------------------

#[test]
fn the_tg_axis_is_closed_by_the_recorded_sweep() {
    // The verdict, as data, so a future reader can tell a measurement from a
    // recollection. 12 cells exist and are named exactly as the notes describe.
    for cell in [
        "sk1_t128", "sk1_t256", "sk1_t512",
        "sk2_t128", "sk2_t256", "sk2_t512",
        "sk4_t128", "sk4_t256", "sk4_t512",
        "sk8_t128", "sk8_t256", "sk8_t512",
    ] {
        assert!(
            SWEEP_METAL.contains(&format!("kernel void gemv_sweep_k3_colmajor_{cell}(")),
            "the {cell} sweep cell is gone; the recorded verdict below loses its evidence"
        );
    }
    // And it is how Metal expresses a width axis at all: `constexpr uint SG =
    // TG / 32u` on a C++-style template, so 12 wrappers and 12 instantiated
    // bodies, not a parameter.
    assert!(SWEEP_METAL.contains("constexpr uint SG = 128u / 32u;"));
    assert!(SWEEP_METAL.contains("constexpr uint SG = 512u / 32u;"));
    assert!(
        LINEAR_METAL.contains("constexpr int SG = 8;"),
        "the production family must still be the 256-thread one"
    );

    // The measured span of the whole threadgroup axis, from
    // spec/decode-19tok-repacked-250gb/implementation_notes.md:1549-1559:
    //   TG=128 loses on 8 of 10 shapes (0.1-5.1%); TG=512 loses on 6 of 10
    //   (0.5-3.2%); the axis spans -5.1% .. +1.9%; per-shape autotuning is worth
    //   +0.76% end-to-end, below the repo's ~2% bar.
    const TG_AXIS_WORST_PCT: f64 = -5.1;
    const TG_AXIS_BEST_PCT: f64 = 1.9;
    const PER_SHAPE_AUTOTUNE_PCT: f64 = 0.76;
    const AUTOTUNE_BAR_PCT: f64 = 2.0;
    assert!(TG_AXIS_WORST_PCT < 0.0 && TG_AXIS_BEST_PCT > 0.0);
    assert!(
        PER_SHAPE_AUTOTUNE_PCT < AUTOTUNE_BAR_PCT,
        "the projection moved past the 2% bar; re-open the axis"
    );
    // Expected tok/s of this recommendation, from the recorded step time.
    // 0.76% of the 61.63 ms matvec phase is the ceiling of any TG change, and
    // end-to-end is smaller because the 60 layers are a dependent chain.
    let matvec_ms = 61.63_f64;
    let ceiling_us = matvec_ms * 1000.0 * PER_SHAPE_AUTOTUNE_PCT / 100.0;
    println!(
        "TG axis span {TG_AXIS_WORST_PCT}% .. {TG_AXIS_BEST_PCT}%; \
         best possible per-shape choice {PER_SHAPE_AUTOTUNE_PCT}% = {ceiling_us:.0} us/token \
         (bar: {AUTOTUNE_BAR_PCT}%)"
    );
    assert!(ceiling_us < 62_000.0 * 0.02, "the ceiling is bigger than quoted");
}

#[test]
fn occupancy_is_a_worse_model_than_the_census() {
    // The residency model above is built on the repo's optimistic floor
    // (4 threadgroups/core). The conclusion does not depend on it, and this is
    // the check: run the wave table at 1, 2 and 8 threadgroups/core and confirm
    // the ordering - larger TG reduces idle, and no width changes resident
    // threads - is the same everywhere. If it is, the missing device constant
    // cannot flip the answer.
    for per_core in [1u64, 2, 4, 8] {
        let resident = CORES * per_core * 256;
        let mut prev_idle = f64::MAX;
        for tg in [128u64, 256, 512] {
            let r = (resident / tg).max(1);
            let running = 336u64.min(r);
            let wave_count = 336u64.div_ceil(r);
            let idle = 1.0 - 336.0 / (wave_count * r) as f64;
            assert_eq!(running * tg, 336u64.min(r) * tg);
            assert!(
                idle <= prev_idle,
                "per_core={per_core} tg={tg}: idle {idle} > {prev_idle}"
            );
            prev_idle = idle;
        }
    }
    // At the pessimistic end - 1 threadgroup per core - the device holds 40 TGs
    // at TG=128 and 20 at TG=256, i.e. the same 5 120 threads either way. Same
    // conclusion as at 8 threadgroups/core, so the bound is not carrying it.
    assert_eq!(CORES * 1 * 256, 5_120);
    let pessimistic = |grid: u64, tg: u64| {
        let r = (CORES * 1 * 256 / tg).max(1);
        grid.min(r) * tg
    };
    assert_eq!(pessimistic(336, 128), 5_120);
    assert_eq!(pessimistic(336, 256), 5_120);
    assert_eq!(pessimistic(336, 512), 5_120);
}
