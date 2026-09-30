//! Loader and dispatch for the decode dispatch-geometry sweep.
//!
//! Twelve cells: `SK` (K-split) in {1, 2, 4, 8} x `TG` (threads per
//! threadgroup) in {128, 256, 512}, for the 3 bpw column-major decode GEMV.
//! See `src/colmajor_sweep.metal` for why the kernels live in their own
//! library and why every one of them is unreachable from the engine.
//!
//! Two properties this module is responsible for:
//!
//! * **The `(SK = 1, TG = 256)` cell is the engine's live geometry.**
//!   `LinearKernels::encode_ishizuki_gemv` dispatches `min(out_features / 16,
//!   KAPLUMBA_GEMV_MAX_GROUPS)` threadgroups of 256 threads for
//!   `linear_gemv_ishizuki_k3_colmajor_contig4`, and `gemv_max_groups()` is
//!   `u64::MAX` unless the env var is set, i.e. `out_features / 16`. The sweep
//!   cuts its grid the same way, so the baseline cell and the engine issue the
//!   same shape of launch. `tn_groups()` is the one place that decision is
//!   written down, and `baseline()` is the one place it is read.
//! * **The split-K reduction is inside the timed region.** `dispatch` emits the
//!   reduction as a second dispatch, exactly as the engine's `_sk2` / `_sk4`
//!   path does, so a cell cannot look cheap by leaving its partial sums
//!   unreduced.

use std::collections::BTreeMap;

use metal::{Buffer, CompileOptions, ComputePipelineState, Device};

use crate::encoder::{MetalEncoder, TensorRef};
use crate::exl3::KernelError;

const SOURCE: &str = include_str!("colmajor_sweep.metal");

/// K-split values the sweep enumerates. `1` is the engine's geometry.
pub const SK_VALUES: [u32; 4] = [1, 2, 4, 8];
/// Threads-per-threadgroup values. `256` is the engine's geometry.
pub const TG_VALUES: [u32; 3] = [128, 256, 512];

/// One cell of the grid: how many threadgroups split one output tile-row's K
/// reduction, and how many threads each threadgroup has.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cell {
    pub sk: u32,
    pub threads: u32,
}

impl Cell {
    /// The engine's live geometry, and the baseline every gap is measured from.
    pub fn baseline() -> Self {
        Cell {
            sk: 1,
            threads: 256,
        }
    }

    pub fn label(&self) -> String {
        format!("sk{}_t{}", self.sk, self.threads)
    }

    /// The `gemv_sweep_k3_colmajor_*` entry point for this cell.
    pub fn kernel_name(&self) -> String {
        format!("gemv_sweep_k3_colmajor_sk{}_t{}", self.sk, self.threads)
    }
}

/// Every cell, in a fixed order so a harness that interleaves them round-robins
/// over the grid rather than over an accident of hash iteration.
pub fn cells() -> Vec<Cell> {
    let mut out = Vec::with_capacity(SK_VALUES.len() * TG_VALUES.len());
    for sk in SK_VALUES {
        for threads in TG_VALUES {
            out.push(Cell { sk, threads });
        }
    }
    out
}

/// Threadgroups covering the output tile-rows, before the K split multiplies
/// them. `out_features / 16` because `gemv_max_groups()` is `u64::MAX` in any
/// run that did not set `KAPLUMBA_GEMV_MAX_GROUPS`, and every measurement this
/// module exists to produce is a run of the default engine.
pub fn tn_groups(out_features: usize) -> u64 {
    (out_features / 16) as u64
}

/// Buffers one cell needs. `partials` is `SK * out_features` floats; `y` is
/// `out_features` floats and is what the reduction leaves behind.
pub struct SweepRequest<'a> {
    pub packed: &'a Buffer,
    /// Byte offset into `packed`. A harness streaming a working set far larger
    /// than any cache moves this per dispatch; a fixed 0 measures L2, not DRAM.
    pub packed_offset: u64,
    pub x: &'a Buffer,
    pub partials: &'a Buffer,
    pub y: &'a Buffer,
    pub in_features: usize,
    pub out_features: usize,
}

/// The bytes of a `set_bytes` argument, the way `linear.rs::params_bytes` does
/// it. `Encoder::set_bytes` copies synchronously, so the pointer only has to
/// stay valid across the call.
fn params_bytes<T>(params: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(params as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// The sweep library: one pipeline per cell, plus the codebook probe.
pub struct SweepKernels {
    gemv: BTreeMap<String, ComputePipelineState>,
    reduce: BTreeMap<u32, ComputePipelineState>,
    mcg_probe: ComputePipelineState,
}

impl SweepKernels {
    pub fn new() -> Result<Self, KernelError> {
        let device = Device::system_default().ok_or(KernelError::DeviceUnavailable)?;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(SOURCE, &options)
            .map_err(KernelError::Shader)?;
        let mut gemv = BTreeMap::new();
        let mut reduce = BTreeMap::new();
        for cell in cells() {
            let name = cell.kernel_name();
            let function = library
                .get_function(&name, None)
                .map_err(KernelError::Shader)?;
            gemv.insert(
                name.clone(),
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(KernelError::Pipeline)?,
            );
            if cell.sk > 1 {
                let reduce_name = format!("gemv_sweep_reduce_sk{}", cell.sk);
                let reduce_function = library
                    .get_function(&reduce_name, None)
                    .map_err(KernelError::Shader)?;
                reduce.insert(
                    cell.sk,
                    device
                        .new_compute_pipeline_state_with_function(&reduce_function)
                        .map_err(KernelError::Pipeline)?,
                );
            }
        }
        let probe_function = library
            .get_function("gemv_sweep_mcg_probe", None)
            .map_err(KernelError::Shader)?;
        let mcg_probe = device
            .new_compute_pipeline_state_with_function(&probe_function)
            .map_err(KernelError::Pipeline)?;
        Ok(Self {
            gemv,
            reduce,
            mcg_probe,
        })
    }

    /// Encode one cell: the GEMV, plus the split-K reduction when `sk > 1`.
    ///
    /// Both dispatches land in the caller's encoder and neither is committed
    /// here, so a caller can put N iterations of a cell in one command buffer
    /// and time the pair together.
    pub fn dispatch(
        &self,
        encoder: &mut MetalEncoder,
        cell: Cell,
        request: &SweepRequest<'_>,
    ) -> Result<(), KernelError> {
        let pipeline = self
            .gemv
            .get(&cell.kernel_name())
            .ok_or_else(|| KernelError::InvalidInput(format!("no sweep kernel for {cell:?}")))?;
        let groups = tn_groups(request.out_features);
        let mut shape = [0u32; 3];
        shape[0] = request.in_features as u32;
        shape[1] = request.out_features as u32;
        shape[2] = groups as u32;
        // Buffer 2 is the kernel's output, and for `SK = 1` that is the final
        // value: `partials` is written and never reduced, so binding it here
        // leaves `y` untouched. The two were the same binding in the first
        // version of this function and every `SK = 1` cell came back all zeros,
        // which is what `examples/sweep_geom.rs` caught.
        let out: &Buffer = if cell.sk == 1 { request.y } else { request.partials };
        let threads_per_group = cell.threads as u64;
        encoder.dispatch_1d_offset(
            pipeline,
            &[
                TensorRef::new(request.packed, request.packed_offset),
                TensorRef::new(request.x, 0),
                TensorRef::new(out, 0),
            ],
            params_bytes(&shape),
            groups * u64::from(cell.sk) * threads_per_group,
            threads_per_group,
        )?;
        if cell.sk == 1 {
            // The SK = 1 kernel writes the final value; there is nothing to sum.
            return Ok(());
        }
        let reduce = self
            .reduce
            .get(&cell.sk)
            .ok_or_else(|| KernelError::InvalidInput(format!("no sk{} reduction", cell.sk)))?;
        let n = request.out_features as u32;
        encoder.dispatch_groups(
            reduce,
            &[Some(request.partials), Some(request.y)],
            params_bytes(&n),
            u64::from(n).div_ceil(256),
            256,
        )
    }

    /// Encode the 256-state MCG decode probe.
    ///
    /// `out` must hold 256 floats. The sweep library carries its own copy of
    /// `decode_state_mcg`, so a caller that wants its table to be about *this
    /// engine's* codebook has to prove the copy is the same function; see the
    /// drift note at the top of `colmajor_sweep.metal`.
    pub fn dispatch_mcg_probe(&self, encoder: &mut MetalEncoder, out: &Buffer) -> Result<(), KernelError> {
        encoder.dispatch_groups(&self.mcg_probe, &[Some(out)], &[], 1, 256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_names_are_unique_and_cover_the_grid() {
        let all = cells();
        assert_eq!(all.len(), SK_VALUES.len() * TG_VALUES.len());
        let mut names: Vec<String> = all.iter().map(Cell::kernel_name).collect();
        names.sort();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count, "two cells share a kernel name");
    }

    #[test]
    fn baseline_is_a_real_cell_and_is_the_engine_geometry() {
        let baseline = Cell::baseline();
        assert!(
            cells().contains(&baseline),
            "the baseline cell has no kernel, so every gap would be against nothing"
        );
        assert_eq!(baseline.threads, 256, "every colmajor kernel hardcodes 8 simdgroups");
        assert_eq!(baseline.sk, 1, "the engine dispatches no split-K");
    }

    #[test]
    fn tn_groups_matches_the_live_dispatch_arithmetic() {
        // `encode_ishizuki_gemv`: `tiles.min(gemv_max_groups()).max(1)` with
        // `gemv_max_groups() == u64::MAX` is `out_features / 16`.
        for (out_features, groups) in [(5376u64, 336), (8192, 512), (21504, 1344), (2048, 128)] {
            assert_eq!(tn_groups(out_features as usize), groups);
        }
    }

    /// The engine must not be able to reach a sweep cell by accident.
    ///
    /// This is the invariant that keeps §21's conclusion ("autotuning is worth
    /// 0.76%, do not add a table") load-bearing rather than a preference: if a
    /// dispatch arm ever grew a match on one of these names, the 0.76% would
    /// silently become the engine's behaviour while the notes still said it was
    /// not worth shipping. The guard is on the *name*, because that is the only
    /// thing the dispatch has to match on.
    #[test]
    fn no_sweep_kernel_is_reachable_from_a_dispatch_arm() {
        let source = include_str!("linear.rs");
        for cell in cells() {
            let name = cell.kernel_name();
            assert!(
                !source.contains(&name),
                "linear.rs names {name}: the engine can now dispatch a sweep cell, so the \
                 measured 0.76% has to be re-derived rather than quoted"
            );
            assert!(
                !source.contains("gemv_sweep"),
                "linear.rs mentions a gemv_sweep symbol; the sweep library must stay \
                 unreachable from the engine's dispatch"
            );
        }
    }

    /// The generated block must be what the generator produces.
    ///
    /// AC6 says the generator runs twice and leaves a clean diff. That is about
    /// `linear.metal`, which `scripts/gen_exl3_gemv.py` rewrites; this file has
    /// its own generator and the same obligation, and it is cheaper to check here
    /// than to discover a stale 12-kernel block at measurement time.
    #[test]
    fn generated_block_matches_the_generator() {
        let source = include_str!("colmajor_sweep.metal");
        let mut expected = String::new();
        for sk in SK_VALUES {
            for threads in TG_VALUES {
                expected.push_str(&format!(
                    "kernel void gemv_sweep_k3_colmajor_sk{sk}_t{threads}(\n"
                ));
                expected.push_str(&format!("    constexpr uint SG = {threads}u / 32u;\n"));
            }
        }
        for sk in SK_VALUES.into_iter().filter(|sk| *sk > 1) {
            expected.push_str(&format!("kernel void gemv_sweep_reduce_sk{sk}(\n"));
        }
        let mut missing: Vec<&str> = Vec::new();
        for line in expected.lines() {
            if !source.contains(line) {
                missing.push(line);
            }
        }
        assert!(
            missing.is_empty(),
            "colmajor_sweep.metal is stale -- run \
             `cargo run -p kaplumba-kernels --example gen_colmajor_sweep`. Missing: {missing:?}"
        );
    }
}