//! Is `linear_gemv_ishizuki_k6_colmajor_contig4_bench` writing half its output?
//!
//! Isolates one question from the mess that is
//! `perf_probe_head_vs_layers.rs`: after four fixes to that file the 6 bpw arm
//! still reports exactly `131072 / 262144` non-zero outputs -- precisely half,
//! and precisely `n/2` for lm_head's 262144-wide head.
//!
//! Two candidate explanations, and they are distinguishable:
//!   (1) the kernel's epilogue covers only half its output (a kernel bug), or
//!   (2) the caller passed a grid or an output buffer that covers half
//!       (a harness bug).
//!
//! This dispatches the kernel directly with an explicit, minimal harness -- no
//! `LinearEncodeRequest`, no `Row`, no `Scratch` -- writing into a buffer
//! deliberately poisoned with a sentinel, so "not written" is distinguishable
//! from "written as zero".
//!
//! Run:
//!   cargo test --release -p kaplumba-engine --test probe_k6_half_output \
//!     -- --ignored --nocapture

use std::ffi::c_void;
use std::mem::size_of;

use kaplumba_kernels::encoder::{CommandBufferGpuTiming, MetalEncoder};
use kaplumba_kernels::linear::LinearKernels;
use metal::{Buffer, Device, MTLResourceOptions};

/// A value no sane dequant produces, so "untouched" is unambiguous.
const SENTINEL: f32 = -1.0e30;

#[test]
#[ignore = "microbenchmark; run explicitly"]
fn k6_writes_every_output_element() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let kernels = LinearKernels::shared().expect("linear kernels");

    // lm_head's real shape: K=5376, N=262144, 6 bpw.
    // 256 codes * 6 bits = 1536 bits = 48 u32 = 96 u16 per 16x16 tile.
    let k = 5376usize;
    let n = 262144usize;
    let words_per_tile = 96usize;
    let tiles_k = k / 16;
    let tiles_n = n / 16;
    let packed_elems = tiles_n * tiles_k * words_per_tile;

    let packed = device.new_buffer((packed_elems * 2) as u64, MTLResourceOptions::StorageModeShared);
    let input = device.new_buffer((k * 2) as u64, MTLResourceOptions::StorageModeShared);
    let output = device.new_buffer((n * 4) as u64, MTLResourceOptions::StorageModeShared);

    // Poison every output slot.
    unsafe {
        let p = output.contents() as *mut f32;
        for i in 0..n {
            *p.add(i) = SENTINEL;
        }
        // Non-zero inputs, so a working kernel cannot produce an all-zero row
        // by accident.
        let q = packed.contents() as *mut u16;
        for i in 0..packed_elems {
            *q.add(i) = 0x5A5Du16 ^ (i as u16);
        }
        let r = input.contents() as *mut f32;
        for i in 0..k {
            *r.add(i) = 0.001f32 * (i % 97) as f32 + 1.0;
        }
    }

    let shape = device.new_buffer(2 * size_of::<u32>() as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        let p = shape.contents() as *mut u32;
        *p = k as u32;
        *p.add(1) = n as u32;
    }

    let groups = (n / 16) as u64;
    let pipeline = kernels
        .ishizuki_gemv_pipeline_by_name("linear_gemv_ishizuki_k6_colmajor_contig4_bench")
        .expect("the k6 bench kernel must exist");

    let mut enc = MetalEncoder::new(&queue).expect("encoder");
    enc.set_max_dispatches(u32::MAX);
    enc.dispatch_groups(
        &pipeline,
        &[Some(&packed), Some(&input), Some(&output), Some(&shape)],
        &[],
        groups,
        256,
    )
    .expect("dispatch k6");
    enc.commit_and_wait().expect("commit");

    unsafe {
        let p = output.contents() as *const f32;
        let values = std::slice::from_raw_parts(p, n);
        let written = values.iter().filter(|v| **v != SENTINEL).count();
        let finite = values.iter().all(|v| v.is_finite());
        let nonzero = values.iter().filter(|v| **v != 0.0).count();

        eprintln!(
            "k6 direct: n={n} groups={groups} written={written} nonzero={nonzero} finite={finite}"
        );

        // Which half survived? Report the first and last untouched indices.
        let first_untouched = values.iter().position(|v| *v == SENTINEL);
        let last_written = values.iter().rposition(|v| *v != SENTINEL);
        eprintln!("  first untouched index: {first_untouched:?}");
        eprintln!("  last written index:    {last_written:?}");

        assert_eq!(
            written, n,
            "the k6 kernel left {}/{} output slots untouched (first untouched: {:?}, \
             last written: {:?}). With groups = n/16 and tiles_n = n/16 the grid covers \
             every output tile, so an unwritten slot is a kernel epilogue defect, not a \
             grid defect. Report the indices above.",
            n - written, n, first_untouched, last_written
        );
        assert!(finite, "k6 produced non-finite outputs");
        assert!(nonzero > n / 2, "k6 wrote only {nonzero} non-zero values");
    }
}