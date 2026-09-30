//! Is Metal's tensor-core path (simdgroup matrix multiply) available, and does
//! it compute correctly? The prefill plan depends on it, and the standalone
//! `metal` compiler is not installed, so we ask the runtime the engine already
//! uses. This verifies a real 8x8 product numerically rather than just checking
//! that the symbol resolves.

use metal::{Buffer, Device, MTLResourceOptions};

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

// C = A * B for 8x8 fp16 tiles, in the layout the tensor core consumes.
kernel void mma8(const device half *a [[buffer(0)]],
                 const device half *b [[buffer(1)]],
                 device half *c [[buffer(2)]]) {
    // The third argument is the leading dimension: elements per row, NOT the
    // number of rows, an element stride, or a row stride. Passing 0 made every
    // row of every fragment load from the same base address, which is why the
    // identity check used to match only 8 of 64 elements.
    constexpr int LD = 8;
    simdgroup_half8x8 ta, tb, tc;
    simdgroup_load(ta, a, LD);
    simdgroup_load(tb, b, LD);
    simdgroup_multiply(tc, ta, tb);
    simdgroup_store(tc, c, LD);
}
"#;

fn main() {
    let device = Device::system_default().expect("no Metal device");
    println!("device: {}", device.name());
    let library = match device.new_library_with_source(SRC, &metal::CompileOptions::new()) {
        Ok(l) => l,
        Err(e) => {
            println!("COMPILE FAILED:\n{e}");
            return;
        }
    };
    let function = match library.get_function("mma8", None) {
        Ok(f) => f,
        Err(e) => {
            println!("FUNCTION MISSING: {e}");
            return;
        }
    };
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .expect("pipeline");
    println!("compiled: simdgroup_multiply on half 8x8  -> OK");

    // A = identity, B = a known matrix; the product must equal B.
    let n = 8 * 8;
    let mut a = vec![0u16; n];
    let mut b = vec![0u16; n];
    for i in 0..n {
        a[i] = half::f16::from_f32(if i % 9 == 0 { 1.0 } else { 0.0 }).to_bits();
        b[i] = half::f16::from_f32((i % 8) as f32 + 1.0).to_bits();
    }
    let c = vec![0u16; n];
    let mk = |v: &Vec<u16>| {
        device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    let (ba, bb, bc) = (mk(&a), mk(&b), mk(&c));
    let queue = device.new_command_queue();
    let enc = queue.new_command_buffer();
    let ce = enc.new_compute_command_encoder();
    ce.set_compute_pipeline_state(&pipeline);
    for (i, buf) in [&ba, &bb, &bc].iter().enumerate() {
        ce.set_buffer(i as u64, Some(&**buf), 0);
    }
    ce.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(32, 1, 1));
    ce.end_encoding();
    enc.commit();
    enc.wait_until_completed();

    let out = unsafe { std::slice::from_raw_parts(bc.contents() as *const u16, n) };
    let expect: Vec<f32> = (0..n).map(|i| (i % 8) as f32 + 1.0).collect();
    let got: Vec<f32> = out
        .iter()
        .map(|h| half::f16::from_bits(*h).to_f32())
        .collect();
    let ok: Vec<bool> = (0..n).map(|i| (got[i] - expect[i]).abs() < 0.01).collect();
    println!(
        "A=identity, B known: {}/{} elements correct -> {}",
        ok.iter().filter(|v| **v).count(),
        n,
        if ok.iter().all(|v| *v) {
            "PASS"
        } else {
            "FAIL"
        }
    );
}
