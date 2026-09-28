//! rms_norm at prefill shape: how bad is `rows` threads at rows=256, and what
//! is the best group geometry for a threadgroup-per-row norm?
//!
//! Metal has no runtime template instantiation, so the group sizes are
//! generated with a macro; the reduction body is shared.

use half::f16;
use metal::{Device, MTLResourceOptions};
use std::time::Instant;

const SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormParams {
    uint rows;
    uint width;
    float eps;
};

kernel void norm_per_row(
    const device half *input [[buffer(0)]],
    const device half *weight [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant NormParams &params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.rows) return;
    uint count = params.width;
    uint base = gid * count;
    float sum = 0.0f;
    for (uint i = 0; i < count; ++i) {
        float v = float(input[base + i]);
        sum += v * v;
    }
    float inv = rsqrt(sum / float(count) + params.eps);
    for (uint i = 0; i < count; ++i) {
        output[base + i] = half(float(input[base + i]) * inv * float(weight[i]));
    }
}

// One threadgroup of G threads per row, generated per G.
#define NORM_TG(G)                                                        \
kernel void norm_tg##G(                                                   \
    const device half *input [[buffer(0)]],                               \
    const device half *weight [[buffer(1)]],                              \
    device half *output [[buffer(2)]],                                    \
    constant NormParams &params [[buffer(3)]],                            \
    uint lid [[thread_position_in_threadgroup]],                          \
    uint tgid [[threadgroup_position_in_grid]]) {                         \
    if (tgid >= params.rows) return;                                      \
    threadgroup float partial[G];                                         \
    uint count = params.width;                                            \
    uint base = tgid * count;                                             \
    float sum = 0.0f;                                                     \
    for (uint i = lid; i < count; i += G)                                 \
        sum += float(input[base+i]) * float(input[base+i]);               \
    partial[lid] = sum;                                                   \
    threadgroup_barrier(mem_flags::mem_threadgroup);                      \
    for (uint s = G/2; s > 0; s >>= 1) {                                  \
        if (lid < s) partial[lid] += partial[lid + s];                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                  \
    }                                                                     \
    float inv = rsqrt(partial[0] / float(count) + params.eps);            \
    for (uint i = lid; i < count; i += G)                                 \
        output[base+i] = half(float(input[base+i]) * inv * float(weight[i])); \
}
NORM_TG(32)
NORM_TG(64)
NORM_TG(128)
NORM_TG(256)
NORM_TG(512)

// G threads, 4 rows per group, one 32-lane chunk per row, simd reduction only.
#define NORM_RPG(G)                                                       \
kernel void norm_rpg##G(                                                  \
    const device half *input [[buffer(0)]],                               \
    const device half *weight [[buffer(1)]],                              \
    device half *output [[buffer(2)]],                                    \
    constant NormParams &params [[buffer(3)]],                            \
    uint lid [[thread_position_in_threadgroup]],                          \
    uint tgid [[threadgroup_position_in_grid]]) {                         \
    const uint L = 32u;                                                   \
    const uint lane = lid % L;                                            \
    const uint row = tgid * (G / L) + lid / L;                            \
    if (row >= params.rows) return;                                       \
    uint count = params.width;                                            \
    uint base = row * count;                                              \
    float sum = 0.0f;                                                     \
    for (uint i = lane; i < count; i += L)                                \
        sum += float(input[base+i]) * float(input[base+i]);               \
    sum += simd_shuffle_down(sum, 16);                                    \
    sum += simd_shuffle_down(sum, 8);                                     \
    sum += simd_shuffle_down(sum, 4);                                     \
    sum += simd_shuffle_down(sum, 2);                                     \
    sum += simd_shuffle_down(sum, 1);                                     \
    if (lane == 0) {                                                      \
        float inv = rsqrt(sum / float(count) + params.eps);               \
        for (uint i = 0; i < count; ++i)                                  \
            output[base+i] = half(float(input[base+i]) * inv * float(weight[i])); \
    }                                                                     \
}
NORM_RPG(128)
NORM_RPG(256)
NORM_RPG(512)
NORM_RPG(1024)
"#;

fn time(
    queue: &metal::CommandQueue,
    pipeline: &metal::ComputePipelineState,
    input: &metal::Buffer,
    weight: &metal::Buffer,
    output: &metal::Buffer,
    rows: u32,
    width: u32,
    groups: u64,
    tpg: u64,
    dispatches: u64,
) -> f64 {
    let mut params_bytes = Vec::new();
    params_bytes.extend_from_slice(&rows.to_le_bytes());
    params_bytes.extend_from_slice(&width.to_le_bytes());
    params_bytes.extend_from_slice(&1e-5f32.to_le_bytes());
    let run = |e: &metal::ComputeCommandEncoderRef| {
        e.set_compute_pipeline_state(pipeline);
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(weight), 0);
        e.set_buffer(2, Some(output), 0);
        e.set_bytes(3, params_bytes.len() as u64, params_bytes.as_ptr() as *const _);
        e.dispatch_thread_groups(
            metal::MTLSize::new(groups, 1, 1),
            metal::MTLSize::new(tpg, 1, 1),
        );
    };
    for _ in 0..3 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        for _ in 0..dispatches {
            run(&e);
        }
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let mut best = f64::MAX;
    for _ in 0..5 {
        let cb = queue.new_command_buffer();
        let e = cb.new_compute_command_encoder();
        let t = Instant::now();
        for _ in 0..dispatches {
            run(&e);
        }
        e.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        best = best.min(t.elapsed().as_secs_f64() * 1e3 / dispatches as f64);
    }
    best
}

fn main() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .unwrap();

    for rows in [256u32, 64] {
        let width: u32 = 5376;
        let bytes = (rows * width * 2) as u64;
        let input = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let output = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let ones: Vec<u16> = vec![f16::from_f32(1.0).to_bits(); width as usize];
        let weight = device.new_buffer_with_data(
            ones.as_ptr() as *const std::ffi::c_void,
            (width * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        println!("== rms_norm rows={rows} width={width} ==");
        let f = lib.get_function("norm_per_row", None).unwrap();
        let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
        let ms = time(&queue, &p, &input, &weight, &output, rows, width, rows as u64, 64, 50);
        println!("  per_row              {ms:>9.4} ms");
        for g in [32u64, 64, 128, 256, 512] {
            let name = format!("norm_tg{g}");
            let f = lib.get_function(&name, None).unwrap();
            let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
            let ms = time(&queue, &p, &input, &weight, &output, rows, width, rows as u64, g, 50);
            println!("  tg G={g:<4}          {ms:>9.4} ms");
        }
        for g in [128u64, 256, 512, 1024] {
            let name = format!("norm_rpg{g}");
            let f = lib.get_function(&name, None).unwrap();
            let p = device.new_compute_pipeline_state_with_function(&f).unwrap();
            let groups = (rows as u64).div_ceil(g / 32);
            let ms = time(&queue, &p, &input, &weight, &output, rows, width, groups, g, 50);
            println!("  rpg G={g:<4} (4 rows) {ms:>8.4} ms");
        }
    }
}
