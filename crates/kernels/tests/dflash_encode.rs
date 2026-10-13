use std::ffi::c_void;
use std::mem::size_of_val;

use kaplumba_kernels::attention::{AttentionKernels, DflashBlockBufferRequest, DflashBlockRequest};
use kaplumba_kernels::elements::ElementKernels;
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, Device, MTLResourceOptions};

fn buffer_with<T>(device: &Device, values: &[T]) -> Buffer {
    device.new_buffer_with_data(
        values.as_ptr() as *const c_void,
        size_of_val(values) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn zero_buffer(device: &Device, bytes: usize) -> Buffer {
    device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared)
}

fn read_u16(buffer: &Buffer, count: usize) -> Vec<u16> {
    let mut values = vec![0u16; count];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents() as *const u16, values.as_mut_ptr(), count);
    }
    values
}

fn read_f32(buffer: &Buffer, count: usize) -> Vec<f32> {
    let mut values = vec![0.0f32; count];
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents() as *const f32, values.as_mut_ptr(), count);
    }
    values
}

#[test]
fn dflash_f32_encode_operations_match_slice_paths_and_validate_buffers() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let elements = ElementKernels::shared().expect("element kernels");

    // Multiple rows and large residual values exercise the f32-input/f16-output
    // boundary without narrowing the residual stream before normalization.
    let rows = 4;
    let width = 32;
    let input: Vec<f32> = (0..rows * width)
        .map(|i| 150_000.0 + (i as f32 - 40.0) * 73.0)
        .collect();
    let weight: Vec<u16> = (0..width)
        .map(|i| half::f16::from_f32(0.75 + i as f32 * 0.002).to_bits())
        .collect();
    let expected = elements
        .rms_norm_f32_in(&input, &weight, rows, width, 1e-5)
        .expect("slice f32-input norm");
    let input_buffer = buffer_with(&device, &input);
    let weight_buffer = buffer_with(&device, &weight);
    let output_buffer = zero_buffer(&device, rows * width * size_of::<u16>());
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    elements
        .encode_rms_norm_f32_in(
            &mut encoder,
            &input_buffer,
            &weight_buffer,
            &output_buffer,
            rows,
            width,
            1e-5,
        )
        .expect("encode f32-input norm");
    encoder.commit_and_wait().expect("norm completion");
    assert_eq!(read_u16(&output_buffer, rows * width), expected);

    let left: Vec<f32> = (0..67).map(|i| 100_000.0 + i as f32 * 19.0).collect();
    let right: Vec<f32> = (0..67).map(|i| 50_000.0 - i as f32 * 7.0).collect();
    let expected = elements
        .residual_add_f32(&left, &right)
        .expect("slice f32 residual");
    assert!(expected.iter().any(|value| *value > 65_504.0));
    let left_buffer = buffer_with(&device, &left);
    let right_buffer = buffer_with(&device, &right);
    let output_buffer = zero_buffer(&device, left.len() * size_of::<f32>());
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    elements
        .encode_residual_add_f32(
            &mut encoder,
            &left_buffer,
            &right_buffer,
            &output_buffer,
            left.len(),
        )
        .expect("encode f32 residual");
    encoder.commit_and_wait().expect("residual completion");
    assert_eq!(read_f32(&output_buffer, left.len()), expected);

    let gate: Vec<u16> = (0..67)
        .map(|i| half::f16::from_f32((i as f32 - 31.0) * 0.25).to_bits())
        .collect();
    let up: Vec<u16> = (0..67)
        .map(|i| half::f16::from_f32(2.0 - i as f32 * 0.015).to_bits())
        .collect();
    let expected = elements.gated_silu(&gate, &up).expect("slice gated SiLU");
    let gate_buffer = buffer_with(&device, &gate);
    let up_buffer = buffer_with(&device, &up);
    let silu_output = zero_buffer(&device, gate.len() * size_of::<u16>());
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    elements
        .encode_gated_silu(
            &mut encoder,
            &gate_buffer,
            &up_buffer,
            &silu_output,
            gate.len(),
        )
        .expect("encode gated SiLU");
    encoder.commit_and_wait().expect("SiLU completion");
    assert_eq!(read_u16(&silu_output, gate.len()), expected);

    let short = zero_buffer(&device, left.len() * size_of::<f32>() - 1);
    let short_u16 = zero_buffer(&device, gate.len() * size_of::<u16>() - 1);
    let mut encoder = MetalEncoder::new(&queue).expect("encoder");
    assert!(
        elements
            .encode_residual_add_f32(
                &mut encoder,
                &left_buffer,
                &right_buffer,
                &short,
                left.len(),
            )
            .is_err()
    );
    assert!(
        elements
            .encode_gated_silu(
                &mut encoder,
                &gate_buffer,
                &up_buffer,
                &short_u16,
                gate.len(),
            )
            .is_err()
    );
    assert!(
        elements
            .encode_gated_silu(&mut encoder, &gate_buffer, &up_buffer, &silu_output, 0,)
            .is_err()
    );
    assert!(
        elements
            .encode_residual_add_f32(&mut encoder, &left_buffer, &right_buffer, &output_buffer, 0,)
            .is_err()
    );
    assert!(
        elements
            .encode_rms_norm_f32_in(
                &mut encoder,
                &input_buffer,
                &weight_buffer,
                &output_buffer,
                rows,
                width,
                0.0,
            )
            .is_err()
    );
    assert!(
        elements
            .encode_rms_norm_f32_in(
                &mut encoder,
                &short,
                &weight_buffer,
                &output_buffer,
                rows,
                width,
                1e-5,
            )
            .is_err()
    );
}

#[test]
fn dflash_block_attention_encoder_matches_slice_path() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let attention = AttentionKernels::new().expect("attention kernels");
    let rows = 4;
    let q_heads = 4;
    let kv_heads = 2;
    let head_dim = 32;
    let q_count = rows * q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let queries: Vec<u16> = (0..q_count)
        .map(|i| half::f16::from_f32((i as f32 % 37.0 - 18.0) * 0.01).to_bits())
        .collect();
    let block_keys: Vec<u16> = (0..rows * kv_width)
        .map(|i| half::f16::from_f32((i as f32 % 29.0 - 14.0) * 0.02).to_bits())
        .collect();
    let block_values: Vec<u16> = (0..rows * kv_width)
        .map(|i| half::f16::from_f32((i as f32 % 31.0 - 15.0) * 0.015).to_bits())
        .collect();

    for (n_ctx, window, bidirectional, first_position) in [
        (3, 0, true, 10_u32),
        (3, 3, false, 10_u32),
        (0, 0, true, 0_u32),
    ] {
        let ctx_keys: Vec<u16> = (0..n_ctx * kv_width)
            .map(|i| half::f16::from_f32((i as f32 % 23.0 - 11.0) * 0.013).to_bits())
            .collect();
        let ctx_values: Vec<u16> = (0..n_ctx * kv_width)
            .map(|i| half::f16::from_f32((i as f32 % 19.0 - 9.0) * 0.017).to_bits())
            .collect();
        let key_positions: Vec<u32> = (0..n_ctx + rows)
            .map(|i| first_position + i as u32)
            .collect();
        let expected = attention
            .dflash_block_attention(DflashBlockRequest {
                queries: &queries,
                ctx_keys: &ctx_keys,
                ctx_values: &ctx_values,
                block_keys: &block_keys,
                block_values: &block_values,
                key_positions: &key_positions,
                rows,
                n_ctx,
                q_heads,
                kv_heads,
                head_dim,
                scale: 1.0 / (head_dim as f32).sqrt(),
                window,
                bidirectional,
            })
            .expect("slice DFlash attention");

        let query_buffer = buffer_with(&device, &queries);
        let ctx_key_buffer = if ctx_keys.is_empty() {
            buffer_with(&device, &[0_u16])
        } else {
            buffer_with(&device, &ctx_keys)
        };
        let ctx_value_buffer = if ctx_values.is_empty() {
            buffer_with(&device, &[0_u16])
        } else {
            buffer_with(&device, &ctx_values)
        };
        let block_key_buffer = buffer_with(&device, &block_keys);
        let block_value_buffer = buffer_with(&device, &block_values);
        let position_buffer = buffer_with(&device, &key_positions);
        let output_buffer = zero_buffer(&device, q_count * size_of::<u16>());
        let request = DflashBlockBufferRequest {
            queries: &query_buffer,
            ctx_keys: &ctx_key_buffer,
            ctx_values: &ctx_value_buffer,
            block_keys: &block_key_buffer,
            block_values: &block_value_buffer,
            key_positions: &position_buffer,
            output: &output_buffer,
            rows,
            n_ctx,
            q_heads,
            kv_heads,
            head_dim,
            scale: 1.0 / (head_dim as f32).sqrt(),
            window,
            bidirectional,
        };
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        attention
            .encode_dflash_block_attention(&mut encoder, request)
            .expect("encode DFlash attention");
        encoder.commit_and_wait().expect("attention completion");
        assert_eq!(read_u16(&output_buffer, q_count), expected);

        let short_output = zero_buffer(&device, q_count * size_of::<u16>() - 1);
        let mut invalid_encoder = MetalEncoder::new(&queue).expect("encoder");
        assert!(
            attention
                .encode_dflash_block_attention(
                    &mut invalid_encoder,
                    DflashBlockBufferRequest {
                        output: &short_output,
                        ..request
                    },
                )
                .is_err()
        );
    }
}
