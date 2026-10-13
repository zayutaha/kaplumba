use std::ffi::c_void;
use std::mem::size_of_val;

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

    let short = zero_buffer(&device, left.len() * size_of::<f32>() - 1);
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
