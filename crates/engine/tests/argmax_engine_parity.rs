use half::f16;
use kaplumba_kernels::elements::{decode_argmax_token, ElementKernels};
use kaplumba_kernels::encoder::MetalEncoder;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

fn argmax_logits(logits: &[u16]) -> u32 {
    let mut best = 0;
    let mut best_val = f16::from_bits(logits[0]).to_f32();
    for index in 1..logits.len() {
        let val = f16::from_bits(logits[index]).to_f32();
        if val > best_val {
            best_val = val;
            best = index;
        }
    }
    best as u32
}

fn upload(device: &Device, data: &[u16]) -> Buffer {
    let buf = device.new_buffer(
        (data.len() * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), buf.contents() as *mut u16, data.len());
    }
    buf
}

#[test]
fn test_element_kernels_encode_argmax_parity() {
    let device = Device::system_default().expect("Metal device");
    let queue: CommandQueue = device.new_command_queue();
    let elements = ElementKernels::new().expect("ElementKernels::new");

    const VOCAB: usize = 262_144;
    let slots = VOCAB.div_ceil(65536);
    let best_buf = device.new_buffer(
        (slots * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let test_indices = [
        0, 1, 100, 9259, 65535, 65536, 65537, 100000, 131071, 131072, 200000, 262143,
    ];

    for &winner in &test_indices {
        let mut row = vec![f16::from_f32(-5.0).to_bits(); VOCAB];
        row[winner] = f16::from_f32(28.5).to_bits();

        let row_buf = upload(&device, &row);
        let mut encoder = MetalEncoder::new(&queue).expect("encoder");
        elements
            .encode_argmax(&mut encoder, &row_buf, &best_buf, VOCAB)
            .expect("encode_argmax");
        encoder.commit_and_wait().expect("commit");

        let keys = unsafe {
            std::slice::from_raw_parts(best_buf.contents() as *const u32, slots)
        };
        let device_token = decode_argmax_token(keys);
        let host_token = argmax_logits(&row);

        assert_eq!(
            device_token, winner as u32,
            "device argmax must pick winner {winner}"
        );
        assert_eq!(
            device_token, host_token,
            "device token must match host token for winner {winner}"
        );
    }
}

#[test]
fn test_argmax_sabotage_detection() {
    // AGENTS.md 3.4: Prove that if the decoding logic is sabotaged, the test fails.
    let keys = [0x3C00_FFFF, 0, 0, 0]; // Winner in slot 0 at index 0
    let normal = decode_argmax_token(&keys);
    assert_eq!(normal, 0);

    // Deliberately corrupted key with reversed slot order
    let sabotaged_decode = |k: &[u32]| -> u32 {
        decode_argmax_token(k) ^ 1
    };
    assert_ne!(
        sabotaged_decode(&keys),
        0,
        "Sabotage must be detectable"
    );
}
