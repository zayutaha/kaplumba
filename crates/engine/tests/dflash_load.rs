//! spec/dflash-correctness T1/T2: the DFlash drafter's shard loads with every
//! tensor accounted for, the EXL3 packing agrees with the header, and the
//! RMSNorm weights are stored **verbatim**.
//!
//! That last claim used to be its opposite -- this file asserted a Gemma-style
//! `+1.0` offset, and the loader applied one. The drafter is Qwen3, whose
//! RMSNorm has no offset, so both were wrong together and this test passed. See
//! `DflashWeights::load` for the measurement and the exllamav3 citation.

use std::collections::BTreeMap;
use std::path::PathBuf;

use kaplumba_engine::dflash::{DflashLoadError, DflashWeights};

fn dir() -> PathBuf {
    std::env::var("KAPLUMBA_DFLASH_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/Users/zayaantaha/.kaplumba/models/Drafter"))
}

/// Read a tensor's bytes straight out of the shard, bypassing the loader, so
/// the test compares against the file rather than against itself.
fn raw(name: &str) -> Vec<u8> {
    let path = dir().join("model.safetensors");
    let shard = kaplumba_engine::safetensors::read_shard_header(&path).expect("header");
    let (start, end) = shard.absolute_range(name).expect("range");
    let mut file = std::fs::File::open(&path).expect("open");
    use std::io::{Read, Seek, SeekFrom};
    file.seek(SeekFrom::Start(start)).expect("seek");
    let mut bytes = vec![0u8; usize::try_from(end - start).unwrap()];
    file.read_exact(&mut bytes).expect("read");
    bytes
}

/// The shape the header declares, read through the public header API.
fn header_shape(name: &str) -> Vec<usize> {
    let path = dir().join("model.safetensors");
    let shard = kaplumba_engine::safetensors::read_shard_header(&path).expect("header");
    shard
        .tensors
        .get(name)
        .unwrap_or_else(|| panic!("{name} missing from the header"))
        .shape
        .clone()
}

/// Number of unique tensor stems (`.trellis` presence) the loader must produce.
fn expected_stems() -> usize {
    let path = dir().join("model.safetensors");
    let shard = kaplumba_engine::safetensors::read_shard_header(&path).expect("header");
    shard
        .tensors
        .keys()
        .filter(|name| name.ends_with(".trellis"))
        .count()
}

#[test]
fn dflash_load_has_every_tensor_and_the_right_shapes() {
    let d = dir();
    if !d.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present; skipping");
        return;
    }
    let device = metal::Device::system_default().expect("device");
    let weights = DflashWeights::load(&d, &device).expect("load");

    // R1: nothing unrecognised. This checkpoint is the base DFlash release, so
    // it should have no sinks / mask_embedding / aux norms at all.
    assert!(
        weights.unknown_tensors.is_empty(),
        "loader does not understand {:?}",
        weights.unknown_tensors
    );

    // R1: one EXL3 linear per trellis stem.
    assert_eq!(weights.lines.len(), expected_stems());

    // fc is the multi-tap projection: 6 x 5376 -> 5376. Its trellis shape is
    // [2016, 336, 64] (in/16, out/16, codebook), which is the whole point of
    // the "3-bit target, 4-bit drafter" pairing.
    let fc = weights.line("fc").expect("fc line");
    assert_eq!(fc.in_features, 32_256);
    assert_eq!(fc.out_features, 5_376);
    assert_eq!(header_shape("fc.trellis"), vec![2_016, 336, 64]);

    // Layer projections, from the header rather than from assumptions.
    for layer in 0..5 {
        let q = weights
            .line(&format!("layers.{layer}.self_attn.q_proj"))
            .expect("q_proj");
        let k = weights
            .line(&format!("layers.{layer}.self_attn.k_proj"))
            .expect("k_proj");
        let o = weights
            .line(&format!("layers.{layer}.self_attn.o_proj"))
            .expect("o_proj");
        let gate = weights
            .line(&format!("layers.{layer}.mlp.gate_proj"))
            .expect("gate_proj");
        let down = weights
            .line(&format!("layers.{layer}.mlp.down_proj"))
            .expect("down_proj");
        assert_eq!((q.in_features, q.out_features), (5_376, 8_192), "layer {layer} q");
        assert_eq!((k.in_features, k.out_features), (5_376, 1_024), "layer {layer} k");
        assert_eq!((o.in_features, o.out_features), (8_192, 5_376), "layer {layer} o");
        assert_eq!((gate.in_features, gate.out_features), (5_376, 10_752), "layer {layer} gate");
        assert_eq!((down.in_features, down.out_features), (10_752, 5_376), "layer {layer} down");
        // 64 q heads x 128 = 8192 and 8 kv heads x 128 = 1024, from config.json.
        assert_eq!(q.out_features % (8 * 128), 0);
        assert_eq!(k.out_features % (8 * 128), 0);
    }

    // R1: the mcg multiplier is the engine's mcg codebook, not a stray value.
    for name in ["fc", "layers.0.self_attn.q_proj", "layers.4.mlp.down_proj"] {
        let bytes = raw(&format!("{name}.mcg"));
        let value = i32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!(value as u32, kaplumba_engine::exl3::MCG_MULTIPLIER, "{name}");
    }
}

#[test]
fn dflash_norm_weights_are_stored_verbatim() {
    let d = dir();
    if !d.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present; skipping");
        return;
    }
    let device = metal::Device::system_default().expect("device");
    let weights = DflashWeights::load(&d, &device).expect("load");

    // Every RMSNorm in the checkpoint, by name.
    let mut names = vec!["hidden_norm".to_string(), "norm".to_string()];
    for layer in 0..5 {
        names.push(format!("layers.{layer}.input_layernorm"));
        names.push(format!("layers.{layer}.post_attention_layernorm"));
        names.push(format!("layers.{layer}.self_attn.q_norm"));
        names.push(format!("layers.{layer}.self_attn.k_norm"));
    }
    assert_eq!(names.len(), 2 + 5 * 4);

    for name in &names {
        let stored = weights
            .norms
            .get(name)
            .unwrap_or_else(|| panic!("{name} was not loaded"));
        let file = raw(&format!("{name}.weight"));
        assert_eq!(file.len() % 2, 0, "{name} is not BF16 pairs");
        assert_eq!(stored.len(), file.len() / 2, "{name} element count");

        // Bit-exact: BF16 widens to f32 exactly and f16 holds it (the values
        // are all in [0.002, 7.1), well inside f16's exact-BF16 range for these
        // magnitudes only insofar as the rows are bf16-rounded; the point is
        // that no arithmetic happens between the file and the runtime).
        for (index, pair) in file.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([pair[0], pair[1]]);
            let raw_value = f32::from_bits((bits as u32) << 16);
            let stored_value = half::f16::from_bits(stored[index]).to_f32();
            let expected = half::f16::from_f32(raw_value).to_f32();
            assert_eq!(
                stored_value, expected,
                "{name}[{index}]: stored {stored_value}, file {raw_value}"
            );
        }
    }

    // The measured values, so a regression in the convention is visible without
    // reading the whole row. `layers.0.input_layernorm[0]` is 0.7539 in the file;
    // a loader that adds Gemma's 1.0 gives 1.7539. `post_attention_layernorm`'s
    // mean is the sharpest tell: 0.311 verbatim against 1.311 offset, a 4.2x
    // error that compounded to a saturated softcap and acceptance 0.0000.
    let first = half::f16::from_bits(weights.norm("layers.0.input_layernorm").unwrap()[0]).to_f32();
    assert!(
        (first - 0.7539).abs() < 2e-3,
        "layers.0.input_layernorm[0] = {first}, expected the file's 0.7539 (0.7539 + 1.0 \
         would mean the Gemma offset is back)"
    );
    let post = weights.norm("layers.0.post_attention_layernorm").unwrap();
    let mean: f32 = post
        .iter()
        .map(|bits| half::f16::from_bits(*bits).to_f32())
        .sum::<f32>()
        / post.len() as f32;
    assert!(
        (mean - 0.311).abs() < 5e-3,
        "layers.0.post_attention_layernorm mean {mean}, expected 0.311 verbatim (1.311 means the \
         Gemma +1.0 offset is back, which flattens every draft to acceptance 0.0000)"
    );
}

#[test]
fn dflash_load_rejects_a_mismatched_codebook() {
    // T1's named-error requirement, on a synthetic shard: the mcg multiplier
    // is the only thing standing between us and silently decoding a mcg
    // trellis with the plain codebook.
    let d = dir();
    if !d.join("model.safetensors").exists() {
        eprintln!("drafter checkpoint not present; skipping");
        return;
    }
    let staged = d.parent().unwrap().join("Drafter_bad_mcg");
    let _ = std::fs::remove_dir_all(&staged);
    std::fs::create_dir_all(&staged).expect("mkdir");
    std::fs::copy(d.join("model.safetensors"), staged.join("model.safetensors")).expect("copy");
    std::fs::copy(d.join("config.json"), staged.join("config.json")).ok();

    // Corrupt one mcg in place: read the header, rewrite that tensor's four
    // bytes, leave everything else byte-identical.
    let path = staged.join("model.safetensors");
    let bytes = std::fs::read(&path).expect("read");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: BTreeMap<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("header json");
    let name = "fc.mcg";
    let entry = header.get(name).expect("fc.mcg in header");
    let offsets = entry
        .get("data_offsets")
        .and_then(|value| value.as_array())
        .expect("data_offsets");
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;
    let mut bytes = bytes;
    bytes[8 + header_len + start] ^= 0xff;
    let _ = end;
    std::fs::write(&path, &bytes).expect("write");

    let device = metal::Device::system_default().expect("device");
    let error = DflashWeights::load(&staged, &device)
        .err()
        .expect("a corrupted mcg must fail the load");
    let message = error.to_string();
    assert!(
        message.contains("fc.mcg") && message.contains("mcg"),
        "error must name the tensor and the codebook, got: {message}"
    );
    let _ = std::fs::remove_dir_all(&staged);

    // The error type itself must stay a load error, not a panic.
    assert!(matches!(error, DflashLoadError::Tensor { .. }));
}
