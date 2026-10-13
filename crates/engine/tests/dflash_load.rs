//! spec/dflash-correctness T1/T2: the DFlash drafter's shard loads with every
//! tensor accounted for, the EXL3 packing agrees with the header, and the
//! RMSNorm weights are stored **verbatim**.
//!
//! That last claim used to be its opposite -- this file asserted a Gemma-style
//! `+1.0` offset, and the loader applied one. The drafter is Qwen3, whose
//! RMSNorm has no offset, so both were wrong together and this test passed. See
//! `DflashWeights::load` for the measurement and the exllamav3 citation.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use kaplumba_engine::dflash::{DflashLoadError, DflashWeights};
use kaplumba_kernels::exl3::Exl3Rate;

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

fn write_minimal_dflash_checkpoint(dir: &std::path::Path, misalign_trellis: bool) -> Vec<u8> {
    fs::create_dir_all(dir).expect("fixture directory");
    let padding = usize::from(misalign_trellis);
    let trellis_end = padding + 8192;
    let mcg_end = trellis_end + 4;
    let suh_end = mcg_end + 256;
    let svh_end = suh_end + 256;
    let norm_end = svh_end + 256;
    let mut entries = serde_json::Map::new();
    if misalign_trellis {
        entries.insert(
            "padding".to_string(),
            serde_json::json!({"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}),
        );
    }
    entries.insert(
        "fc.mcg".to_string(),
        serde_json::json!({"dtype": "I32", "shape": [1], "data_offsets": [trellis_end, mcg_end]}),
    );
    entries.insert(
        "fc.suh".to_string(),
        serde_json::json!({"dtype": "F16", "shape": [128], "data_offsets": [mcg_end, suh_end]}),
    );
    entries.insert(
        "fc.svh".to_string(),
        serde_json::json!({"dtype": "F16", "shape": [128], "data_offsets": [suh_end, svh_end]}),
    );
    entries.insert(
        "fc.trellis".to_string(),
        serde_json::json!({"dtype": "I16", "shape": [8, 8, 64], "data_offsets": [padding, trellis_end]}),
    );
    entries.insert(
        "norm.weight".to_string(),
        serde_json::json!({"dtype": "BF16", "shape": [128], "data_offsets": [svh_end, norm_end]}),
    );
    let header = serde_json::Value::Object(entries);
    let mut header = serde_json::to_vec(&header).expect("fixture header json");
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut packed = Vec::with_capacity(8192);
    for word in 0..4096_u16 {
        packed.extend_from_slice(&word.to_le_bytes());
    }
    let mut payload = Vec::with_capacity(norm_end);
    if misalign_trellis {
        payload.push(0xa5);
    }
    payload.extend_from_slice(&packed);
    payload.extend_from_slice(&kaplumba_engine::exl3::MCG_MULTIPLIER.to_le_bytes());
    for _ in 0..256 {
        payload.extend_from_slice(&0x3c00_u16.to_le_bytes());
    }
    for _ in 0..128 {
        payload.extend_from_slice(&0x3f80_u16.to_le_bytes());
    }
    let mut file = Vec::new();
    file.extend_from_slice(&(header.len() as u64).to_le_bytes());
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    fs::write(dir.join("model.safetensors"), &file).expect("write fixture shard");
    packed
}

#[test]
fn dflash_loader_uses_the_base_mmap_and_mlock_path() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("kaplumba-dflash-loader-{stamp}"));
    let expected_packed = write_minimal_dflash_checkpoint(&dir, false);
    let shard = kaplumba_engine::safetensors::read_shard_header(&dir.join("model.safetensors"))
        .expect("fixture header");
    let device = metal::Device::system_default().expect("device");
    {
        let weights = DflashWeights::load(&dir, &device).expect("load synthetic drafter");
        let line = weights.line("fc").expect("fc");
        let resident_mode = std::env::var_os("KAPLUMBA_RESIDENT_WEIGHTS").is_some();
        assert_eq!(
            weights.mapped_trellis_bytes,
            if resident_mode {
                0
            } else {
                expected_packed.len() as u64
            },
            "drafter packed weights must follow the base loader's selected residency path"
        );
        assert!(
            weights.pinned_bytes() <= shard.file_size,
            "page-rounded mlock accounting must stay within the shard mapping"
        );
        assert!(
            line.packed.is_empty(),
            "packed weights must not be copied to a Vec"
        );
        assert_eq!(line.packed_len, Some(expected_packed.len() / 2));
        let buffer = line.metal_buffer.as_ref().expect("packed Metal buffer");
        assert!(line.metal_offset + expected_packed.len() as u64 <= buffer.length());
        let actual = unsafe {
            std::slice::from_raw_parts(
                (buffer.contents() as *const u8).add(line.metal_offset as usize),
                expected_packed.len(),
            )
        };
        assert!(
            actual == expected_packed.as_slice(),
            "packed view bytes differ from the shard payload at offset {}",
            line.metal_offset
        );
        if !resident_mode {
            assert!(
                line.metal_offset > 0,
                "mmap view must retain the tensor offset"
            );
            assert_eq!(weights.mapped_trellis_bytes, 8192);
        }
        assert_eq!(weights.norm("norm").expect("norm").len(), 128);
    }
    fs::remove_dir_all(dir).expect("remove fixture");
}

#[test]
fn dflash_loader_realigns_misaligned_shard_views() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("kaplumba-dflash-unaligned-{stamp}"));
    let expected_packed = write_minimal_dflash_checkpoint(&dir, true);
    let device = metal::Device::system_default().expect("device");
    {
        let weights = DflashWeights::load(&dir, &device).expect("load unaligned synthetic drafter");
        let line = weights.line("fc").expect("fc");
        assert!(weights.unknown_tensors.iter().any(|name| name == "padding"));
        assert_eq!(weights.mapped_trellis_bytes, 0);
        assert_eq!(
            line.metal_offset, 0,
            "unaligned views must be copied to an aligned buffer"
        );
        let buffer = line.metal_buffer.as_ref().expect("packed Metal buffer");
        let actual = unsafe {
            std::slice::from_raw_parts(buffer.contents() as *const u8, expected_packed.len())
        };
        assert!(
            actual == expected_packed.as_slice(),
            "realignment changed packed bytes"
        );
    }
    fs::remove_dir_all(dir).expect("remove fixture");
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

    // fc is the multi-tap projection: 6 x 5376 -> 5376. Its trellis tile
    // depth declares the integer EXL3 rate (64 words = 4 bpw, 96 = 6 bpw).
    let fc = weights.line("fc").expect("fc line");
    assert_eq!(fc.in_features, 32_256);
    assert_eq!(fc.out_features, 5_376);
    let fc_shape = header_shape("fc.trellis");
    assert_eq!(&fc_shape[..2], &[2_016, 336]);
    let fc_bits = u8::try_from(fc_shape[2] / 16).expect("integer EXL3 bpw");
    assert_eq!(fc_shape[2], 16 * usize::from(fc_bits));
    assert_eq!(fc.rate, Exl3Rate::Integer(fc_bits));
    let shard = kaplumba_engine::safetensors::read_shard_header(&d.join("model.safetensors"))
        .expect("drafter header");
    let trellis_bytes: u64 = shard
        .tensors
        .iter()
        .filter(|(name, _)| name.ends_with(".trellis"))
        .map(|(_, info)| info.byte_len())
        .sum();
    for (name, info) in shard
        .tensors
        .iter()
        .filter(|(name, _)| name.ends_with(".trellis"))
    {
        let shape = &info.shape;
        assert_eq!(shape.len(), 3, "{name} rank");
        let bits = u8::try_from(shape[2] / 16).expect("integer EXL3 bpw");
        assert_eq!(shape[2], 16 * usize::from(bits), "{name} tile depth");
        let line = weights.line(name.trim_end_matches(".trellis")).expect(name);
        assert_eq!(line.rate, Exl3Rate::Integer(bits), "{name} rate");
        assert_eq!(
            line.packed_len,
            Some(info.byte_len() as usize / 2),
            "{name} words"
        );
    }
    let mapped_trellis_bytes: u64 = shard
        .tensors
        .iter()
        .filter(|(name, _)| name.ends_with(".trellis"))
        .filter_map(|(name, info)| {
            let (start, _) = shard.absolute_range(name).expect("trellis range");
            // EXL3's Metal code pointer is u32-aligned.
            (start % 4 == 0).then_some(info.byte_len())
        })
        .sum();
    let resident_mode = std::env::var_os("KAPLUMBA_RESIDENT_WEIGHTS").is_some();
    assert_eq!(
        weights.mapped_trellis_bytes,
        if resident_mode {
            0
        } else {
            mapped_trellis_bytes
        },
        "mapped-byte accounting must include aligned mmap views only; unaligned views use the checked copy fallback"
    );
    assert!(mapped_trellis_bytes <= trellis_bytes);
    assert!(
        weights.pinned_bytes() <= shard.file_size,
        "page-rounded mlock accounting must stay within the shard mapping"
    );
    eprintln!(
        "DFlash loader: mapped={} bytes, mlocked={} bytes",
        weights.mapped_trellis_bytes,
        weights.pinned_bytes()
    );
    assert!(
        fc.packed.is_empty(),
        "mapped EXL3 weights must not be copied to a Vec"
    );
    let fc_info = shard
        .tensors
        .get("fc.trellis")
        .expect("fc trellis metadata");
    assert_eq!(fc.packed_len, Some((fc_info.byte_len() / 2) as usize));
    let buffer = fc.metal_buffer.as_ref().expect("fc Metal view");
    let (start, _) = shard.absolute_range("fc.trellis").expect("fc range");
    let page_offset = fc.metal_offset;
    let packed_bytes = fc_info.byte_len() as usize;
    assert!(page_offset + packed_bytes as u64 <= buffer.length());
    let actual = unsafe {
        std::slice::from_raw_parts(
            (buffer.contents() as *const u8).add(page_offset as usize),
            packed_bytes,
        )
    };
    assert_eq!(
        actual,
        raw("fc.trellis"),
        "mapped view must bind the exact tensor bytes"
    );
    if !resident_mode {
        assert_ne!(start, 0, "fixture must exercise a nonzero shard offset");
        assert!(
            fc.metal_offset > 0,
            "mapped view must preserve the tensor's intra-view offset"
        );
    }

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
        assert_eq!(
            (q.in_features, q.out_features),
            (5_376, 8_192),
            "layer {layer} q"
        );
        assert_eq!(
            (k.in_features, k.out_features),
            (5_376, 1_024),
            "layer {layer} k"
        );
        assert_eq!(
            (o.in_features, o.out_features),
            (8_192, 5_376),
            "layer {layer} o"
        );
        assert_eq!(
            (gate.in_features, gate.out_features),
            (5_376, 10_752),
            "layer {layer} gate"
        );
        assert_eq!(
            (down.in_features, down.out_features),
            (10_752, 5_376),
            "layer {layer} down"
        );
        // 64 q heads x 128 = 8192 and 8 kv heads x 128 = 1024, from config.json.
        assert_eq!(q.out_features % (8 * 128), 0);
        assert_eq!(k.out_features % (8 * 128), 0);
    }

    // R1: the mcg multiplier is the engine's mcg codebook, not a stray value.
    for name in ["fc", "layers.0.self_attn.q_proj", "layers.4.mlp.down_proj"] {
        let bytes = raw(&format!("{name}.mcg"));
        let value = i32::from_le_bytes(bytes[..4].try_into().unwrap());
        assert_eq!(
            value as u32,
            kaplumba_engine::exl3::MCG_MULTIPLIER,
            "{name}"
        );
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
    std::fs::copy(
        d.join("model.safetensors"),
        staged.join("model.safetensors"),
    )
    .expect("copy");
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
