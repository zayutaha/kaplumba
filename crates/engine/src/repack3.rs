//! spec/trellis-dense-repack RT1: lossless trellis-blob repackers.
//!
//! The EXL3 trellis for one linear is a flat blob of packed u16 words; tile
//! `(it, ot)` (16×16 weights) lives at `(it*out_tiles + ot)` tiles from the
//! start, each tile `tile_u16` u16 words wide. The decode kernels walk tiles
//! in `it` order; in the native layout consecutive `it` for fixed `ot` are
//! `out_tiles*tile_u16` words apart, i.e. short strided bursts (RT0 showed
//! this caps k=3 streams ~50-75 GB/s while lm_head's contiguous 64 KB runs
//! reach ~228 GB/s).
//!
//! `Wide4` reorders the blob so each 4-out-tile block × full in-tile range is
//! ONE contiguous span (slab): order `[otblock][it][ot_local]`. A densen decode
//! warp then advances linearly through a slab — the same per-lane accumulation
//! order over `it`, hence bit-exact outputs with today's kernels — while reads
//! become long sequential runs (the target mechanism).
//!
//! All transforms are pure bit permutations of the SAME words: unpacking a
//! dense blob must reproduce the original blob byte-for-byte. No values change.

use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenseLayout {
    /// Slabs of 4 out-tiles, ordered `[otblock][it][ot_local]`. Tile content,
    /// word order and padding inside a tile are preserved verbatim.
    Wide4,
}

#[derive(Debug, Error)]
pub enum RepackError {
    #[error("out_tiles={0} is not a multiple of 4 (layout requires it)")]
    OutTilesNotDivisible(usize, DenseLayout),
    #[error("blob length {0} does not match in_tiles={1} × out_tiles={2} × tile_u16={3} u16 words")]
    BadLength(usize, usize, usize, usize),
}

/// u16 words per tile, validated from the blob geometry.
pub fn tile_u16_of(blob_len: usize, in_tiles: usize, out_tiles: usize) -> Result<usize, RepackError> {
    let tiles = in_tiles
        .checked_mul(out_tiles)
        .filter(|t| *t > 0)
        .ok_or(RepackError::BadLength(blob_len, in_tiles, out_tiles, 0))?;
    let w = blob_len / 2 / tiles;
    if blob_len != tiles * w * 2 {
        return Err(RepackError::BadLength(blob_len, in_tiles, out_tiles, w));
    }
    Ok(w)
}

/// `[otblock][it][ot_local]` repack. Returns dense `Vec<u8>` of the same
/// number of bytes as `blob`.
pub fn pack_wide4(
    blob: &[u8],
    in_tiles: usize,
    out_tiles: usize,
) -> Result<Vec<u8>, RepackError> {
    if out_tiles == 0 || out_tiles % 4 != 0 {
        return Err(RepackError::OutTilesNotDivisible(out_tiles, DenseLayout::Wide4));
    }
    let tile = tile_u16_of(blob.len(), in_tiles, out_tiles)?;
    let words = blob.len() / 2;
    let tile_words = tile;
    let tile_bytes = tile_words * 2;
    let mut dense = vec![0u8; blob.len()];
    let src = |it: usize, ot: usize| (it * out_tiles + ot) * tile_bytes;
    let mut dst = 0usize;
    for s in 0..out_tiles / 4 {
        for it in 0..in_tiles {
            let base_ot = s * 4;
            for otl in 0..4 {
                let s0 = src(it, base_ot + otl);
                let d0 = dst;
                dense[d0..d0 + tile_bytes].copy_from_slice(&blob[s0..s0 + tile_bytes]);
                dst += tile_bytes;
            }
        }
    }
    debug_assert_eq!(dst, words * 2);
    Ok(dense)
}

/// Inverse of [`pack_wide4`]: reconstructs the original word blob.
pub fn unpack_wide4(
    dense: &[u8],
    in_tiles: usize,
    out_tiles: usize,
) -> Result<Vec<u8>, RepackError> {
    if out_tiles == 0 || out_tiles % 4 != 0 {
        return Err(RepackError::OutTilesNotDivisible(out_tiles, DenseLayout::Wide4));
    }
    let tile = tile_u16_of(dense.len(), in_tiles, out_tiles)?;
    let blob = vec![0u8; dense.len()];
    let tile_bytes = tile * 2;
    let mut out = blob;
    let dst = |it: usize, ot: usize| (it * out_tiles + ot) * tile_bytes;
    let mut src = 0usize;
    for s in 0..out_tiles / 4 {
        for it in 0..in_tiles {
            let base_ot = s * 4;
            for otl in 0..4 {
                let s0 = src;
                let d0 = dst(it, base_ot + otl);
                out[d0..d0 + tile_bytes].copy_from_slice(&dense[s0..s0 + tile_bytes]);
                src += tile_bytes;
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_blob(in_tiles: usize, out_tiles: usize, tile_u16: usize) -> Vec<u8> {
        let n = in_tiles * out_tiles * tile_u16 * 2;
        (0..n)
            .map(|i| ((i as u32).wrapping_mul(0x9E3779B9).rotate_left(7) ^ 0x5A5A5A5A) as u8)
            .collect()
    }

    #[test]
    fn wide4_roundtrips_synthetic_shapes() {
        for (it, ot, tw) in [
            (1usize, 4usize, 24usize),
            (7, 4, 48),
            (336, 1344, 48),
            (1344, 336, 48),
            (512, 512, 48),
            (336, 4, 96),
            (2, 4, 1), // degenerate 2-byte tiles
        ] {
            let blob = synth_blob(it, ot, tw);
            let dense = pack_wide4(&blob, it, ot).unwrap();
            assert_eq!(dense.len(), blob.len());
            let back = unpack_wide4(&dense, it, ot).unwrap();
            assert_eq!(back, blob, "round-trip (it={it} ot={ot} tw={tw})");
        }
    }

    #[test]
    fn wide4_rejects_unaligned_out_tiles() {
        let blob = synth_blob(4, 6, 48);
        assert!(matches!(
            pack_wide4(&blob, 4, 6),
            Err(RepackError::OutTilesNotDivisible(6, DenseLayout::Wide4))
        ));
    }

    #[test]
    fn wide4_rejects_bad_length() {
        let blob = synth_blob(4, 4, 48);
        let short = &blob[..blob.len() - 2];
        assert!(matches!(
            pack_wide4(short, 4, 4),
            Err(RepackError::BadLength(..))
        ));
    }

    #[test]
    fn wide4_real_artifact_roundtrips() {
        let Some(dir) = artifact_dir() else {
            eprintln!("skip: no artifact (KAPLUMBA_MODEL_DIR)");
            return;
        };
        // (<layer>, <which>) spot layers from the spec: 0, 29, 59.
        for (layer, which) in [
            (0usize, "self_attn.v_proj"),
            (0, "mlp.down_proj"),
            (29, "mlp.gate_proj"),
            (59, "self_attn.o_proj"),
        ] {
            let base = format!("model.language_model.layers.{layer}.{which}");
            let (blob, _, shape) = read_tensor(&dir, &format!("{base}.trellis")).unwrap();
            // safetensors stores shape as [in_tiles, out_tiles, words_per_tile].
            assert_eq!(shape.len(), 3, "{base} shape {shape:?}");
            let (it, ot) = (shape[0], shape[1]);
            assert!(it > 0 && ot > 0 && ot % 4 == 0, "{base} {it}x{ot}");
            let tw = shape[2];
            assert_eq!(tw, 48, "{base} k=3 => 48 u16/tile");
            assert_eq!(blob.len(), it * ot * tw * 2, "{base} byte len");

            let dense = pack_wide4(&blob, it, ot).unwrap();
            assert_eq!(dense.len(), blob.len());
            let back = unpack_wide4(&dense, it, ot).unwrap();
            assert_eq!(back, blob, "{base} round-trip");
            // Slab 0 sanity: dense starts with it=0 ots 0..4 contiguous.
            let tile = tw * 2;
            assert_eq!(&dense[0..tile], &blob[0..tile], "{base} first tile verbatim");
            for otl in 1..4 {
                let d0 = otl * tile;
                let b0 = (0 * ot + otl) * tile;
                assert_eq!(
                    &dense[d0..d0 + tile],
                    &blob[b0..b0 + tile],
                    "{base} slab0 tile ot_local={otl}"
                );
            }
        }
    }

    fn artifact_dir() -> Option<std::path::PathBuf> {
        let dir = match std::env::var("KAPLUMBA_MODEL_DIR") {
            Ok(d) => std::path::PathBuf::from(d),
            Err(_) => {
                let h = std::env::var("HOME").ok()?;
                std::path::PathBuf::from(format!("{h}/.kaplumba/models/Gxma/Gxma"))
            }
        };
        dir.join("config.json").is_file().then_some(dir)
    }

    /// Read one tensor's raw bytes + dtype + shape straight from the shards
    /// (mirrors gemma4.rs tests' helper; avoids GPU residency just to pack).
    fn read_tensor(
        dir: &std::path::Path,
        name: &str,
    ) -> Option<(Vec<u8>, String, Vec<usize>)> {
        use std::io::{Read, Seek, SeekFrom};
        let index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?,
        )
        .ok()?;
        let shard = index["weight_map"][name].as_str()?.to_string();
        let mut f = std::fs::File::open(dir.join(&shard)).ok()?;
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).ok()?;
        let hlen = u64::from_le_bytes(len_buf) as usize;
        let mut hbuf = vec![0u8; hlen];
        f.read_exact(&mut hbuf).ok()?;
        let header: serde_json::Value = serde_json::from_slice(&hbuf).ok()?;
        let t = header.get(name)?;
        let dtype = t["dtype"].as_str()?.to_string();
        let shape: Vec<usize> = t["shape"]
            .as_array()?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();
        let off = t["data_offsets"][0].as_u64()?;
        let end = t["data_offsets"][1].as_u64()?;
        f.seek(SeekFrom::Start(8 + hlen as u64 + off)).ok()?;
        let mut bytes = vec![0u8; (end - off) as usize];
        f.read_exact(&mut bytes).ok()?;
        Some((bytes, dtype, shape))
    }
}