//! Every compute pipeline in the crate exists before the timed loops do.
//!
//! # What is being claimed
//!
//! A decode benchmark quotes `decode_tok_s`, which is a sum of per-step wall
//! times. If a kernel is compiled inside that loop, the step that reaches it
//! carries Metal's shader build, the mean moves, and the number describes
//! compilation rather than the kernel. The error is not small and not signed: a
//! variant reached for the first time on step 40 looks catastrophically slow and
//! its sibling looks fast, so an A/B can come out backwards.
//!
//! The engine's defence is that all three kernel families build every pipeline
//! eagerly, inside `new()`, behind a `OnceLock`. That is a claim about code, so
//! this file makes it a claim about a run: it builds the same three objects the
//! model does and asserts that decode-shaped dispatches add nothing to the build
//! count.
//!
//! # What this does NOT prove, and it is worth being exact
//!
//! * It counts `newComputePipelineStateWithFunction` calls. A kernel absent
//!   from the count cannot be measuring compilation. It says nothing about a
//!   kernel's *first dispatch*, which is a separate cost -- loading the built
//!   pipeline into the GPU's instruction memory -- and which is what
//!   `warmup_decode` exists for. The two are easily conflated and are not the
//!   same number.
//! * It builds the families, not a model. A kernel reachable only from a real
//!   `TextModel` forward is not dispatched here. That is why
//!   `KAPLUMBA_ASSERT_NO_LATE_PIPELINES=1` also exists and runs inside the real
//!   decode loop: this file is the fast always-on check, the env var covers
//!   what the test cannot reach.
//! * It is one test binary, so the `OnceLock` caches are fresh and every family
//!   is genuinely constructed here rather than found already built. That is the
//!   point: it is the only place where "the family builds its pipelines in
//!   `new()`" is observed rather than assumed.

use kaplumba_kernels::attention::AttentionKernels;
use kaplumba_kernels::elements::ElementKernels;
use kaplumba_kernels::encoder::MetalEncoder;
use kaplumba_kernels::exl3::{Exl3Codebook, Exl3Rate};
use kaplumba_kernels::linear::{LinearEncodeRequest, LinearKernels};
use kaplumba_kernels::pipeline_audit;
use metal::{Buffer, CommandQueue, Device, MTLResourceOptions};

/// The model's two bit rates, 3 bpw for the decoder linears and 6 bpw for
/// `lm_head` (`validate_current_graph` pins both).
const RATES: [u8; 2] = [3, 6];
/// `packed_offset % 4 == 0` takes the ishizuki arm; anything else falls to the
/// unaligned EXL3 arm. `lm_head`'s trellis really does start at `% 4 == 2`, so
/// the second value is not hypothetical.
const OFFSETS: [u64; 2] = [0, 2];
/// One of each distinct decoder shape: 5376x5376 (q/k/v, o), 5376x21504
/// (gate/up/down) and 5376x262144 (`lm_head`).
const OUT_FEATURES: [usize; 3] = [5376, 21504, 262144];
const IN_FEATURES: usize = 5376;

/// A shared buffer of zeroes.
///
/// `StorageModeShared` and an explicit zero, because every kernel here reads
/// these buffers and an uninitialised one is a silent wrong-answer generator
/// that has nothing to do with what this file is checking.
fn zeroes(device: &Device, bytes: usize) -> Buffer {
    let buffer = device.new_buffer(bytes as u64, MTLResourceOptions::StorageModeShared);
    unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, bytes) };
    buffer
}

/// Serialises the tests in this binary against each other.
///
/// The audit's window is process-global, so two tests running concurrently both
/// see each other's builds: `the_attention_family_...` failed here with a list
/// of 140 `linear_*` kernels it had nothing to do with, because
/// `decode_shaped_dispatches_compile_nothing` was inside `LinearKernels::shared()`
/// at the same moment. That is the worst shape of test failure -- a red test
/// blaming the wrong code -- and it is silent about the property under test.
///
/// `OnceLock` construction is one-shot per family, so the alternative (one test
/// binary per family) would mean three files; the lock costs one line and keeps
/// the argument in one place.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|error| error.into_inner())
}

/// Encode `body`, commit, and wait.
///
/// Deliberately a real commit and a real wait: a dispatch that is encoded and
/// dropped never reaches the GPU. A pipeline bound only on that path would look
/// used to the encoder and unused to the driver, and this file is about the
/// driver's view.
fn drive<F>(queue: &CommandQueue, body: F)
where
    F: FnOnce(&mut MetalEncoder<'_>) -> Result<(), String>,
{
    let mut encoder = MetalEncoder::new(queue).expect("encoder");
    body(&mut encoder).expect("dispatch");
    encoder.commit_and_wait().expect("wait");
}

/// Load completes and every pipeline exists: the count is non-zero, the time is
/// charged, and neither moves while decode-shaped work runs.
///
/// This is the test that fails if someone moves a `pipeline_audit::build` out of
/// `new()` and into a dispatch path, which is the change that would quietly tax
/// every future measurement.
#[test]
fn decode_shaped_dispatches_compile_nothing() {
    let _serial = serial();
    let Some(device) = Device::system_default() else {
        // No Metal device: nothing can be dispatched, so there is nothing to
        // prove. This is the convention every other Metal test here follows,
        // and unlike the audit's own probes there is no vacuous-pass risk in
        // claiming it -- the sibling test in `pipeline_audit` is the one that
        // must never no-op.
        return;
    };

    // Fresh in this process: this binary does not otherwise build these.
    let linear = LinearKernels::shared().expect("linear kernels");
    let elements = ElementKernels::shared().expect("element kernels");

    let after_build = pipeline_audit::build_count();
    assert!(
        after_build > 0,
        "no pipeline was built, so the rest of this test would be vacuous: the \
         families under test constructed nothing"
    );
    assert!(
        pipeline_audit::build_micros() > 0,
        "pipelines were built but no time was charged to them, so the timing half \
         of the audit is not measuring what it claims to"
    );

    let packed = zeroes(&device, IN_FEATURES * 262144 / 8 + 4096);
    let input = zeroes(&device, IN_FEATURES * 2 + 4096);
    let input_signs = zeroes(&device, IN_FEATURES * 2 + 4096);
    let output_signs = zeroes(&device, 262144 * 2 + 4096);
    let transformed = zeroes(&device, IN_FEATURES * 2 + 4096);
    let raw = zeroes(&device, 262144 * 4 + 4096);
    let output = zeroes(&device, 262144 * 2 + 4096);
    let queue = device.new_command_queue();

    for rate in RATES {
        for packed_offset in OFFSETS {
            for out_features in OUT_FEATURES {
                // `perm_mode` 1 is the colmajor selection, which picks a
                // different kernel family for the same rate, so both are
                // exercised: the dispatch is a function of this axis as much as
                // of the rate.
                for perm_mode in [0u32, 1u32] {
                    let request = LinearEncodeRequest {
                        packed: &packed,
                        packed_offset,
                        perm_mode,
                        input: &input,
                        input_signs: &input_signs,
                        output_signs: &output_signs,
                        transformed: &transformed,
                        raw: &raw,
                        output: &output,
                        in_features: IN_FEATURES,
                        out_features,
                        rate: Exl3Rate::Integer(rate),
                        codebook: Exl3Codebook::Mcg,
                        pretransformed: false,
                    };
                    drive(&queue, |encoder| {
                        linear
                            .encode_gemv(encoder, request)
                            .map_err(|error| error.to_string())
                    });
                }
            }
        }
    }

    // The element kernels, on the same argument: a decode step runs one of these
    // between every pair of GEMVs, and they are their own library with their own
    // `new()`.
    let hidden = zeroes(&device, IN_FEATURES * 2 + 4096);
    let norm_weight = zeroes(&device, IN_FEATURES * 2 + 4096);
    drive(&queue, |encoder| {
        elements
            .encode_rms_norm_residual(encoder, &hidden, &norm_weight, &hidden, &hidden, 1, IN_FEATURES, 1e-6)
            .map_err(|error| error.to_string())
    });

    let after_dispatch = pipeline_audit::build_count();
    assert_eq!(
        after_dispatch,
        after_build,
        "decode-shaped dispatches compiled {} pipeline(s) after load: {:?}. Any timed \
         loop reaching one of these is measuring Metal shader compilation, not the kernel.",
        after_dispatch - after_build,
        pipeline_audit::builds_since(after_build),
    );
}

/// Every kernel the families built is *named* in the log.
///
/// A count with no names cannot answer "which kernel", and "which kernel" is
/// what a reader needs when a late compile shows up. It also catches the
/// quieter regression where `record` stops being called at one site while the
/// total still moves, which the count-only assertion above would not see.
#[test]
fn every_built_pipeline_is_named() {
    let _serial = serial();
    if Device::system_default().is_none() {
        return;
    }
    let _ = LinearKernels::shared().expect("linear kernels");
    let _ = ElementKernels::shared().expect("element kernels");
    let names = pipeline_audit::built_names();
    assert!(
        names.len() > 100,
        "only {} distinct kernels were built; these families build well over a \
         hundred, so a small number means the log is not recording them",
        names.len()
    );
    for expected in [
        // The live decode matvec, under both names it can be reached by: the
        // 3-word slot and the 2-word default.
        "linear_gemv_ishizuki_k3",
        "linear_gemv_ishizuki_k3_w2",
        // `lm_head` at 6 bpw, and the unaligned EXL3 arm it actually falls to
        // (its trellis starts at `packed_offset % 4 == 2`).
        "linear_gemv_ishizuki_k6",
        "linear_gemv_exl3_k6_u_mcg",
        // The finalize and the input transform that bracket every decode GEMV.
        "linear_finalize_simd",
        "linear_input_transform",
        // An element kernel, so the assertion is not only about the linear
        // family -- a log that recorded only one library would still pass the
        // names above.
        "rms_norm_residual_tg",
    ] {
        assert!(
            names.iter().any(|name| name == expected),
            "{expected} was not built, so the audit cannot vouch for a pipeline it has \
             no record of. Built: {names:?}"
        );
    }
}

/// The attention family is covered too, and separately: it is a third library
/// with its own `new()`, its own `OnceLock`, and the only place split-KV decode
/// dispatches from.
///
/// Held in its own test because a test binary's `OnceLock`s are one-shot: a
/// second `AttentionKernels::new()` in another test would compile a second copy
/// of the library and make the other test's counts meaningless.
#[test]
fn the_attention_family_builds_every_decode_specialisation() {
    let _serial = serial();
    let Some(device) = Device::system_default() else {
        return;
    };
    let before = pipeline_audit::build_count();
    let attention = AttentionKernels::new().expect("attention kernels");
    let built = pipeline_audit::build_count() - before;
    assert!(
        built > 0,
        "AttentionKernels::new() built no pipelines, so this file's claim that the \
         families construct eagerly does not hold for the attention family"
    );

    // Both model shapes, at one key each: the sliding layers are 32/16 heads at
    // head_dim 256 and the full ones 32/4 at 512, so the split-KV decode kernel
    // has two distinct `(head_dim, nsg)` specialisations and a decode step uses
    // both.
    let queries = zeroes(&device, 32 * 512 * 2 + 4096);
    let keys = zeroes(&device, 16 * 512 * 2 + 4096);
    let values = zeroes(&device, 16 * 512 * 2 + 4096);
    let query_positions = zeroes(&device, 4096);
    let key_positions = zeroes(&device, 4096);
    let output = zeroes(&device, 32 * 512 * 2 + 4096);
    let partials = zeroes(&device, 16 * 32 * (512 + 2) * 4 + 4096);
    let queue = device.new_command_queue();

    for (q_heads, kv_heads, head_dim) in [(32usize, 16usize, 256usize), (32, 4, 512)] {
        let request = kaplumba_kernels::attention::AttentionEncodeRequest {
            queries: &queries,
            keys: &keys,
            values: &values,
            query_positions: &query_positions,
            key_positions: &key_positions,
            output: &output,
            queries_count: 1,
            q_heads,
            kv_heads,
            head_dim,
            key_count: 1,
            scale: 1.0,
            window: 1024,
        };
        drive(&queue, |encoder| {
            attention
                .encode_online_attention_split(encoder, request, &partials)
                .map_err(|error| error.to_string())
        });
    }

    assert_eq!(
        pipeline_audit::build_count() - before,
        built,
        "an attention dispatch compiled a pipeline after construction: {:?}",
        pipeline_audit::builds_since(before)
    );
}
