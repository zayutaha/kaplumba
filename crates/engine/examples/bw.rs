//! What does this machine actually stream at?
//!
//! Every ceiling in this project has been computed from the 273 GB/s spec sheet.
//! This measures a plain strided read over a buffer far larger than any cache,
//! with nothing but an add in the loop, so the roofs stop being arithmetic.
use metal::{Device, MTLResourceOptions};
use std::time::Instant;
const SRC: &str = include_str!("../../kernels/src/bw.metal");
fn main() {
    let device = Device::system_default().expect("Metal device");
    let queue = device.new_command_queue();
    let lib = device
        .new_library_with_source(SRC, &metal::CompileOptions::new())
        .expect("compile");
    let st = device
        .new_compute_pipeline_state_with_function(&lib.get_function("stream_sum", None).unwrap())
        .unwrap();
    let threads = 20 * 1024u64;
    for gib in [2u64, 6] {
        let bytes = gib * 1_000_000_000u64;
        let words = (bytes / 4) as u32;
        let data = device.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        // TOUCH THE BUFFER BEFORE STREAMING IT. `new_buffer` on macOS hands back
        // untouched zero pages, and a read of never-written shared memory is not
        // a memory-system measurement: the pages are not resident, and the
        // kernel may be served a path that a real weight stream never takes.
        // This probe reported 55 GB/s while reading exactly this, and that
        // figure became the "achievable bandwidth" denominator quoted to every
        // other experiment in the project. A bandwidth ceiling measured over
        // pages nobody wrote is not a ceiling.
        {
            let ptr = data.contents() as *mut u8;
            // A byte pattern with no long runs, so a prefetcher cannot treat
            // the fill as a constant and the stream cannot compress.
            let mut x: u32 = 0x9e37_79b9;
            for i in (0..bytes as usize).step_by(4096) {
                unsafe { std::ptr::write_unaligned(ptr.add(i) as *mut u32, x) };
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            }
            // Fault every page in, not every 4096th byte: one touch per page is
            // what makes the whole page resident.
            for page in 0..(bytes / 4096) as usize {
                unsafe { std::ptr::write_volatile(ptr.add(page * 4096), 0) };
            }
        }
        let out = device.new_buffer(threads * 4, MTLResourceOptions::StorageModeShared);
        // Report MEDIAN as well as best. This probe reported min-of-passes, and
        // min-of-passes on a shared box is not a peak, it is the one pass that
        // happened to run while a competitor was idle: an agent watching this
        // binary saw the mean rise 24.8 -> 80.6 GB/s over 180 s "competitors
        // exiting", and best-swing 79.9 -> 230.5 while load *fell*. That
        // selection effect is the most likely origin of the project's 248.5 GB/s
        // figure, which sits above what NVIDIA's GB10 -- an identical
        // 256-bit LPDDR5X-8533 part -- measures at 229-230 GB/s.
        let mut times: Vec<f64> = Vec::new();
        for round in 0..10 {
            let cb = queue.new_command_buffer();
            let e = cb.new_compute_command_encoder();
            e.set_compute_pipeline_state(&st);
            e.set_buffer(0, Some(&data), 0);
            e.set_buffer(1, Some(&out), 0);
            e.set_bytes(2, 4, &words as *const _ as *const std::ffi::c_void);
            e.dispatch_threads(
                metal::MTLSize::new(threads, 1, 1),
                metal::MTLSize::new(256, 1, 1),
            );
            e.end_encoding();
            let t = Instant::now();
            cb.commit();
            cb.wait_until_completed();
            if round >= 3 {
                times.push(t.elapsed().as_secs_f64());
            }
        }
        times.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timings"));
        let best = *times.first().expect("at least one timed pass");
        let median = times[times.len() / 2];
        let worst = *times.last().expect("at least one timed pass");
        let gbps = bytes as f64 / best / 1e9;
        let med_gbps = bytes as f64 / median / 1e9;
        let worst_gbps = bytes as f64 / worst / 1e9;
        println!(
            "  stream {gib:>2} GB: med {med_gbps:>6.1} GB/s ({:>4.0}% of 273) -> {:>5.1} tok/s | \
             best {gbps:>6.1} | worst {worst_gbps:>6.1} | spread {:>4.1}x",
            med_gbps / 273.0 * 100.0,
            med_gbps / 12.06,
            best / worst
        );
    }
}
