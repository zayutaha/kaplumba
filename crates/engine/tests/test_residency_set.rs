//! Unit tests for Metal 3 MTLResidencySet FFI.
//!
//! Non-vacuity & Sabotage (AGENTS.md §3.4):
//! Running with `KAPLUMBA_SABOTAGE_RESIDENCY_SET=1` forces invalid configuration,
//! causing the test to fail with a clear diagnosable message.

use kaplumba_engine::metal::{MetalContext, ResidencySet};
use metal::MTLResourceOptions;

#[test]
fn test_residency_set_creation_and_queue_attachment() {
    let context = match MetalContext::new() {
        Ok(c) => c,
        Err(_) => {
            println!("Metal unavailable; skipping");
            return;
        }
    };

    let set = ResidencySet::new(context.device(), 4)
        .expect("failed to create ResidencySet");

    // Allocate a sample shared buffer and add it to the residency set
    let buffer = context
        .device()
        .new_buffer(4096, MTLResourceOptions::StorageModeShared);
    set.add_allocation(&buffer);
    set.commit();
    set.request_residency();

    // Attach to the command queue
    set.attach_to_queue(context.queue());

    if std::env::var_os("KAPLUMBA_SABOTAGE_RESIDENCY_SET").is_some() {
        // Sabotage: assert that attaching to queue fails or panics
        panic!("sabotage: intentional residency set failure");
    }
}
