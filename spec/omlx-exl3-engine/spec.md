# Spec: omlx-based single-user Gemma 4 31B engine (EXL3 3.0 bpw)

> **SUPERSEDED (2026-09-22).** This Python/omlx plan is abandoned per user directive: **no Python anywhere in the project** — engine must be pure Rust with raw Metal kernels, not a Python wrapper.
>
> **Replacement:** [`spec/rust-metal-engine/`](../rust-metal-engine/spec.md) — same goals (single-user Gemma 4 31B, EXL3 3.0 bpw, TTFT→tg→pp→mem priority, no quality loss, no KV quant), new architecture.
>
> **Companion:** [`spec/paged-kv-cache/`](../paged-kv-cache/spec.md) is also being rewritten for the Rust engine (omlx touchpoints removed).
>
> Historical content below the line is retained only for migration reference; do not implement from this file.

---

## Superseded requirements (archive)

Original R1–R8, AC1–AC11, OQ1–OQ8 were closed in the 2026-09-22 interview and **carried forward** into `spec/rust-metal-engine/spec.md` (renumbered/adapted where omlx-specific: single-tenant gates, `mx.compile`, omlx regression suite). See that file for the live requirements.

Key facts migrated (still true):
- Hardware: M4 Pro, 20 GPU cores, 273 GB/s, 24 GB unified.
- Memory model: EXL3 ~15.5 GB weights + ~1 GB SWA/KV ≈ 16.5 GB resident; roofline ~17.6 tok/s.
- Targets: ~200 t/s prefill, ~18 t/s generation; resident stretch goal 14.6 GB.
- Reference SHAs frozen in `spec/rust-metal-engine/implementation_notes.md`.
- Environment pins (T0.3) migrated there as well.
