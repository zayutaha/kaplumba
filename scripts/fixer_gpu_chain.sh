#!/usr/bin/env bash
# fixer_gpu_chain.sh -- ONE script for the remaining GPU work of the
# spec/prefill-24k-90tps fixer pass, so every arm of every finding is produced in
# a single session on an uncontended box and nothing is left half-run.
#
# Run it with nothing else on the GPU. It appends per arm and never pipes a long
# measurement into `tail` (AGENTS.md / NEXT.md 6).
set -uo pipefail
cd "$(dirname "$0")/.."
M=${KAPLUMBA_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
KAP=${KAPLUMBA_BIN:-./target/release/kaplumba}
GPUQ=${GPUQ_BIN:-scripts/gpuq}
E=spec/prefill-24k-90tps/fixer_evidence
mkdir -p "$E/f3_gap" "$E/f5_bq" "$E/f3_control"

prompt() { printf 'Explain the theory of relativity. %.0s' $(seq 1 "$(( $1 / 6 ))"); }

# ---------------------------------------------------------------- F4 + gate ----
# `bench_split_rule_quality.sh` covers F4 (does the workgroups rule change greedy
# tokens at a context where it CAN?) and the quality gate's >=512-token
# temperature-0 generations, and its `passes0` arm is F3's bit-exactness check on
# real generated text.
echo "=== [1/5] F4 quality sweep + temperature-0 gate ==="
OUT="$E/f4_quality" CONTEXTS="1024 4096 16394" MAXNEW=512 \
  scripts/bench_split_rule_quality.sh > "$E/f4_quality.log" 2>&1
echo "  rc=$? ; $(tail -1 "$E/f4_quality.log")"

echo "=== [2/5] F4/F5 quality gate at a sampling temperature (fixed seed) ==="
OUT="$E/f4_quality_sampled" CONTEXTS="16394" MAXNEW=512 TEMPERATURE=0.7 SEED=20260905 \
  scripts/bench_split_rule_quality.sh > "$E/f4_quality_sampled.log" 2>&1
echo "  rc=$? ; $(tail -1 "$E/f4_quality_sampled.log")"

# ------------------------------------------------------------- F3 the gap ----
# F3 asks what the shipped default (OQ9 passes ON at key_count > 8192) costs.
# The wall A/B is `f3_default/`; this is the per-kernel view, which is the only
# place the extra dispatches are visible at all.
for arm in defaultpasses onedispatch; do
  echo "=== [3/5] F3 profile arm=$arm ==="
  if [ "$arm" = onedispatch ]; then export KAPLUMBA_PREFILL_PASS_KEYS=0; else unset KAPLUMBA_PREFILL_PASS_KEYS; fi
  ( while :; do date +%s; sleep 5; done ) > "$E/f3_gap/$arm.polls" 2>&1 &
  pid=$!
  KAPLUMBA_GPU_PROFILE=1 "$GPUQ" try -- /usr/bin/time -l "$KAP" --model "$M" \
    --prompt "$(prompt 16384)" --max-new-tokens 1 --temperature 0 --metrics \
    > "$E/f3_gap/$arm.log" 2>&1
  rc=$?
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "  rc=$rc  $(grep -o 'prefill_tok_s=[0-9.]*' "$E/f3_gap/$arm.log" | head -1)"
done
unset KAPLUMBA_PREFILL_PASS_KEYS

# ------------------------------------------------- positive control for F3 ----
# A "no measurable difference" is only believable after something moved the same
# instrument in the same build. `KAPLUMBA_SCALAR_ATTENTION=1` replaces the flash
# body with the scalar one and is documented to move prefill by ~20%.
echo "=== [4/5] F3 positive control (scalar attention vs shipped) ==="
for arm in shipped scalar; do
  if [ "$arm" = scalar ]; then export KAPLUMBA_SCALAR_ATTENTION=1; else unset KAPLUMBA_SCALAR_ATTENTION; fi
  "$GPUQ" try -- "$KAP" --model "$M" --prompt "$(prompt 16384)" \
    --max-new-tokens 1 --temperature 0 --metrics \
    > "$E/f3_control/$arm.log" 2>&1
  echo "  $arm rc=$?  $(grep -o 'prefill_tok_s=[0-9.]*' "$E/f3_control/$arm.log" | head -1)"
done
unset KAPLUMBA_SCALAR_ATTENTION

# --------------------------------------------------------- F5 the BQ lever ---
# `FLASH_QUERY_BLOCK_MIN_KEYS = 16384` was chosen from the byte model §12
# withdrew, and the ablation bench ranks `BQ=1` 5.8% FASTER than `BQ=2` at 24 576
# keys (which is why §12.2 flagged the threshold as suspect). One env var, so
# this costs nothing to test end to end at the context that matters.
echo "=== [5/5] F5 KAPLUMBA_FLASH_BQ at 24 590 ==="
for arm in bq2_default bq1_forced; do
  if [ "$arm" = bq1_forced ]; then export KAPLUMBA_FLASH_BQ=1; else unset KAPLUMBA_FLASH_BQ; fi
  ( while :; do date +%s; sleep 5; done ) > "$E/f5_bq/$arm.polls" 2>&1 &
  pid=$!
  "$GPUQ" try -- /usr/bin/time -l "$KAP" --model "$M" --prompt "$(prompt 24590)" \
    --max-new-tokens 1 --temperature 0 --metrics > "$E/f5_bq/$arm.log" 2>&1
  rc=$?
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "  $arm rc=$rc  $(grep -o 'prefill_tok_s=[0-9.]*' "$E/f5_bq/$arm.log" | head -1)  $(grep -o 'peak memory footprint' -A0 "$E/f5_bq/$arm.log" | head -1)"
  grep -oE '[0-9]+ peak memory footprint' "$E/f5_bq/$arm.log" | head -1
done
unset KAPLUMBA_FLASH_BQ
echo "FIXER-CHAIN-DONE"