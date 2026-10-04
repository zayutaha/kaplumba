#!/usr/bin/env bash
# bench_decode_splits.sh -- sweep KAPLUMBA_DECODE_SPLITS (or another env knob)
# end to end, one arm at a time, with a trustworthy memory gate.
#
# USAGE
#   LOG=/tmp/x.log scripts/bench_decode_splits.sh <keys> <split...>
#   LOG=/tmp/x.log VAR=KAPLUMBA_DECODE_FLASH ROPE="0 unset" \
#       scripts/bench_decode_splits.sh 8192 --          # -- means "leave unset"
#
# ENV
#   LOG                   append one line per arm here (default /tmp/bench_sweep.log)
#   VAR                   env knob to pin per arm   (default KAPLUMBA_DECODE_SPLITS)
#   ROPE                  space-separated arm values (default "$*")
#   ROUNDS                interleaved rounds, A B C A B C (default 1)
#   MIN_UNUSED_GIB        pre-arm gate threshold    (default 4)
#   MAX_NEW_TOKENS        decode tokens per arm     (default 24)
#   GPUQ_BIN              override scripts/gpuq (tests stub this)
#   KAPLUMBA_BIN          override the binary under test
#   KAPLUMBA_MODEL        override the model path
#   DRY_RUN=1             run the whole pipeline, substitute a stub for the binary
#
# WHY IT IS BUILT THIS WAY
#
# * One line per arm is APPENDED as each arm finishes. Never pipe this into
#   `tail`: tail buffers until EOF, so an interrupted run loses everything. That
#   cost a ~50-minute 24k sweep (implementation_notes.md 12.3).
# * Arms are INTERLEAVED by round (A B C / A B C / ...), never A A A B B B, so a
#   drift in box conditions cannot masquerade as an effect. This box has been
#   measured at 3.7 / 3.9 / 4.3 / 4.8 / 6.0 / 6.4 tok/s from the same binary
#   minutes apart (AGENTS.md 10.2 #5).
# * The memory gate checks the box BEFORE the arm AND samples it DURING the arm.
#   A pre-arm check alone is not enough, because the arm is what pushes a 24k
#   run into compression: its own phys_footprint is 18.35 GiB of a 24 GiB box.
#   Post-hoc an arm whose sample is INVALID is logged as INVALID and carries no
#   number, because the OQ5 distortion is 99x and one-sided (0.055 tok/s against
#   5.44) -- a sweep that only emits pessimistic samples reads as "more splits is
#   much worse" and would send the next person to delete query splitting.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpuq_mem.sh
. ./scripts/gpuq_mem.sh

M=${KAPLUMBA_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
GPUQ=${GPUQ_BIN:-scripts/gpuq}
KAP=${KAPLUMBA_BIN:-./target/release/kaplumba}
VAR=${VAR:-KAPLUMBA_DECODE_SPLITS}
MIN_UNUSED_GIB=${MIN_UNUSED_GIB:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-24}
POLL_SECS=${POLL_SECS:-3}
LOG=${LOG:-/tmp/bench_sweep.log}
DRY_RUN=${DRY_RUN:-0}

KEYS=${1:-24576}
shift || true
ROUNDS=${ROUNDS:-1}
[ "$#" -gt 0 ] && ROPE="$*" || ROPE=${ROPE:-"1 4 8 16"}

# The prompt sentence used throughout this spec, repeated to reach ~$KEYS tokens.
# "Explain the theory of relativity. " is 6 tokens, hence KEYS/6 repetitions.
PROMPT=$(printf 'Explain the theory of relativity. %.0s' $(seq 1 "$(( KEYS / 6 ))"))

: > "$LOG"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# One arm. Args: <rope value>. Appends its own lines to $LOG.
# Echoes nothing: everything goes to the log, so an interrupted run still has it.
run_arm() {
  local arm=$1 poll="$TMP/poll" out rc verdict
  : > "$poll"

  # Pre-arm gate: refuse to start on a box we cannot vouch for.
  if ! "$GPUQ" status 2>/dev/null | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then
    local avail
    avail=$("$GPUQ" status 2>/dev/null | gpuq_mem_available_gib)
    echo "$VAR=$arm SKIPPED box-unhealthy (available='${avail:-UNPARSEABLE}' GiB, need >=${MIN_UNUSED_GIB})" >> "$LOG"
    return 0
  fi

  # Poll DURING the arm. The arm is what tips the box into compression.
  (
    while :; do mem_poll_record "$poll"; sleep "$POLL_SECS"; done
  ) & local poll_pid=$!

  local rc_file="$TMP/rc"
  : > "$rc_file"
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY-RUN %s=%s keys=%s\n' "$VAR" "$arm" "$KEYS" >> "$LOG"
  else
    # `env VAR=...` rather than a prefix assignment so a value of "0" or an empty
    # value is still passed, and gpuq's exit 75 can be told apart from the
    # engine's. NO `tail` anywhere: capture, then grep.
    if [ "$arm" = "--" ]; then
      env -u "$VAR" "$GPUQ" try -- /usr/bin/time -l "$KAP" \
        --model "$M" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature 0 --metrics > "$TMP/out" 2>&1
    else
      env "$VAR=$arm" "$GPUQ" try -- /usr/bin/time -l "$KAP" \
        --model "$M" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature 0 --metrics > "$TMP/out" 2>&1
    fi
    rc=$?
    printf '%s\n' "$rc" > "$rc_file"
  fi
  rc=$(cat "$rc_file"); rc=${rc:-0}

  kill "$poll_pid" 2>/dev/null; wait "$poll_pid" 2>/dev/null

  if [ "$rc" = "75" ]; then
    echo "$VAR=$arm GPU-BUSY (exit 75; do other work, retry later)" >> "$LOG"
    return 0
  fi
  if [ "$DRY_RUN" = "1" ]; then
    verdict=$(mem_sample_verdict "$poll" 2>&1 || true)
    echo "    mem: $verdict" >> "$LOG"
    return 0
  fi
  if [ "$rc" != "0" ]; then
    echo "$VAR=$arm FAILED rc=$rc (see $TMP/out)" >> "$LOG"
    sed -n '1,5p' "$TMP/out" >> "$LOG"
    return 0
  fi

  verdict=$(mem_sample_verdict "$poll" 2>&1) && verdict_ok=1 || verdict_ok=0
  local pre dec post
  pre=$(grep -o 'prompt_tokens=[0-9]*' "$TMP/out" | head -1)
  dec=$(grep -o 'decode_tok_s=[0-9.]*' "$TMP/out" | head -1)
  post=$(grep -o 'prefill_tok_s=[0-9.]*' "$TMP/out" | head -1)
  local peak
  peak=$(awk -F'[ \t]+' '/peak memory footprint/ { printf "%.2f", $NF / 1073741824; exit }' "$TMP/out")
  local swap
  swap=$(grep -o 'swaps=[0-9]*' "$TMP/out" | head -1)

  if [ "$verdict_ok" = "1" ]; then
    echo "$VAR=$arm $pre $dec $post $swap footprint=${peak:-?}GiB mem=$verdict" >> "$LOG"
  else
    # The number is deliberately NOT recorded on an invalid sample: OQ5's
    # distortion is 99x and always pessimistic, so an invalid number is worse
    # than no number.
    echo "$VAR=$arm INVALID-NO-NUMBER $pre footprint=${peak:-?}GiB mem=$verdict" >> "$LOG"
  fi
}

echo "sweep $VAR rope='$ROPE' keys=$KEYS rounds=$ROUNDS prompt_tokens~$((KEYS)) log=$LOG" >> "$LOG"
for r in $(seq 1 "$ROUNDS"); do
  for arm in $ROPE; do
    echo "round=$r" >> "$LOG"
    run_arm "$arm"
  done
done
echo "SWEEP-DONE" >> "$LOG"