#!/usr/bin/env bash
# bench_decode_splits.sh -- N4: sweep `KAPLUMBA_DECODE_SPLITS` with the MMA
# decode kernel, interleaved, with the OQ5 gate.
#
# usage: ROUNDS=3 OUTDIR=/tmp/n4 scripts/bench_decode_splits.sh <keys> [splits ...]
#        " --" in the split list means "leave KAPLUMBA_DECODE_SPLITS unset"
#
# Why the cap matters. `ATTENTION_SPLITS_MAX = 16` has never been probed above
# (implementation_notes.md 12.3), and the parallelism trade changed completely when
# T2b shipped the MMA kernel: the scalar kernel was latency-bound on a 64-workgroup
# grid, the MMA one may want MORE. So the sweep includes values ABOVE the cap and
# records what the engine actually did with them -- `decode_splits` clamps, so an
# arm above the cap must be reported as clamped, not as a distinct measurement.
#
# The engine prints its effective split count only indirectly, so this also records
# the partials-buffer requirement: `splits * q_heads * (head_dim + 2)` f32, which is
# what the cap exists to bound.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpuq_mem.sh
. ./scripts/gpuq_mem.sh

M=${KAPLUMBA_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
KAP=${KAPLUMBA_BIN:-./target/release/kaplumba}
GPUQ=${GPUQ_BIN:-scripts/gpuq}
MIN_UNUSED_GIB=${MIN_UNUSED_GIB:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-32}
POLL_SECS=${POLL_SECS:-5}
ROUNDS=${ROUNDS:-3}
OUTDIR=${OUTDIR:-/tmp/bench_splits}
# The shipped cap, mirrored here so the log can label clamped arms. Read from the
# binary's own log rather than hardcoded if the engine ever prints it.
CAP=${ATTENTION_SPLITS_MAX:-16}

KEYS=${1:-8192}
shift || true
[ "$#" -gt 0 ] && SPLITS="$*" || SPLITS=${SPLITS:-"1 2 4 8 16 24 32"}

PROMPT=$(printf 'Explain the theory of relativity. %.0s' $(seq 1 "$(( KEYS / 6 ))"))

mkdir -p "$OUTDIR"
# `LOG` is accepted as an alias for the summary path, because that is the name the
# other harnesses and the tests use. Without it a caller passing `LOG=` silently
# gets the summary in OUTDIR and nothing where it asked -- which is exactly how
# this harness's own positive control ended up reading an absent file.
SUMMARY=${LOG:-$OUTDIR/summary.tsv}
: > "$SUMMARY"

{
  echo "# N4 decode split sweep (MMA kernel), keys=$KEYS rounds=$ROUNDS splits=[$SPLITS]"
  echo "# cap=$CAP; arms above the cap are expected to CLAMP and are labelled as such"
  echo "# columns: round label requested_splits prompt_tokens decode_tok_s footprint"
} >> "$SUMMARY"

run_arm() { # <label> <requested splits, or --> <round>
  local label=$1 want=$2 round=$3
  local log="$OUTDIR/${label}_r${round}.log"
  local poll="$OUTDIR/${label}_r${round}.polls"
  : > "$poll"
  local avail
  avail=$("$GPUQ" status 2>/dev/null | gpuq_mem_available_gib)
  if ! "$GPUQ" status 2>/dev/null | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then
    printf 'r%s %s %s SKIPPED box-unhealthy available=%s need=%s\n' \
      "$round" "$label" "$want" "${avail:-UNPARSEABLE}" "$MIN_UNUSED_GIB" >> "$SUMMARY"
    return 0
  fi
  ( while :; do mem_poll_record "$poll"; sleep "$POLL_SECS"; done ) &
  local poll_pid=$!
  local marker="$OUTDIR/.loaded.$$"
  rm -f "$marker"
  ( local waited=0
    while [ "$waited" -lt 1800 ]; do
      grep -q '^loaded in ' "$log" 2>/dev/null && { : > "$marker"; break; }
      sleep 0.5; waited=$(( waited + 1 ))
    done ) &
  local watch_pid=$!

  if [ "$want" = "--" ]; then
    env -u KAPLUMBA_DECODE_SPLITS "$GPUQ" try -- /usr/bin/time -l "$KAP" \
      --model "$M" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature 0 --metrics > "$log" 2>&1
  else
    KAPLUMBA_DECODE_SPLITS="$want" "$GPUQ" try -- /usr/bin/time -l "$KAP" \
      --model "$M" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature 0 --metrics > "$log" 2>&1
  fi
  local rc=$?
  kill "$poll_pid" "$watch_pid" 2>/dev/null; wait "$poll_pid" 2>/dev/null
    if [ "$rc" = "75" ]; then
    printf 'r%s %s %s GPU-BUSY\n' "$round" "$label" "$want" >> "$SUMMARY"
    rm -f "$marker"; return 0
  fi
  if [ "$rc" != "0" ]; then
    printf 'r%s %s %s FAILED rc=%s\n' "$round" "$label" "$want" "$rc" >> "$SUMMARY"
    sed -n '1,4p' "$log" >> "$SUMMARY"
    rm -f "$marker"; return 0
  fi

  local verdict cpuverdict ok=1 toks dec foot
  verdict=$(MEM_PHASE_LOAD_DONE="$marker" mem_sample_verdict "$poll") || ok=0
  cpuverdict=$(mem_cpu_compression_ok "$log")
  rm -f "$marker"
  toks=$(grep -o 'prompt_tokens=[0-9]*' "$log" | head -1)
  dec=$(grep -o 'decode_tok_s=[0-9.]*' "$log" | head -1)
  foot=$(awk '/peak memory footprint/ { printf "%.2f", $1 / 1073741824; exit }' "$log")
  # An arm above the cap that produced the SAME number as the cap arm is evidence
  # of clamping, not of a speedup. Reported so the reader never has to guess.
  local clamped=""
  if [ "$want" != "--" ] && [ "$want" -gt "$CAP" ] 2>/dev/null; then clamped="above-cap($CAP)"; fi
  if [ "$ok" = "1" ]; then
    printf 'r%s %s %s %s %s footprint=%sGiB VALID %s mem=[%s] cpu=[%s]\n' \
      "$round" "$label" "$want" "$toks" "$dec" "${foot:-?}" "$clamped" \
      "$verdict" "$cpuverdict" >> "$SUMMARY"
  else
    printf 'r%s %s %s %s INVALID-NO-NUMBER footprint=%sGiB %s mem=[%s] cpu=[%s]\n' \
      "$round" "$label" "$want" "$toks" "${foot:-?}" "$clamped" \
      "$verdict" "$cpuverdict" >> "$SUMMARY"
  fi
}

for r in $(seq 1 "$ROUNDS"); do
  run_arm shipped -- "$r"
  for s in $SPLITS; do run_arm "splits$s" "$s" "$r"; done
done
echo "SWEEP-DONE" >> "$SUMMARY"

grep -v '^#' "$SUMMARY"
echo "--- decode medians over VALID samples ---"
# `grep VALID` also matches `INVALID-NO-NUMBER`, which silently folded discarded
# samples into the medians -- caught because an INVALID arm appeared in the output
# with a number. Match the whole word instead.
grep -E ' VALID( |$)' "$SUMMARY" | awk '
  { label = $2; for (i = 1; i <= NF; i++) if ($i ~ /^decode_tok_s=/) { split($i, kv, "="); v = kv[2] + 0 }
    if (v > 0) { n[label]++; s[label] += v; if (v > mx[label]) mx[label] = v
                 if (mn[label] == 0 || v < mn[label]) mn[label] = v } }
  END { for (k in n) printf "%-12s n=%d mean=%.3f min=%.3f max=%.3f\n", k, n[k], s[k]/n[k], mn[k], mx[k] }
' | sort