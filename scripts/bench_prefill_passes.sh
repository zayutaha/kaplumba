#!/usr/bin/env bash
# bench_prefill_passes.sh -- interleaved A/B of the OQ9 prefill key-range passes.
#
# usage: ROUNDS=3 scripts/bench_prefill_passes.sh <keys> [pass_keys ...]
#
# Arm A is always `pass_keys = 0` (one dispatch: the shipped body). Arms B are the
# requested pass sizes. Arms are interleaved per round (A B C / A B C / ...), each
# appending to its OWN log as it finishes -- never piped into `tail`, which buffers
# until EOF and cost a ~50-minute sweep here.
#
# Memory is polled DURING each arm and the sample is classified afterwards: an
# invalid sample records NO number at all, because the OQ5 distortion is ~99x and
# always pessimistic, and a sweep that only emits pessimistic samples reads as
# "passes are much worse".
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpuq_mem.sh
. ./scripts/gpuq_mem.sh

M=${KAPLUMBA_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
KAP=${KAPLUMBA_BIN:-./target/release/kaplumba}
GPUQ=${GPUQ_BIN:-scripts/gpuq}
MIN_UNUSED_GIB=${MIN_UNUSED_GIB:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1}
POLL_SECS=${POLL_SECS:-5}
ROUNDS=${ROUNDS:-3}
OUTDIR=${OUTDIR:-/tmp/bench_passes}

KEYS=${1:-16384}
shift || true
[ "$#" -gt 0 ] && PASSES="$*" || PASSES=${PASSES:-"4096 8192 16384"}

# "Explain the theory of relativity. " is 6 tokens, so KEYS/6 repetitions lands on
# the token count this spec uses everywhere else (6.01 tokens/occurrence).
PROMPT=$(printf 'Explain the theory of relativity. %.0s' $(seq 1 "$(( KEYS / 6 ))"))

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.tsv"
: > "$SUMMARY"

run_arm() { # <label> <pass_keys> <round>
  local label=$1 pass=$2 round=$3
  local log="$OUTDIR/${label}_r${round}.log"
  local poll; poll=$(mktemp)
  local avail
  avail=$("$GPUQ" status 2>/dev/null | gpuq_mem_available_gib)
  if ! "$GPUQ" status 2>/dev/null | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then
    printf 'r%s %s %s SKIPPED box-unhealthy available=%s need=%s\n' \
      "$round" "$label" "$pass" "${avail:-UNPARSEABLE}" "$MIN_UNUSED_GIB" >> "$SUMMARY"
    rm -f "$poll"; return 0
  fi
  ( while :; do mem_poll_record "$poll"; sleep "$POLL_SECS"; done ) &
  local poll_pid=$!
  local -a env_args=("KAPLUMBA_PREFILL_PASS_KEYS=$pass")
  "${env_args[@]}" "$GPUQ" try -- /usr/bin/time -l "$KAP" \
    --model "$M" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0 --metrics > "$log" 2>&1
  local rc=$?
  kill "$poll_pid" 2>/dev/null; wait "$poll_pid" 2>/dev/null

  if [ "$rc" = "75" ]; then
    printf 'r%s %s %s GPU-BUSY\n' "$round" "$label" "$pass" >> "$SUMMARY"
    rm -f "$poll"; return 0
  fi
  if [ "$rc" != "0" ]; then
    printf 'r%s %s %s FAILED rc=%s\n' "$round" "$label" "$pass" "$rc" >> "$SUMMARY"
    sed -n '1,4p' "$log" >> "$SUMMARY"
    rm -f "$poll"; return 0
  fi

  local verdict ok=1 toks pre dec foot
  verdict=$(mem_sample_verdict "$poll") || ok=0
  toks=$(grep -o 'prompt_tokens=[0-9]*' "$log" | head -1)
  pre=$(grep -o 'prefill_tok_s=[0-9.]*' "$log" | head -1)
  dec=$(grep -o 'decode_tok_s=[0-9.]*' "$log" | head -1)
  foot=$(awk -F'[ \t\t]+' '/peak memory footprint/ { printf "%.2f", $NF / 1073741824; exit }' "$log")
  if [ "$ok" = "1" ]; then
    printf 'r%s %s %s %s %s %s footprint=%sGiB VALID\n' \
      "$round" "$label" "$pass" "$toks" "$pre" "$dec" "${foot:-?}" >> "$SUMMARY"
  else
    printf 'r%s %s %s %s INVALID-NO-NUMBER footprint=%sGiB %s\n' \
      "$round" "$label" "$pass" "$toks" "${foot:-?}" "$verdict" >> "$SUMMARY"
  fi
  rm -f "$poll"
}

{
  echo "# OQ9 prefill key-range pass A/B"
  echo "# keys=$KEYS prompt_tokens~$KEYS rounds=$ROUNDS passes=[$PASSES]"
  echo "# columns: round label pass_keys prompt_tokens prefill_tok_s decode_tok_s"
} >> "$SUMMARY"

for r in $(seq 1 "$ROUNDS"); do
  run_arm onedispatch 0 "$r"
  for p in $PASSES; do run_arm "pass${p}" "$p" "$r"; done
done
echo "SWEEP-DONE" >> "$SUMMARY"

cat "$SUMMARY"
echo "--- medians over VALID samples ---"
grep VALID "$SUMMARY" | awk -v OFS='\t' '
  {
    label = $2; pk = $3;
    for (i = 1; i <= NF; i++) if ($i ~ /^prefill_tok_s=/) { split($i, kv, "="); v = kv[2] + 0 }
    if (v > 0) { n[label]++; s[label] += v; if (v > mx[label]) mx[label] = v
                 if (mn[label] == "" || v < mn[label]) mn[label] = v }
  }
  END { for (k in n) printf "%-14s n=%d mean=%.2f min=%.2f max=%.2f\n", k, n[k], s[k]/n[k], mn[k], mx[k] }
' | sort