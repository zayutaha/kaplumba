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
  # The poll file is KEPT, not mktemp'd away: when a sample comes back INVALID the
  # first question is always "what did the gate actually see", and an empty
  # scratch file answers nothing. Cost is a few KB per arm.
  local poll="$OUTDIR/${label}_r${round}.polls"
  : > "$poll"
  local avail
  avail=$("$GPUQ" status 2>/dev/null | gpuq_mem_available_gib)
  if ! "$GPUQ" status 2>/dev/null | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then
    printf 'r%s %s %s SKIPPED box-unhealthy available=%s need=%s\n' \
      "$round" "$label" "$pass" "${avail:-UNPARSEABLE}" "$MIN_UNUSED_GIB" >> "$SUMMARY"
    return 0
  fi
  # A prefix assignment (not `env VAR=... cmd`) so a value of "0" still reaches
  # the binary: the off arm needs `KAPLUMBA_PREFILL_PASS_KEYS=0` set EXPLICITLY,
  # because "unset" and "0" both mean one dispatch but only `0` overrides a
  # default someone might later change.
  ( while :; do mem_poll_record "$poll"; sleep "$POLL_SECS"; done ) &
  local poll_pid=$!
  # The OQ5 marker: the load transient takes this box from ~17 GiB unused to ~2 GiB
  # in 8 seconds on its own (measured on a 626-token run that returned 100.6
  # tok/s), so polls before the load completes are excluded from the verdict.
  # `MEM_PHASE_LOAD_DONE` is read by `mem_sample_verdict`; removed up front so a
  # stale marker from a previous arm cannot mask this one.
  local marker="$OUTDIR/.loaded.$$"
  rm -f "$marker"
  # A watcher that touches the marker the instant the engine reports its load
  # complete ("loaded in NNNN ms", line 2 of the run's output). Using the
  # engine's own line rather than a sleep means the marker is correct on both a
  # 15-second 614-token run and a 7-minute 24k one, and it cannot drift if the
  # load gets slower.
  (
    local waited=0
    while [ "$waited" -lt 1800 ]; do
      if grep -q '^loaded in ' "$log" 2>/dev/null; then : > "$marker"; break; fi
      sleep 0.5; waited=$(( waited + 1 ))
    done
  ) &
  local watch_pid=$!

  KAPLUMBA_PREFILL_PASS_KEYS="$pass" "$GPUQ" try -- /usr/bin/time -l "$KAP" \
    --model "$M" --prompt "$PROMPT" --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature 0 --metrics > "$log" 2>&1
  local rc=$?
  kill "$poll_pid" "$watch_pid" 2>/dev/null; wait "$poll_pid" 2>/dev/null

  if [ "$rc" = "75" ]; then
    printf 'r%s %s %s GPU-BUSY\n' "$round" "$label" "$pass" >> "$SUMMARY"
    rm -f "$marker"; return 0
  fi
  if [ "$rc" != "0" ]; then
    printf 'r%s %s %s FAILED rc=%s\n' "$round" "$label" "$pass" "$rc" >> "$SUMMARY"
    sed -n '1,4p' "$log" >> "$SUMMARY"
    rm -f "$marker"; return 0
  fi

  local verdict cpuverdict ok=1 toks pre dec foot
  # `MEM_PHASE_LOAD_DONE` restricts the verdict to polls taken after the engine's
  # own load completed. Exported per-call rather than left global so two harnesses
  # cannot interfere, and it is unset immediately afterwards.
  verdict=$(MEM_PHASE_LOAD_DONE="$marker" mem_sample_verdict "$poll") || ok=0
  # Keep the marker mtime in the log line: an INVALID verdict with
  # `load-skipped=0` means the gate watched the whole run including the load
  # transient, which is a different failure from "the compute itself was
  # compressed". Distinguishing them is the point of keeping the polls.
  local mark_note="marker=$(stat -f %m "$marker" 2>/dev/null || echo none)"
  rm -f "$marker"
  # OQ5's CPU rule, reported but NOT gated on. Measured on a healthy 626-token run:
  # 4.12 s of `sys` over 14.40 s wall = 28.6%, and 9.32/227.91 = 4.1% on a healthy
  # 16 394-token run that returned 74.6 tok/s. Both are good runs, so a 2% limit
  # rejects them -- `sys` on this box is dominated by the 11.5 GiB mmap'd weight
  # read (file-backed page faults), not by compression. Recorded because OQ5 asks
  # for it, with the number that says it does not discriminate here.
  cpuverdict=$(mem_cpu_compression_ok "$log")
  toks=$(grep -o 'prompt_tokens=[0-9]*' "$log" | head -1)
  pre=$(grep -o 'prefill_tok_s=[0-9.]*' "$log" | head -1)
  dec=$(grep -o 'decode_tok_s=[0-9.]*' "$log" | head -1)
  # `/usr/bin/time -l` prints "  <bytes>  peak memory footprint": the value is
  # field 1 and the LABEL is field 2, so matching on the label and taking $NF
  # reads the label as a number. Take $1 -- and per AGENTS.md 10.3 this is
  # `phys_footprint`, never `maximum resident set size`, which misses Metal pages.
  foot=$(awk '/peak memory footprint/ { printf "%.2f", $1 / 1073741824; exit }' "$log")
  rss=$(awk '/maximum resident set size/ { printf "%.2f", $1 / 1073741824; exit }' "$log")
  # `rss` is recorded ONLY as a contrast, never as the footprint: AGENTS.md 10.3
  # records it understating real usage by 5.5 GiB because it misses Metal pages.
  if [ "$ok" = "1" ]; then
    printf 'r%s %s %s %s %s %s footprint=%sGiB rss=%sGiB VALID mem=[%s] cpu=[%s] %s\n' \
      "$round" "$label" "$pass" "$toks" "$pre" "$dec" "${foot:-?}" "${rss:-?}" \
      "$verdict" "$cpuverdict" "$mark_note" >> "$SUMMARY"
  else
    printf 'r%s %s %s %s INVALID-NO-NUMBER footprint=%sGiB rss=%sGiB mem=[%s] cpu=[%s] %s\n' \
      "$round" "$label" "$pass" "$toks" "${foot:-?}" "${rss:-?}" \
      "$verdict" "$cpuverdict" "$mark_note" >> "$SUMMARY"
  fi
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
# `grep VALID` also matches `INVALID-NO-NUMBER`, which folds discarded samples into
# the medians -- found by noticing an INVALID arm appear in the median table with a
# number on it. Match the whole word instead.
grep -E ' VALID( |$)' "$SUMMARY" | awk -v OFS='\t' '
  {
    label = $2; pk = $3;
    for (i = 1; i <= NF; i++) if ($i ~ /^prefill_tok_s=/) { split($i, kv, "="); v = kv[2] + 0 }
    if (v > 0) { n[label]++; s[label] += v; if (v > mx[label]) mx[label] = v
                 if (mn[label] == "" || v < mn[label]) mn[label] = v }
  }
  END { for (k in n) printf "%-14s n=%d mean=%.2f min=%.2f max=%.2f\n", k, n[k], s[k]/n[k], mn[k], mx[k] }
' | sort