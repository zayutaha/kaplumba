#!/usr/bin/env bash
# scorecard.sh -- N6: the final interleaved A/B scorecard.
#
# usage: ROUNDS=3 OUTDIR=/tmp/n6 scripts/scorecard.sh
#
# Two arms, interleaved per round, at every context in the spec:
#
#   shipped     no env overrides -- what a user gets
#   workgroups  KAPLUMBA_DECODE_SPLIT_RULE=workgroups -- the one measured
#               improvement N4 found (decode splits 16 -> 13, +0.93% @8k and
#               +1.97% @16k), shipped as an opt-in so the scorecard can carry it
#               without flipping a default
#
# Both arms run every context, so any regression the opt-in causes at a short
# context is visible in the same table rather than assumed absent.
#
# Prefill and decode are measured in the same run: one model load serves both, the
# prompt length sets the context, and `--max-new-tokens` sets the decode budget.
# Peak `phys_footprint` comes from `/usr/bin/time -l` and NEVER from
# `ru_maxrss` (AGENTS.md 10.3: it misses Metal pages and understated real usage by
# 5.5 GiB); the rss figure is recorded beside it only as a contrast.
#
# Memory is polled during every run and the sample is classified afterwards; an
# arm whose sample is invalid records NO number at all.
set -uo pipefail
cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpuq_mem.sh
. ./scripts/gpuq_mem.sh

M=${KAPLUMBA_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
KAP=${KAPLUMBA_BIN:-./target/release/kaplumba}
GPUQ=${GPUQ_BIN:-scripts/gpuq}
MIN_UNUSED_GIB=${MIN_UNUSED_GIB:-4}
POLL_SECS=${POLL_SECS:-5}
ROUNDS=${ROUNDS:-3}
OUTDIR=${OUTDIR:-/tmp/scorecard}
MAXNEW=${MAXNEW:-32}

# tokens:decode-tokens per arm. 24k decode is deliberately absent: OQ8, three
# invalid samples, and the brief's instruction not to spend budget chasing it.
CONTEXTS=${CONTEXTS:-"614:8 8192:32 16384:32"}

mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/scorecard.tsv"
: > "$SUMMARY"

run_one() { # <label> <keys> <max_new> <round> <env...>
  local label=$1 keys=$2 maxnew=$3 round=$4; shift 4
  local rep=$(printf 'Explain the theory of relativity. %.0s' $(seq 1 "$(( keys / 6 ))"))
  local log="$OUTDIR/${label}_${keys}_r${round}.log"
  local poll="$OUTDIR/${label}_${keys}_r${round}.polls"
  : > "$poll"
  local avail
  avail=$("$GPUQ" status 2>/dev/null | gpuq_mem_available_gib)
  if ! "$GPUQ" status 2>/dev/null | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then
    printf '%s %s r%s SKIPPED box-unhealthy available=%s need=%s\n' \
      "$label" "$keys" "$round" "${avail:-UNPARSEABLE}" "$MIN_UNUSED_GIB" >> "$SUMMARY"
    return 0
  fi
  ( while :; do mem_poll_record "$poll"; sleep "$POLL_SECS"; done ) &
  local poll_pid=$!
  local marker="$OUTDIR/.loaded.$$"
  rm -f "$marker"
  ( local w=0
    while [ "$w" -lt 1800 ]; do
      grep -q '^loaded in ' "$log" 2>/dev/null && { : > "$marker"; break; }
      sleep 0.5; w=$(( w + 1 ))
    done ) &
  local watch_pid=$!

  # `env` with the arm's overrides first, so an arm with none is the shipped
  # configuration and an arm with one is the opt-in -- not a different binary.
  env "$@" "$GPUQ" try -- /usr/bin/time -l "$KAP" \
    --model "$M" --prompt "$rep" --max-new-tokens "$maxnew" \
    --temperature 0 --metrics > "$log" 2>&1
  local rc=$?
  kill "$poll_pid" "$watch_pid" 2>/dev/null; wait "$poll_pid" 2>/dev/null

  if [ "$rc" = "75" ]; then
    printf '%s %s r%s GPU-BUSY\n' "$label" "$keys" "$round" >> "$SUMMARY"; rm -f "$marker"; return 0
  fi
  if [ "$rc" != "0" ]; then
    printf '%s %s r%s FAILED rc=%s\n' "$label" "$keys" "$round" "$rc" >> "$SUMMARY"
    sed -n '1,4p' "$log" >> "$SUMMARY"; rm -f "$marker"; return 0
  fi

  local verdict ok=1 toks pre dec foot rss swaps
  verdict=$(MEM_PHASE_LOAD_DONE="$marker" mem_sample_verdict "$poll") || ok=0
  rm -f "$marker"
  toks=$(grep -o 'prompt_tokens=[0-9]*' "$log" | head -1)
  pre=$(grep -o 'prefill_tok_s=[0-9.]*' "$log" | head -1)
  dec=$(grep -o 'decode_tok_s=[0-9.]*' "$log" | head -1)
  # AC4 asks for `swaps 0`. The engine does NOT emit a `swaps` metric -- verified
  # by listing every key the run prints -- so this is read from the box instead:
  # `vm_stat`'s cumulative Swapouts counter, which is what a swap would move.
  # Sampled after the run because a swap that happened and finished leaves the
  # counter raised but the box idle. Reported as `swapouts=` so it cannot be
  # confused with a metric the engine produced.
  swaps="swapouts=$(vm_stat 2>/dev/null | awk -F: '/^Swapouts:/ { gsub(/[.]/,"",$NF); print $NF }')"
  foot=$(awk '/peak memory footprint/ { printf "%.2f", $1 / 1073741824; exit }' "$log")
  rss=$(awk '/maximum resident set size/ { printf "%.2f", $1 / 1073741824; exit }' "$log")
  if [ "$ok" = "1" ]; then
    printf '%s %s r%s %s %s %s %s footprint=%s rss=%s VALID\n' \
      "$label" "$keys" "$round" "$toks" "$pre" "$dec" "$swaps" \
      "${foot:-?}" "${rss:-?}" >> "$SUMMARY"
  else
    printf '%s %s r%s %s INVALID-NO-NUMBER footprint=%s rss=%s %s mem=[%s]\n' \
      "$label" "$keys" "$round" "$toks" "${foot:-?}" "${rss:-?}" "$swaps" \
      "$verdict" >> "$SUMMARY"
  fi
}

{
  echo "# N6 scorecard: rounds=$ROUNDS maxnew=$MAXNEW contexts=[$CONTEXTS]"
  echo "# arms: shipped (no overrides) vs workgroups (KAPLUMBA_DECODE_SPLIT_RULE=workgroups)"
  echo "# columns: arm keys round prompt_tokens prefill_tok_s decode_tok_s swaps footprint rss"
} >> "$SUMMARY"

for r in $(seq 1 "$ROUNDS"); do
  for spec in $CONTEXTS; do
    local_keys=${spec%%:*}
    local_new=${spec##*:}
    run_one shipped     "$local_keys" "$local_new" "$r"
    run_one workgroups  "$local_keys" "$local_new" "$r" KAPLUMBA_DECODE_SPLIT_RULE=workgroups
  done
done
echo "SWEEP-DONE" >> "$SUMMARY"

echo "=== raw ==="
grep -v '^#' "$SUMMARY"
echo
echo "=== per (arm, context) over VALID samples ==="
grep -E ' VALID( |$)' "$SUMMARY" | awk '
  { arm = $1; ctx = $2
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^prefill_tok_s=/) { split($i, kv2, "="); p = k[2] + 0 }
      if ($i ~ /^decode_tok_s=/)  { split($i, kv2, "="); d = k[2] + 0 }
      if ($i ~ /^footprint=/)     { split($i, kv2, "="); fp = k[2] + 0 }
      if ($i ~ /^prompt_tokens=/) { split($i, kv2, "="); pt = kv2[2] + 0 }
    }
    key = arm " ctx=" ctx
    n[key]++
    if (p > 0) { ps[key] += p; if (p > pmax[key]) pmax[key] = p; if (pmin[key] == 0 || p < pmin[key]) pmin[key] = p }
    if (d > 0) { ds[key] += d; if (d > dmax[key]) dmax[key] = d; if (dmin[key] == 0 || d < dmin[key]) dmin[key] = d }
    if (fp > fpmax[key]) fpmax[key] = fp
    toks[key] = pt
  }
  END { for (k in n) printf "%-28s n=%d tokens=%s prefill=%.2f[%.2f-%.2f] decode=%.3f[%.3f-%.3f] foot_max=%.2f\n",
        k, n[k], toks[k], ps[k]/n[k], pmin[k], pmax[k], ds[k]/n[k], dmin[k], dmax[k], fpmax[k] }
' | sort -t= -k2 -n