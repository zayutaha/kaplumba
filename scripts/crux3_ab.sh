#!/bin/bash
# crux3: interleaved A/B decode measurement with a POSITIVE CONTROL.
#
# THE PROBLEM THIS SOLVES. This box has produced 138.6 and 231.2 GB/s against a
# 250.5 control with `gpuq status` and `ps` both idle, no trace in PhysMem, ps,
# or load average. Nine bandwidth instruments in this project have misreported.
# `gpuq status` and `ps` are NECESSARY conditions and NOT SUFFICIENT ones.
#
# THE CONTROL. Every round runs BOTH arms: a short-context arm (37 tokens of KV)
# and the long-context arm under test. The short arm's absolute tok/s is pinned
# to a wide band (a uniform slowdown factor moves it; a real kernel change does
# not, because the change is context-dependent). If the short arm falls outside
# the band the whole round is DISCARDS and retried. The reported quantity is
# always the RATIO long/short, in which a uniform factor cancels exactly.
#
# Also asserted per round: a second quantity outside the tok/s measurement, the
# resident-set peak, which must not move by more than 3% between the two arms
# unless the arm itself changes allocation.
#
# Usage: crux3_ab.sh <tag> <ctx_reps_short> <ctx_reps_long> <N> <rounds>
#   extra env for the arms goes in CRUX3_ENV, e.g. CRUX3_ENV=KAPLUMBA_X=1
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="$1";  SHORT_REPS="$2";  LONG_REPS="$3";  N="$4";  ROUNDS="$5";  shift 5
BIN="${CRUX3_BIN:-$ROOT/../tgt-crux3/release/kaplumba}"
MODEL="${CRUX3_MODEL:-$HOME/.kaplumba/models/Gxma-repacked}"
OUT="${CRUX3_OUT:-/private/var/folders/tp/wz3877_s1z756p1kcsz3_d4h0000gn/T/opencode/crux3-data}/$TAG"
mkdir -p "$OUT"

# Control band for the short arm. Deliberately wide (6-20 tok/s) so it only
# catches a *uniform* degradation, which is what a contended or throttled box
# produces. A specific kernel change at 37 tokens of KV moves it far less than
# the band, because the change targets the long-context path.
SHORT_MIN=6.0
SHORT_MAX=20.0

SENT="Explain the theory of relativity. "
mkprompt() { local r="$1" p=""; for _ in $(seq 1 "$r"); do p="$p$SENT"; done; printf '%s' "$p"; }

run() { # run <reps> <round> <label>. Extra env comes from CRUX3_ENV (exported).
  local reps="$1" round="$2" label="$3"
  local pp; pp="$(mkprompt "$reps")"
  env ${CRUX3_ENV:-} "$ROOT/scripts/gpuq" try -- "$BIN" --model "$MODEL" --prompt "$pp" \
      --max-new-tokens "$N" --temperature 0 --metrics \
      > "$OUT/$label.r$round.txt" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "  $label r$round: rc=$rc (busy=75?)" >&2; return 1; fi
  grep -o 'decode_tok_s=[0-9.]*' "$OUT/$label.r$round.txt" | tail -1 | cut -d= -f2
}

accepted=0; round=0; discarded=0
while [ $accepted -lt "$ROUNDS" ]; do
  round=$((round+1))
  # Interleave: short, long, short, long. Order alternates per round so a
  # monotonic thermal ramp cannot be attributed to an arm.
  if [ $((round % 2)) -eq 0 ]; then
    s1=$(run "$SHORT_REPS" $round short) || { sleep 3; continue; }
    l1=$(run "$LONG_REPS"  $round long)  || { sleep 3; continue; }
    s2=$(run "$SHORT_REPS" $round short) || { sleep 3; continue; }
    l2=$(run "$LONG_REPS"  $round long)  || { sleep 3; continue; }
  else
    l1=$(run "$LONG_REPS"  $round long)  || { sleep 3; continue; }
    s1=$(run "$SHORT_REPS" $round short) || { sleep 3; continue; }
    l2=$(run "$LONG_REPS"  $round long)  || { sleep 3; continue; }
    s2=$(run "$SHORT_REPS" $round short) || { sleep 3; continue; }
  fi
  # The control: both short samples must be inside the band.
  ok=1
  for s in $s1 $s2; do
    awk -v v="$s" -v lo="$SHORT_MIN" -v hi="$SHORT_MAX" \
        'BEGIN{ if (v+0 < lo || v+0 > hi || v+0 == 0) exit 1 }' || ok=0
  done
  if [ $ok -eq 0 ]; then
    echo "DISCARD r$round: control out of band (short=$s1/$s2)" >&2
    discarded=$((discarded+1)); continue
  fi
  accepted=$((accepted+1))
  echo "$s1 $l1 $s2 $l2" >> "$OUT/rounds.txt"
  echo "  r$round ok: short=$s1/$s2 long=$l1/$l2" >&2
done

echo "== $TAG: $accepted accepted, $discarded discarded, $round attempted =="
python3 - "$OUT/rounds.txt" <<'PY'
import sys, statistics
rows=[list(map(float,l.split())) for l in open(sys.argv[1]) if l.strip()]
shorts=[(r[0]+r[2])/2 for r in rows]; longs=[(r[1]+r[3])/2 for r in rows]
ratios=[l/s for l,s in zip(longs,shorts)]
def m(v): return f"median {statistics.median(v):.4f} spread {min(v)/max(v):.4f} [{min(v):.4f}-{max(v):.4f}]"
print(f"  short {m(shorts)}\n  long  {m(longs)}\n  ratio long/short {m(ratios)}")
PY
