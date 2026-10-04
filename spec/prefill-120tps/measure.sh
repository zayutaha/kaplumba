#!/usr/bin/env bash
# T0/T7 prefill measurement harness.
# Usage: measure.sh <reps> <name> <ctx> [env...]
#   ctx: 614 | 8k | 16k | 24k
set -u
cd "$(dirname "$0")/../.." || exit 1
M=~/.kaplumba/models/Gxma/Gxma
BIN=./target/release/kaplumba

reps=${1:-3}
name=${2:-run}
ctx=${3:-614}

case "$ctx" in
  614)  reps_n=100 ;;
  8k)   reps_n=1300 ;;
  16k)  reps_n=2660 ;;
  24k)  reps_n=3990 ;;
  *) echo "bad ctx $ctx" >&2; exit 2 ;;
esac
PP=$(python3 -c "print('Explain the theory of relativity. '*$reps_n)")

for i in $(seq 1 "$reps"); do
  out=$(KAPLUMBA_MAX_IN_FLIGHT=1 scripts/gpuq try -- "$BIN" --model "$M" --prompt "$PP" \
        --max-new-tokens 1 --temperature 0 --metrics 2>&1)
  pf=$(printf '%s\n' "$out" | grep -o 'prefill_tok_s=[0-9.]*' | head -1 | cut -d= -f2)
  tt=$(printf '%s\n' "$out" | grep -o 'ttft_ms=[0-9.]*' | head -1 | cut -d= -f2)
  pt=$(printf '%s\n' "$out" | grep -o 'prompt_tokens=[0-9]*' | head -1 | cut -d= -f2)
  if [ -z "${pf:-}" ]; then
    echo "$name $ctx r$i FAIL"
    printf '%s\n' "$out" | tail -5
  else
    echo "$name $ctx r$i prefill_tok_s=$pf ttft_ms=$tt prompt_tokens=$pt"
  fi
done
