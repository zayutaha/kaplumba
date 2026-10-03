#!/bin/zsh
# Interleaved A/B benchmark for the device-side sampler (spec/device-sampler).
#
# AGENTS.md §10.5: decode on this box has been measured at 3.7, 3.9, 4.3, 4.8,
# 6.0 and 6.4 tok/s from the SAME binary minutes apart, so one reading is not a
# result and a difference smaller than the spread is noise. This runs N
# interleaved pairs of two *binaries* and reports the median and the range of
# each arm.
#
#   scripts/sampler_ab.sh <binA> <labelA> <binB> <labelB> <pairs>
#
# Environment: KAPLUMBA_AB_MODEL, KAPLUMBA_AB_PROMPT, KAPLUMBA_AB_TEMP,
# KAPLUMBA_AB_MAXNEW, KAPLUMBA_AB_SEED.
#
# The model is loaded fresh in every run (the engine's flock makes two concurrent
# instances impossible), so a run costs ~10 s of load plus the generation.

set -u

M=${KAPLUMBA_AB_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
PROMPT=${KAPLUMBA_AB_PROMPT:-Explain the theory of relativity in two paragraphs.}
TEMP=${KAPLUMBA_AB_TEMP:-1.0}
MAXNEW=${KAPLUMBA_AB_MAXNEW:-96}
SEED=${KAPLUMBA_AB_SEED:-1234}

if [[ $# -lt 5 ]]; then
  print -u2 "usage: $0 <binA> <labelA> <binB> <labelB> <pairs>"
  exit 2
fi
BIN_A=$1; LA=$2; BIN_B=$3; LB=$4; N=$5

one() {
  local bin=$1 label=$2
  local out=$("$bin" --model "$M" --prompt "$PROMPT" \
          --max-new-tokens $MAXNEW --temperature $TEMP --seed $SEED \
          --metrics 2>/dev/null)
  local tps=$(print -r -- "$out" | grep -o 'decode_tok_s=[0-9.]*' | head -1 | cut -d= -f2)
  # The GENERATED TEXT ONLY. Hashing the whole metrics line would fold
  # load_ms / ttft_ms into the comparison, and those vary per run -- which has
  # produced a false "divergence" alarm before.
  local text=$(print -r -- "$out" | grep -v '^prompt_tokens=')
  local th=$(print -rn -- "$text" | shasum | cut -c1-12)
  print -r -- "$label $tps $th"
}

typeset -a A B
print "temp=$TEMP maxnew=$MAXNEW seed=$SEED pairs=$N"
for ((i=1; i<=N; i++)); do
  A+=( "$(one $BIN_A $LA)" )
  B+=( "$(one $BIN_B $LB)" )
  print "pair $i: ${A[-1]}   |   ${B[-1]}"
done

# zsh's `print -r -- "$@"` collapses the argument list to one word, so feed the
# values through `awk` to re-split them; an earlier version of this printed the
# median and the "range" as the same number twice.
med() { print -r -- "$*" | tr ' ' '\n' | sort -g | awk '{v[NR]=$1} END{print (NR%2)? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2}'; }
sp()  { print -r -- "$*" | tr ' ' '\n' | sort -g | awk 'NR==1{lo=$1} {hi=$1} END{print "min "lo" max "hi" spread_pct "100*(hi-lo)/((hi+lo)/2)}'; }

ta=(); tb=()
for row in "${A[@]}"; do ta+=( "$(print -r -- $row | awk '{print $2}')" ); done
for row in "${B[@]}"; do tb+=( "$(print -r -- $row | awk '{print $2}')" ); done
print "---"
print "$LA: median $(med $ta) tok/s   $(sp $ta)"
print "$LB: median $(med $tb) tok/s   $(sp $tb)"
print "distinct text hashes: A=$(print -r -- "${A[@]}" | awk '{print $3}' | sort -u | wc -l | tr -d ' ') B=$(print -r -- "${B[@]}" | awk '{print $3}' | sort -u | wc -l | tr -d ' ')"