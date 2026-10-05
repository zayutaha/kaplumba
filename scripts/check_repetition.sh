#!/usr/bin/env bash
# check_repetition.sh -- is a long generation still coherent, or has it collapsed?
#
# usage: scripts/check_repetition.sh <file> [<file> ...]
#
# AC5 and the quality gate ask for >=512-token generations to be "checked for
# repetition collapse". Eyeballing a 512-token answer is not a check, and neither
# is counting repeated characters: degenerate output is measured best as the
# **distinct n-gram ratio** -- the fraction of n-token windows that occur exactly
# once. Healthy English text sits high and falls smoothly with n; a loop drives it
# toward zero by n = 20.
#
# Works on the engine's `--show-ids` token-id stream when one is available (a
# token is the unit that matters, not a character) and on raw text otherwise.
# A run that repeats one 8-token phrase 40 times reads `distinct_8 ≈ 0.2`; the same
# 512 tokens written normally read `distinct_8 > 0.95`.
set -uo pipefail

for f in "$@"; do
  [ -r "$f" ] || { printf '%s: unreadable\n' "$f"; continue; }
  case "$f" in
    *.ids) toks=$(tr ' ' '\n' < "$f" | grep -v '^$') ;;
    *)     toks=$(tr ' ' '\n' < "$f" | grep -v '^$') ;;
  esac
  n=$(printf '%s\n' "$toks" | wc -l | tr -d ' ')
  printf '%s  tokens=%s\n' "$f" "$n"
  for k in 4 8 16 32; do
    printf '%s\n' "$toks" | awk -v k="$k" -v n="$n" '
      { w[NR] = $0 }
      END {
        if (n < k) { printf "  distinct_%-3d n/a (fewer than %d tokens)\n", k, k; exit }
        delete seen
        total = 0; uniq = 0
        for (i = 1; i + k - 1 <= n; i++) {
          key = ""
          for (j = 0; j < k; j++) key = key w[i + j] "\036"
          total++
          if (!(key in seen)) { seen[key] = 1; uniq++ }
        }
        printf "  distinct_%-3d %.4f  (%d windows, %d unique)\n", k, uniq / total, total, uniq
      }'
  done
done