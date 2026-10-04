#!/usr/bin/env bash
# gpuq_mem.sh -- sourceable memory-health helpers for measurement harnesses.
#
# USAGE:  . "$(dirname "$0")/gpuq_mem.sh"
#         printf '%s\n' "$(scripts/gpuq status)" | gpuq_mem_healthy 4
#
# WHY THIS IS A SEPARATE, SOURCEABLE FILE
#
# The first version of this logic lived only inside a measurement harness in
# /tmp, and its unit test (/tmp/test_parse.sh) tested a *reimplementation* of the
# parser pasted into the test file. That is a coverage illusion: the test could
# pass while the harness's real parser was broken. The parser now lives here,
# once, and both the harnesses and `scripts/test_gpuq_mem.sh` source this file.
#
# WHAT IS AND IS NOT PARSED HERE
#
# We do NOT parse `top` for *available* memory. `scripts/gpuq` already parses
# top's K/M/G/T unit switching correctly, and AGENTS.md 10.3 records the two bugs
# that came from doing it by hand: `[^0-9]*` backtracking to the *last* digit run
# so "7.4 GiB" read as "4", and a single-token "17G" having no space before the
# unit. `gpuq status` prints one fixed shape:
#
#   memory: 0.1 GiB available (reclaimable), 9.0 GiB in use by processes, 0 MiB swap
#
# (scripts/gpuq:150). We take that number verbatim and require the literal
# " GiB ", so a line in ANY other shape fails to match. Failing to match is the
# safe direction: the harness skips the arm instead of sampling a box it cannot
# vouch for.
#
# `top` IS parsed for the compressor and unused fields, which `gpuq status` does
# not expose and which the OQ5 rule needs. That parse is by FIELD NAME through a
# unit-aware tokenizer -- never by stripping non-digits, never by taking the last
# comma-separated field (AGENTS.md 10.3 records that this yields "unused." rather
# than a number).

# --- `top` PhysMem reader ----------------------------------------------------

# One snapshot, all named fields, in GiB: "used wired compressor unused".
# Reads a `top` PhysMem line from STDIN, so it is pure and unit-testable.
# Prints "-- -- -- --" for any field it cannot find, so a caller can never
# mistake a missing field for a zero, and prints nothing at all when there is no
# PhysMem line on stdin.
#
# macOS formats PhysMem as e.g.
#   PhysMem: 23G used (9165M wired, 5954M compressor), 110M unused.
# `used`, `wired` and `compressor` are inside/around the parens and `unused` is
# outside them, so a positional comma split is wrong (an early version of this
# file made exactly that mistake: split(",") yields 2 fields, so it found none).
top_physmem_fields() {
  awk '
    /PhysMem/ {
      used = wired = comp = unused = "--"
      if (match($0, "[0-9]+(\\.[0-9]+)?[KMGTkmgt]?[ \t]+used"))       used    = tok($0, RSTART, RLENGTH)
      if (match($0, "[0-9]+(\\.[0-9]+)?[KMGTkmgt]?[ \t]+wired"))      wired   = tok($0, RSTART, RLENGTH)
      if (match($0, "[0-9]+(\\.[0-9]+)?[KMGTkmgt]?[ \t]+compressor")) comp    = tok($0, RSTART, RLENGTH)
      if (match($0, "[0-9]+(\\.[0-9]+)?[KMGTkmgt]?[ \t]+unused"))     unused  = tok($0, RSTART, RLENGTH)
      print used, wired, comp, unused
      exit
    }
    function tok(s, rs, rl,   seg, t) {
      seg = substr(s, rs, rl)
      if (!match(seg, /^[0-9]+(\.[0-9]+)?[KMGTkmgt]?/)) return "--"
      t = substr(seg, RSTART, RLENGTH)
      if (t ~ /[Tt]$/)      return sprintf("%.2f", (t+0) * 1099511627776 / 1073741824)
      else if (t ~ /[Gg]$/) return sprintf("%.2f", (t+0) * 1073741824 / 1073741824)
      else if (t ~ /[Mm]$/) return sprintf("%.2f", (t+0) * 1048576 / 1073741824)
      else if (t ~ /[Kk]$/) return sprintf("%.2f", (t+0) * 1024 / 1073741824)
      else                  return sprintf("%.2f", (t+0))
    }'
}

# The same, live: one `top` snapshot.
top_physmem_live() { top -l 1 -n 0 2>/dev/null | top_physmem_fields; }

# One named field of a PhysMem line given as $1. Pure; used by the unit test.
#   top_physmem_gib_on "PhysMem: 23G used (11G wired, 554M compressor), 126M unused." unused
top_physmem_gib_on() {
  _physmem_field "${1:-}" "${2:-unused}"
}

# Convenience: one named LIVE field in GiB.  $1 = used|wired|compressor|unused
top_physmem_gib() { _physmem_field "$(top_physmem_live)" "${1:-unused}"; }

_physmem_field() {
  local idx
  case "${2:-unused}" in
    used) idx=1 ;; wired) idx=2 ;; compressor) idx=3 ;;
    unused) idx=4 ;;
    *) echo "--"; return ;;
  esac
  local all
  all=$(printf '%s\n' "$1" | top_physmem_fields)
  [ -n "$all" ] || { echo "--"; return; }
  # shellcheck disable=SC2086  # deliberate word split into 4 fields
  set -- $all
  [ "$#" -ge 4 ] && printf '%s\n' "${!idx}" || echo "--"
}

# --- `gpuq status` body readers ----------------------------------------------

# Print the available GiB from a `gpuq status` body read on stdin.
# Prints NOTHING when the body is missing, malformed, or in another shape.
gpuq_mem_available_gib() {
  sed -n 's/^memory: \([0-9.]*\) GiB available (reclaimable).*/\1/p' | head -1
}

# Is the box healthy?  $1 = min available GiB (default 4).  stdin = gpuq status.
# Fails closed: no parseable number is unhealthy, never healthy.
gpuq_mem_healthy() {
  local min=${1:-4} avail
  avail=$(gpuq_mem_available_gib)
  [ -n "$avail" ] || return 1
  awk -v a="$avail" -v m="$min" 'BEGIN { exit !(a + 0 >= m + 0) }'
}

# Convenience: run `gpuq status` ourselves and report health.  $1 = min GiB.
box_healthy() { scripts/gpuq status 2>/dev/null | gpuq_mem_healthy "${1:-4}"; }

# --- sample validity ---------------------------------------------------------
#
# OQ5: a 24k measurement taken while the box is in memory compression is not a
# property of the code under test. The measured distortion is 99x pessimistic
# (0.055 tok/s against 5.44) and one-sided, so a sweep that only ever produces
# pessimistic samples reads as "more splits is much worse".
#
# A gate that samples only BEFORE an arm is insufficient, because the arm is
# what pushes the box over (implementation_notes.md 13.3). So harnesses poll
# DURING the arm and classify afterwards with mem_sample_verdict.

# Default thresholds: the OQ5 invalidity rule, and the "box is thrashing" rule.
MEM_COMPRESSOR_LIMIT_GIB=${MEM_COMPRESSOR_LIMIT_GIB:-2.0}
MEM_UNUSED_LIMIT_GIB=${MEM_UNUSED_LIMIT_GIB:-0.5}

# Append one poll record: "<epoch> <compressor_gib> <unused_gib>".
# One `top` snapshot per record so the two fields are consistent.
mem_poll_record() {
  local all c u
  all=$(top_physmem_live)
  # shellcheck disable=SC2086  # deliberate word split into 4 fields
  set -- $all
  if [ "$#" -ge 4 ]; then c=${3}; u=${4}; else c="?"; u="?"; fi
  printf '%s %s %s\n' "$(date +%s)" "$c" "$u" >> "$1"
}

# Classify a poll-record file.
#   exit 0 -> sample VALID      (reason on stdout)
#   exit 1 -> sample INVALID    (reason on stdout; must NOT be reported as a number)
mem_sample_verdict() {
  local f=$1 cmax=0 umin=99999 c u n=0
  if [ -s "$f" ]; then
    while read -r _t c u; do
      n=$(( n + 1 ))
      case "$c" in ''|*[!0-9.]*) ;; *)
        awk -v a="$c" -v b="$cmax" 'BEGIN{exit !(a+0>b+0)}' && cmax=$c ;; esac
      case "$u" in ''|*[!0-9.]*) ;; *)
        awk -v a="$u" -v b="$umin" 'BEGIN{exit !(a+0<b+0)}' && umin=$u ;; esac
    done < "$f"
  fi
  if [ "$n" = "0" ]; then
    echo "INVALID no-memory-poll (the gate could not watch the run)"
    return 1
  fi
  if awk -v c="$cmax" -v l="$MEM_COMPRESSOR_LIMIT_GIB" 'BEGIN{exit !(c+0 > l+0)}'; then
    echo "INVALID compressor-peaked-at-${cmax}GiB (limit ${MEM_COMPRESSOR_LIMIT_GIB})"
    return 1
  fi
  if awk -v u="$umin" -v l="$MEM_UNUSED_LIMIT_GIB" 'BEGIN{exit !(u+0 < l+0)}'; then
    echo "INVALID unused-dropped-to-${umin}GiB (limit ${MEM_UNUSED_LIMIT_GIB})"
    return 1
  fi
  echo "VALID polls=$n compressor_max=${cmax}GiB unused_min=${umin}GiB"
  return 0
}