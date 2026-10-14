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
#   memory: 0.1 GiB top-unused, 9.0 GiB system active pages, 0 MiB swap
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
# `top` is a *snapshot* command and macOS throttles it: back-to-back `top -l 1`
# calls can return an empty PhysMem section under load, which is exactly when the
# reading matters. So retry a couple of times before giving up, and treat a still-
# empty result as "unknown" (`--`) rather than zero -- a zero here would read as
# "the box is healthy", which is the failure direction that costs a 50-minute
# sweep. Verified: 10 consecutive `top_physmem_live` calls return all four fields.
top_physmem_live() {
  local out attempt
  for attempt in 1 2 3; do
    out=$(top -l 1 -n 0 2>/dev/null | top_physmem_fields)
    [ -n "$out" ] && { printf '%s\n' "$out"; return 0; }
    sleep 0.2
  done
  # Fall back to `vm_stat`, which is a sysctl read and therefore unaffected by
  # `top`'s own throttling. This fallback is not optional: measured during an 8k
  # prefill, EVERY `top -l 1 -n 0` call inside a 4-second poll loop returned no
  # PhysMem line at all (25 of 25 polls read `--`), so a `top`-only poller is
  # blind exactly while the box is busiest. `vm_stat` answers under the same load.
  vm_stat_mem_gib
}

# Compressor occupancy and reclaimable memory in GiB from a `vm_stat` body read
# on stdin: "compressor unused". Pure, so the unit test can check the page
# arithmetic against a hand-computed line instead of against the live box.
#
# Reclaimable ("unused") is free + inactive + speculative + purgeable, NOT
# `Pages free` alone. AGENTS.md 10.3 records why: `Pages free` counts only pages
# that have never been touched, so it ignores the file cache, and on an idle box
# it read 1.47 GiB while `top` said 14 GiB unused -- which made `scripts/gpuq` warn
# on a healthy box and stop agents from running. It is the same sum `scripts/gpuq`'s
# own fallback uses, so the two readers cannot disagree about what the word means.
#
# Prints "-- --" when a field is absent, never 0: a zero here reads as "the box is
# healthy", which is the failure direction that costs a whole sweep.
vm_stat_gib_from_stdin() {
  local out
  # `vm_stat` lines are `Label: value.` -- the label contains SPACES and colons
  # ("Pages occupied by compressor:"), so the value is stripped from the RIGHT by
  # removing trailing dots, not by splitting on ":". A `gsub(/\./,"")` over the
  # whole line would corrupt nothing here (the labels have no digits) but would
  # break the moment one did.
  out=$(awk '
    { v = $NF; gsub(/[.]/, "", v) }
    /^Pages occupied by compressor:/ { c = v }
    /^Pages free:/        { s += v }
    /^Pages inactive:/    { s += v }
    /^Pages speculative:/ { s += v }
    /^Pages purgeable:/   { s += v }
    END {
      if (c == "") c = "--"; else c = sprintf("%.2f", c * 16384 / 1073741824)
      if (s == 0)  s = "--"; else s = sprintf("%.2f", s * 16384 / 1073741824)
      print c, s
    }')
  printf '%s\n' "$out"
}

vm_stat_compressor_gib() { vm_stat_gib_from_stdin < <(vm_stat 2>/dev/null) | cut -d' ' -f1; }
vm_stat_mem_gib()        { vm_stat_gib_from_stdin < <(vm_stat 2>/dev/null) | cut -d' ' -f2; }

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

# Print the top-unused GiB from a `gpuq status` body read on stdin.
# Prints NOTHING when the body is missing, malformed, or in another shape.
gpuq_mem_available_gib() {
  sed -n 's/^memory: \([0-9.]*\) GiB top-unused.*/\1/p' | head -1
}

# Is the box healthy?  $1 = min top-unused GiB (default 4).  stdin = gpuq status.
# Fails closed: no parseable number is unhealthy, never healthy.
gpuq_mem_healthy() {
  local min=${1:-4} avail
  avail=$(gpuq_mem_available_gib)
  [ -n "$avail" ] || return 1
  awk -v a="$avail" -v m="$min" 'BEGIN { exit !(a + 0 >= m + 0) }'
}

# Parse `sysctl -n vm.swapusage`, reporting the `used` field in MiB. Read the
# system field by name: positional parsing here once reported `total` as swap
# usage because the three values have the same unit suffix.
gpuq_swap_used_mib_from_stdin() {
  awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i != "used") continue
        value = $(i + 2)
        unit = substr(value, length(value), 1)
        sub(/[KMG]$/, "", value)
        multiplier = unit == "G" ? 1024 : unit == "K" ? (1 / 1024) : unit == "M" ? 1 : 0
        if (multiplier == 0 || value !~ /^[0-9]+([.][0-9]+)?$/) break
        printf "%.0f", value * multiplier
        found = 1
        exit
      }
    }
    END { if (!found) print "?" }
  '
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

# --- thresholds, calibrated on this box (2026-10-04) -------------------------
#
# **`unused` is the discriminator; the compressor is not.** Measured on a healthy
# box, with runs whose throughput is uncontroversially good:
#
# | run | prefill tok/s | unused_min | compressor_max |
# |---|---|---|---|
# | 626 tokens | 100.9 | 2.42 GiB | 6.32 GiB |
# | 16 394 tokens | 74.6 | 0.67 GiB | 7.01 GiB |
#
# **Loading the model itself drives the compressor to 6-7 GiB** (the 11.5 GiB of
# weights evict the desktop's own pages), so a 2 GiB compressor limit rejects
# *every* run including perfect ones. The documented OQ5 stall signature
# (implementation_notes.md 13.1, 14.4) is `126M unused` with a 5.5 GB
# compressor -- i.e. it is `unused` that separates the two states, and the
# compressor number is incidental.
#
# So: `unused` below the limit is INVALID, and the compressor is recorded but not
# gated on. **Measured, and this is the uncomfortable part: `unused` does not
# discriminate either.** The OQ5 stall signature reads 110-126 MiB unused
# (implementation_notes.md 13.1, 14.4), but a 626-token run returning **100.6
# tok/s** on this same box also polls down to **0.09 GiB**. Measured traces:
#
# | run | result | unused_min | compressor_max |
# |---|---|---|---|
# | 626 tok, 3 s polls | 100.6 tok/s | 2.41 GiB | 0.23 GiB |
# | 626 tok, 5 s polls | 100.9 tok/s | 0.09 GiB | 9.47 GiB |
# | 16 394 tok | 74.6 tok/s | 0.67 GiB | 7.01 GiB |
#
# The difference between those first two rows is the desktop's own state, not the
# benchmark: whether the GUI's pages have been evicted yet when the poll lands.
# **A memory-pressure gate on this box cannot tell a good run from a stalled one,
# and a threshold tuned on either signal rejects good runs.** The limit is
# therefore set where only a genuinely wedged box trips it (0.05 GiB = 51 MiB,
# below every value observed above), and the REAL instrument is interleaved A/B
# with the poll file kept as context -- see `bench_prefill_passes.sh`.
#
# This is OQ8 restated: the measurement, not the kernel, is the binding
# constraint. Recorded rather than tuned away, because a future session that
# "fixes" the threshold will otherwise rediscover that it cannot be fixed.
MEM_UNUSED_LIMIT_GIB=${MEM_UNUSED_LIMIT_GIB:-0.05}
# Informational only. Set to a negative number to gate on it (off by default,
# because the calibration above shows it does not discriminate on this box).
MEM_COMPRESSOR_LIMIT_GIB=${MEM_COMPRESSOR_LIMIT_GIB:--1}
# OQ5's rule: ">2% CPU in compression means the sample is INVALID".
MEM_CPU_COMPRESSION_LIMIT_PCT=${MEM_CPU_COMPRESSION_LIMIT_PCT:-2.0}

# Append one poll record: "<epoch> <compressor_gib> <unused_gib>".
# One `top` snapshot per record so the two fields are consistent.
mem_poll_record() {
  # NOTE: the destination must be captured BEFORE any `set --` below, which
  # overwrites the positional parameters. Getting this wrong makes the function
  # append to a file whose name is the first PhysMem field, so the poll file
  # stays empty and every sample is classified "no-memory-poll".
  local dest=$1 all c u
  all=$(top_physmem_live)
  # shellcheck disable=SC2086  # deliberate word split into 4 fields
  set -- $all
  if [ "$#" -ge 4 ]; then c=${3}; u=${4}; else c="?"; u="?"; fi
  # `top` can be throttled to nothing under load. `vm_stat` is a sysctl read and
  # answers regardless -- measured: 25 of 25 `top` polls returned nothing during an
  # 8k prefill while `vm_stat` answered every time. Prefer `top` (it is what the
  # rest of this repo reads, and its `unused` is the field AGENTS.md quotes) and
  # fall back per-field, so a throttled `top` still produces a usable record
  # instead of 25 lines of `--`.
  case "$c" in ''|--|*[!0-9.]*) c=$(vm_stat_compressor_gib) ;; esac
  case "$u" in ''|--|*[!0-9.]*) u=$(vm_stat_mem_gib) ;; esac
  printf '%s %s %s\n' "$(date +%s)" "$c" "$u" >> "$dest"
}

# Classify a poll-record file.
#   exit 0 -> sample VALID      (reason on stdout)
#   exit 1 -> sample INVALID    (reason on stdout; must NOT be reported as a number)
# Only polls at or after the marker named by `MEM_PHASE_LOAD_DONE` count. A model
# LOAD takes this box from 17 GiB unused to ~2 GiB in about 8 seconds on its own --
# measured on a 626-token run that returned 100.6 tok/s: unused went
# 16.66 -> 7.15 -> 3.10 -> 2.41 GiB across the load and then stayed put. A gate
# that watches the whole run sees that transient and calls a perfectly good run
# compressed, which is how a calibration of the threshold ended up rejecting every
# sample including the known-good ones.
#
# The marker file is written by the harness once the engine's own load phase has
# finished. Absent a marker, every poll counts (the conservative behaviour).
mem_sample_verdict() {
  # Read the marker from the environment at CALL time, not at source time: the
  # harness sets `MEM_PHASE_LOAD_DONE` per arm, and a value latched when this file
  # was sourced would apply one arm's marker to the next.
  local mem_phase_load_done=${MEM_PHASE_LOAD_DONE:-}
  mem_sample_verdict_from "$1" "$mem_phase_load_done"
}

mem_sample_verdict_from() { # <poll file> <marker path, or empty>
  local f=$1 marker=${2:-}
  local cmax=0 umin="" c u n=0 unreadable=0 skipped=0 t mark=0
  if [ -n "$marker" ] && [ -e "$marker" ]; then
    mark=$(stat -f %m "$marker" 2>/dev/null || echo 0)
  fi
  if [ -s "$f" ]; then
    while read -r t c u; do
      if [ "$mark" != "0" ] && [ "$t" -lt "$mark" ]; then
        skipped=$(( skipped + 1 ))
        continue
      fi
      n=$(( n + 1 ))
      case "$c" in ''|--|*[!0-9.]*) unreadable=$(( unreadable + 1 )) ;;
        *) awk -v a="$c" -v b="$cmax" 'BEGIN{exit !(a+0>b+0)}' && cmax=$c ;; esac
      case "$u" in ''|--|*[!0-9.]*) unreadable=$(( unreadable + 1 )) ;;
        *) if [ -z "$umin" ] || awk -v a="$u" -v b="$umin" 'BEGIN{exit !(a+0<b+0)}'; then
             umin=$u
           fi ;; esac
    done < "$f"
  fi
  if [ "$skipped" -gt 0 ] && [ "$n" = "0" ]; then
    echo "INVALID all $skipped polls predate the load marker (the gate never watched the compute)"
    return 1
  fi
  if [ "$n" = "0" ]; then
    echo "INVALID no-memory-poll (the gate could not watch the run)"
    return 1
  fi
  # A poll that could read neither field is not a healthy reading, it is a blind
  # one. Reported rather than silently dropped: a run whose polls are all
  # unreadable has NOT been shown to be healthy, and the safe direction is to
  # refuse the sample.
  if [ "$unreadable" -gt 0 ]; then
    echo "INVALID $unreadable of $(( n * 2 )) polled fields were unreadable"
    return 1
  fi
  # Skipped-but-present polls are a warning, not a verdict: the load transient is
  # expected, and the compute polls are what matter. Reported so a reader can see
  # how much of the run the gate actually watched.
  if awk -v c="$cmax" -v l="$MEM_COMPRESSOR_LIMIT_GIB" \
      'BEGIN { exit !(l >= 0 && c + 0 > l + 0) }'; then
    echo "INVALID compressor-peaked-at-${cmax}GiB (limit ${MEM_COMPRESSOR_LIMIT_GIB})"
    return 1
  fi
  if awk -v u="$umin" -v l="$MEM_UNUSED_LIMIT_GIB" 'BEGIN{exit !(u+0 < l+0)}'; then
    echo "INVALID unused-dropped-to-${umin}GiB (limit ${MEM_UNUSED_LIMIT_GIB})"
    return 1
  fi
  echo "VALID polls=$n load-skipped=$skipped compressor_max=${cmax}GiB unused_min=${umin}GiB"
  return 0
}

# OQ8's rule, as a checkable predicate: "a sample with >2% CPU in compression is
# invalid". Implemented against `/usr/bin/time -l`'s own `sys` figure, which is
# what AGENTS.md 10.2 #7 and OQ5 actually describe ("check ... the process's CPU
# time; a sample with >2% CPU in compression is invalid"). `real` is wall time and
# `sys` is kernel time on the same process, so their ratio is the fraction of
# wall time spent in the kernel -- which is where compression shows up.
#
# Reported separately from `mem_sample_verdict` because it needs the run's
# `time -l` output, not the poll file, and because it is a *corroborating* check:
# a run can compress without the poller seeing it, and vice versa.
mem_cpu_compression_ok() { # <path to a /usr/bin/time -l log>
  local log=$1 real sys pct
  # `/usr/bin/time -l` line: `  104.26 real         0.50 user         5.25 sys`
  # so real is $1, user is $3 and sys is $5. Anchoring on `$1` and stepping by
  # two is what makes this robust to the column count `time` happens to emit.
  real=$(awk '$2 == "real" { print $1; exit }' "$log")
  sys=$(awk '$6 == "sys" { print $5; exit }' "$log")
  case "$real$sys" in
    ''|*[!0-9.]*) echo "unknown (no time -l real/sys in $log)"; return 0 ;;
  esac
  awk -v r="$real" -v s="$sys" -v n="$MEM_CPU_COMPRESSION_LIMIT_PCT" \
    'BEGIN { p = (r > 0) ? 100 * s / r : 0; exit !(p <= n) }' || {
    pct=$(awk -v r="$real" -v s="$sys" 'BEGIN{printf "%.1f", (r>0)?100*s/r:0}')
    echo "sys-cpu ${pct}% of wall (limit ${MEM_CPU_COMPRESSION_LIMIT_PCT}%)"
    return 1
  }
  pct=$(awk -v r="$real" -v s="$sys" 'BEGIN{printf "%.1f", (r>0)?100*s/r:0}')
  echo "sys-cpu ${pct}% of wall"
  return 0
}
