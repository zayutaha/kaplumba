#!/usr/bin/env bash
# test_gpuq_mem.sh -- unit test for scripts/gpuq_mem.sh (N5).
#
# This file is the fixed, committed successor of /tmp/test_parse.sh. Two things
# changed beyond the bug fix:
#
#  1. It sources the REAL parser (scripts/gpuq_mem.sh) instead of a pasted copy.
#     The old test reimplemented the parser, so it could pass while the harness's
#     parser was broken -- a coverage illusion.
#  2. It adds an END-TO-END detection test: scripts/bench_decode_splits.sh is run
#     with a STUBBED gpuq that emits a garbage memory line, and the arm must be
#     refused. "The unit test passes" is not the claim the brief asks for; "the
#     harness actually skips an unhealthy sample" is.
#
# Why a parser unit test exists at all: AGENTS.md 10.3 records a 1100x
# overstatement ("554M unused" read as 554 GiB) that shipped in scripts/gpuq
# before a test caught it, and this parser was itself wrong twice in a row
# (`[^0-9]*` backtracks to the LAST digit run, so "7.4 GiB" parsed as "4"; and a
# single-token "17G" has no space before the unit). Both were caught by a test,
# not by inspection.
set -uo pipefail

cd "$(dirname "$0")/.."
# shellcheck source=scripts/gpuq_mem.sh
. ./scripts/gpuq_mem.sh

MIN_UNUSED_GIB=${MIN_UNUSED_GIB:-4}
fail=0

check() { # name expected_healthy input
  local got
  if printf '%s\n' "$3" | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then got=1; else got=0; fi
  if [ "$got" = "$2" ]; then printf '  ok   %-26s healthy=%s\n' "$1" "$got"
  else printf '  FAIL %-26s expected healthy=%s got %s\n' "$1" "$2" "$got"; fail=1; fi
}

echo "== synthetic (gpuq's fixed 'memory: N GiB available (reclaimable)' format) =="
check "healthy 11.0 GiB"      1 "memory: 11.0 GiB available (reclaimable), 4.4 GiB in use by processes, 0 MiB swap"
check "healthy 4.0 GiB (==)"  1 "memory: 4.0 GiB available (reclaimable), 7 GiB in use, 0 MiB swap"
# The exact historical bug: a multi-digit decimal must not be read as its last
# digit run. This is the case a `[^0-9]*` sed expression got wrong.
check "healthy 7.4 GiB"       1 "memory: 7.4 GiB available (reclaimable), 8 GiB in use, 0 MiB swap"
check "stalled 0.12 GiB"      0 "memory: 0.12 GiB available (reclaimable), 23 GiB in use, 0 MiB swap"
check "unhealthy 1.47 GiB"    0 "memory: 1.47 GiB available (reclaimable), 0 GiB in use, 0 MiB swap"
# The OQ5 stall signature, already normalised by gpuq.
check "OQ5 stall 0.12"       0 "memory: 0.12 GiB available (reclaimable), 23 GiB in use by processes, 0 MiB swap"

echo "== must be treated as UNHEALTHY (fail closed, never invent a number) =="
check "no memory line"       0 "queue: 0 waiting"
check "wrong shape (17G)"    0 "memory: 17G available (reclaimable), 1G in use, 0 MiB swap"
check "unreclaimable only"   0 "warning: only 2.0 GiB available (unreclaimable); a load needs ~11 GiB"
check "garbage line"         0 "memory: no available (reclaimable), ???"
check "empty"                0 ""

echo "== the 554 MiB trap that AGENTS.md 10.3 records =="
# Raw `top` says "554M unused" = 0.55 GiB. If a future change feeds raw `top`
# output here, this must NOT become 554.
check "raw top 554M line"    0 "PhysMem: 23G used (11G wired, 554M compressor), 126M unused."
# ...and the named-field reader must agree, in GiB, not in bytes.
GOT_COMPRESSOR=$(top_physmem_gib_on "PhysMem: 23G used (11G wired, 554M compressor), 126M unused." compressor)
if [ "$GOT_COMPRESSOR" = "0.54" ]; then printf '  ok   %-26s %s\n' "554M compressor -> GiB" "$GOT_COMPRESSOR"
else printf '  FAIL %-26s expected 0.54 got %s\n' "554M compressor -> GiB" "$GOT_COMPRESSOR"; fail=1; fi
GOT_UNUSED=$(top_physmem_gib_on "PhysMem: 23G used (11G wired, 554M compressor), 126M unused." unused)
if [ "$GOT_UNUSED" = "0.12" ]; then printf '  ok   %-26s %s\n' "126M unused -> GiB" "$GOT_UNUSED"
else printf '  FAIL %-26s expected 0.12 got %s\n' "126M unused -> GiB" "$GOT_UNUSED"; fail=1; fi

TD=${TD_OVERRIDE:-$(mktemp -d)}
mkdir -p "$TD"
[ -n "${TD_KEEP:-}" ] || trap 'rm -rf "$TD"' EXIT

echo "== live =="
LIVE=$(scripts/gpuq status 2>/dev/null)
printf '  raw: %s\n' "$(printf '%s\n' "$LIVE" | grep '^memory:')"
AVAIL=$(printf '%s\n' "$LIVE" | gpuq_mem_available_gib)
printf '  parsed: %s GiB (threshold %s)\n' "${AVAIL:-<EMPTY>}" "$MIN_UNUSED_GIB"
if [ -z "$AVAIL" ]; then echo "  FAIL: live parse empty"; fail=1
elif printf '%s\n' "$LIVE" | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then echo "  live verdict: healthy"
else echo "  live verdict: unhealthy (not a failure -- the box is busy)"; fi
printf '  live PhysMem: used/wired/compressor/unused = %s\n' "$(top_physmem_live)"

echo "== live poll record (the classifier's only input) =="
# `mem_poll_record` must actually append a line. This caught a bug where the
# destination argument was clobbered by the `set --` that splits the PhysMem
# fields, so every poll wrote to a file named after the first field and the poll
# file stayed empty -- which the verdict classifier then reported as "no poll",
# making EVERY sample invalid. The failure was silent and one-sided: it can only
# ever discard good samples, never invent bad ones, but it discards all of them.
POLLFILE="$TD/poll"
: > "$POLLFILE"
mem_poll_record "$POLLFILE"
if [ -s "$POLLFILE" ]; then
  printf '  ok   %-26s %s\n' "poll appended" "$(cat "$POLLFILE")"
else
  printf '  FAIL %-26s the poll file is empty\n' "poll appended"; fail=1
fi
# The record must be three fields, or the verdict below is guessing -- and the two
# memory fields must be NUMBERS. A `top` throttled to nothing under load produces
# `--`, and a record of `-- -- --` that the classifier reads as "fine" is how a
# whole sweep silently loses its gate.
if awk 'NF == 3 && $2 ~ /^[0-9.]+$/ && $3 ~ /^[0-9.]+$/ { ok = 1 } END { exit !ok }' "$POLLFILE"; then
  printf '  ok   %-26s 3 numeric fields\n' "record shape"
else
  printf '  FAIL %-26s expected 3 fields with numeric memory readings, got: %s\n' \
    "record shape" "$(cat "$POLLFILE")"
  fail=1
fi

echo "== the vm_stat fallback the poller relies on under load =="
# `top -l 1 -n 0` is throttled and returns no PhysMem line while the box is busy;
# measured 25/25 empty polls during an 8k prefill. `vm_stat` is a sysctl read and
# answers. Both readers are checked against the SAME source line so they cannot
# silently disagree about units.
LINE="PhysMem: 23G used (9165M wired, 5954M compressor), 110M unused."
# 5954 pages of compressor = 5954 * 16384 / 2^30 = 0.0909 GiB. Reclaimable =
# (100 + 300 + 400 + 500) pages = 1300 * 16384 / 2^30 = 0.0198 GiB. Both computed
# by hand from the 16384-byte page size so a change to the divisor is caught.
VS=$(printf 'Mach Virtual Memory Statistics: (page size of 16384 bytes)\n\
Pages free: 100.\n\
Pages active: 200.\n\
Pages inactive: 300.\n\
Pages speculative: 400.\n\
Pages purgeable: 500.\n\
Pages occupied by compressor: 5954.\n' | vm_stat_gib_from_stdin)
if [ "$VS" = "0.09 0.02" ]; then
  printf '  ok   %-26s compressor=0.09 unused=0.02 GiB\n' "vm_stat arithmetic"
else
  printf '  FAIL %-26s expected "0.09 0.02", got "%s"\n' "vm_stat arithmetic" "$VS"; fail=1
fi
# `Pages active` must NOT be counted as reclaimable: it is in-use anonymous memory,
# not something the kernel can hand back. Including it is how a busy box reads as
# having free memory it does not have.
VS2=$(printf 'Pages free: 100.\n\
Pages active: 64000.\n\
Pages inactive: 300.\n\
Pages speculative: 400.\n\
Pages purgeable: 500.\n\
Pages occupied by compressor: 100.\n' | vm_stat_gib_from_stdin)
if [ "$VS2" = "0.00 0.02" ]; then
  printf '  ok   %-26s active pages excluded\n' "vm_stat reclaimable set"
else
  printf '  FAIL %-26s expected "0.00 0.02", got "%s"\n' "vm_stat reclaimable set" "$VS2"; fail=1
fi
# An empty body must report "--", never 0. A zero reads as "healthy", which is the
# failure direction that silently drops a whole sweep.
VSE=$(printf 'nothing here\n' | vm_stat_gib_from_stdin)
if [ "$VSE" = "-- --" ]; then printf '  ok   %-26s %s\n' "empty body fails closed" "$VSE"
else printf '  FAIL %-26s expected "-- --", got "%s"\n' "empty body fails closed" "$VSE"; fail=1; fi

echo "== verdict classifier (in-run sampling) =="
TD2="$TD"
# A healthy run: low compressor, plenty unused.
printf '%s %s %s\n' 1 0.10 6.20 > "$TD/ok"
printf '%s %s %s\n' 2 0.30 3.10 >> "$TD/ok"
if mem_sample_verdict "$TD/ok" > "$TD/v"; then printf '  ok   %-26s %s\n' "healthy run VALID" "$(cat "$TD/v")"
else printf '  FAIL %-26s %s\n' "healthy run VALID" "$(cat "$TD/v")"; fail=1; fi
# **The stall signature is INDISTINGUISHABLE from a good run on this box.** The
# documented OQ5 stall (implementation_notes.md 13.1) reads 5.9 GiB compressor and
# 110 MiB unused; the measured 626-token run that returned 100.9 tok/s read 9.47 GiB
# and 0.09 GiB -- *more* extreme on both fields. So this gate CANNOT detect the
# OQ5 stall, and the test asserts that fact rather than asserting a discrimination
# that does not exist.
#
# Two consequences, and both are the honest outcome:
#  1. the shipped threshold (0.05 GiB unused) is set below every value observed,
#     so it rejects nothing rather than rejecting good runs;
#  2. **the gate is not the instrument.** Interleaved A/B with spread, plus the
#     kept poll file, is. This is OQ8 restated -- the binding constraint is the
#     measurement, not the kernel -- and it is why the sweep harness records
#     `INVALID-NO-NUMBER` rather than a number whenever it is unsure.
printf '%s %s %s\n' 1 5.90 0.11 > "$TD/stall"
printf '  note  %-26s stall=5.90GiB/0.11 unused vs good=9.47GiB/0.09 unused -> indistinguishable\n' "OQ5 stall"
# The threshold still does what it says on the box, for a genuinely wedged one:
# zero usable memory is caught.
printf '1 0.00 0.01\n' > "$TD/wedged"
if mem_sample_verdict "$TD/wedged" > "$TD/v2"; then
  printf '  FAIL %-26s accepted a wedged box: %s\n' "wedged box caught" "$(cat "$TD/v2")"; fail=1
else printf '  ok   %-26s %s\n' "wedged box caught" "$(cat "$TD/v2")"; fi
# The MEASURED healthy runs from the table in gpuq_mem.sh. All four returned good
# throughput (100.6, 100.9, 74.6 tok/s), and all four must be VALID -- including
# the two whose `unused` dipped to 0.09 GiB, which is the point: on this box a
# good run and a stalled one can print the same memory numbers, so a threshold
# tuned on them rejects good runs. Both of these were rejections during this work.
printf '1 7.01 0.67\n' > "$TD/cal16k"
for cal in "6.32 2.42:626-tok/3s-polls" "9.47 0.09:626-tok/5s-polls" "7.01 0.67:16k-tok"; do
  vals=${cal%%:*}
  name=${cal#*:}
  printf '1 %s %s\n' ${vals} > "$TD/cal"
  if mem_sample_verdict "$TD/cal" > "$TD/vc"; then
    printf '  ok   %-26s %s\n' "$name" "$(cat "$TD/vc")"
  else
    printf '  FAIL %-26s rejected a known-good run: %s\n' "$name" "$(cat "$TD/vc")"; fail=1
  fi
done
# The compressor limit is OFF by default; turning it back on must bite.
if MEM_COMPRESSOR_LIMIT_GIB=5.0 mem_sample_verdict "$TD/cal16k" > "$TD/v9"; then
  printf '  FAIL %-26s the opt-in compressor limit did not bite\n' "compressor limit opt-in"; fail=1
else printf '  ok   %-26s %s\n' "compressor limit opt-in" "$(cat "$TD/v9")"; fi
# No polling at all must be INVALID, not silently VALID.
if mem_sample_verdict "$TD/none" > "$TD/v3"; then printf '  FAIL %-26s %s\n' "no-poll caught" "$(cat "$TD/v3")"; fail=1
else printf '  ok   %-26s %s\n' "no-poll caught" "$(cat "$TD/v3")"; fi
# Polls that exist but are all UNREADABLE must also be INVALID. This is the case
# `top` throttling produces under load, and treating it as healthy silently
# disables the gate for exactly the runs that need it.
printf '1 -- --\n2 -- --\n' > "$TD/blind"
if mem_sample_verdict "$TD/blind" > "$TD/v4"; then printf '  FAIL %-26s %s\n' "blind polls caught" "$(cat "$TD/v4")"; fail=1
else printf '  ok   %-26s %s\n' "blind polls caught" "$(cat "$TD/v4")"; fi
# Partly unreadable is still INVALID -- a blind sample is not a healthy sample.
printf '1 0.20 6.00\n2 -- --\n' > "$TD/half"
if mem_sample_verdict "$TD/half" > "$TD/v5"; then printf '  FAIL %-26s %s\n' "partly-blind caught" "$(cat "$TD/v5")"; fail=1
else printf '  ok   %-26s %s\n' "partly-blind caught" "$(cat "$TD/v5")"; fi
# A single readable poll with high memory is VALID -- the unreadable rule must not
# be so eager that it rejects a good sample.
printf '1 0.20 6.00\n' > "$TD/good1"
if mem_sample_verdict "$TD/good1" > "$TD/v6"; then printf '  ok   %-26s %s\n' "single good poll" "$(cat "$TD/v6")"
else printf '  FAIL %-26s %s\n' "single good poll" "$(cat "$TD/v6")"; fail=1; fi

echo "== OQ5's CPU rule (mem_cpu_compression_ok) =="
# `/usr/bin/time -l` prints "  <real> real  <user> user  <sys> sys". A 5% sys
# fraction must be rejected at the 2% limit; 0.4% must pass. Field order matters:
# taking the LAST field of the `sys` line gets the unit word, not the number.
printf '      104.26 real         0.50 user         5.25 sys\n' > "$TD/bad_cpu"
printf '      104.26 real         1.20 user         0.40 sys\n' > "$TD/good_cpu"
printf 'no time output here\n' > "$TD/no_cpu"
if mem_cpu_compression_ok "$TD/bad_cpu" > "$TD/c1"; then
  printf '  FAIL %-26s accepted 5.0%% sys: %s\n' "high sys rejected" "$(cat "$TD/c1")"; fail=1
else printf '  ok   %-26s %s\n' "high sys rejected" "$(cat "$TD/c1")"; fi
if mem_cpu_compression_ok "$TD/good_cpu" > "$TD/c2"; then
  printf '  ok   %-26s %s\n' "low sys accepted" "$(cat "$TD/c2")"
else printf '  FAIL %-26s rejected 0.4%% sys: %s\n' "low sys accepted" "$(cat "$TD/c2")"; fail=1; fi
mem_cpu_compression_ok "$TD/no_cpu" > "$TD/c3"
printf '  ok   %-26s %s\n' "no time output" "$(cat "$TD/c3")"

echo "== sabotage checks (AGENTS.md 3.4: a test that cannot fail is not a test) =="
# 1. An unreachable threshold must make the gate refuse a healthy box.
if printf '%s\n' "memory: 11.0 GiB available (reclaimable)" | gpuq_mem_healthy 1000000; then
  echo "  FAIL: gate passed an impossible threshold"; fail=1
else echo "  ok   gate rejects an unreachable threshold"; fi
# 2. The parser must not be fooled back into the greedy-digit bug.
P=$(printf '%s\n' "memory: 7.4 GiB available (reclaimable)" | gpuq_mem_available_gib)
if [ "$P" = "7.4" ]; then echo "  ok   greedy-digit bug stays fixed ($P)"; else
  echo "  FAIL: parser returned '$P' for 7.4"; fail=1; fi

echo "== end-to-end: does the harness actually REFUSE an unhealthy box? =="
STUB="$TD/gpuq"; LOG="$TD/sweep.log"; SENTINEL="$TD/binary-was-run"
rm -f "$SENTINEL"
cat > "$STUB" <<STUBEOF
#!/usr/bin/env bash
# Stub gpuq: reports a box whose memory line is garbage. Any subcommand that
# would run the real binary instead drops a sentinel file, because the harness
# redirects the arm's stdout/stderr into its own temp file -- so a marker
# printed to stderr would be invisible to the test.
if [ "\${1:-}" = "status" ]; then
  echo "gpu: none"
  echo "queue: 0 waiting"
  echo "memory: no available (reclaimable), ??? GiB in use by processes"
  exit 0
fi
: > "$SENTINEL"
echo "the binary was run on a box the gate could not parse"
exit 0
STUBEOF
chmod +x "$STUB"
GPUQ_BIN="$STUB" LOG="$LOG" MIN_UNUSED_GIB=4 MAXNEW=1 \
  scripts/bench_decode_splits.sh 64 1 > "$TD/out" 2>&1

echo "== end-to-end positive control: a HEALTHY box must let the arm through =="
STUB2="$TD/gpuq_ok"
cat > "$STUB2" <<'STUBEOF'
#!/usr/bin/env bash
# A healthy `gpuq`: the status line the gate parses, and for `try` it RUNS the
# command it was handed -- an earlier version just exited 0, so the stub binary was
# never invoked and the positive control was asserting against an empty log.
if [ "${1:-}" = "status" ]; then
  echo "memory: 9.9 GiB available (reclaimable), 4 GiB in use by processes, 0 MiB swap"
  exit 0
fi
shift
[ "${1:-}" = "--" ] && shift
exec "$@"
STUBEOF
chmod +x "$STUB2"
# And now the positive control, against the harness's own dispatch path with a
# stubbed `top` reporting a healthy box and a stubbed `KAPLUMBA_BIN` that succeeds.
# This is the assertion that the "it skipped" check above skipped for the right
# reason: it proves the harness is capable of running an arm at all.
STUBBIN="$TD/kaplumba"
cat > "$STUBBIN" <<'BINEOF'
#!/usr/bin/env bash
# Must take long enough for the harness's memory poller to land at least twice
# (POLL_SECS=1), because "no poll records" is a verdict the harness is right to
# return and the positive control must exercise the VALID path, not the
# INVALID-no-poll path. It must also print `time -l`'s shape, because the OQ5 CPU
# check reads it.
echo "loading stub model ... "
sleep 1
echo "loaded in 1000 ms"
sleep 1
echo "It does not matter."
echo "prompt_tokens=64 generated_tokens=1 prefill_tok_s=100.000 decode_tok_s=15.000"
echo "       2.50 real         0.10 user         0.05 sys"
exit 0
BINEOF
chmod +x "$STUBBIN"
LOG2="$TD/sweep2.log"
rm -f "$LOG2"
GPUQ_BIN="$STUB2" KAPLUMBA_BIN="$STUBBIN" LOG="$LOG2" \
  MIN_UNUSED_GIB=4 MAXNEW=1 POLL_SECS=1 ROUNDS=1 \
  scripts/bench_decode_splits.sh 64 1 > "$TD/out2" 2>&1 || true
if [ ! -s "$LOG2" ]; then
  echo "  FAIL: positive control wrote no log at all"; cat "$TD/out2" 2>/dev/null; fail=1
elif grep -qE 'splits1 1 .* VALID' "$LOG2"; then
  echo "  ok   healthy box dispatched the arm   $(grep -m1E ' VALID' "$LOG2" | cut -c1-96)"
else
  echo "  FAIL: healthy box did not dispatch a VALID arm"; cat "$LOG2"; fail=1
fi
# Now the SKIP side, which needs the log to exist at all. That check is why the
# missing log is itself a failure rather than a silent pass: this positive control
# once pointed at a `DRY_RUN` branch the committed harness no longer had, so the
# harness wrote nothing, the stub gpuq was never invoked, and the skip assertion
# below passed on a log that did not exist.
if [ ! -s "$LOG" ]; then
  echo "  FAIL: the skip arm wrote no log at $LOG"; cat "$TD/out" 2>/dev/null; fail=1
elif grep -q 'SKIPPED box-unhealthy' "$LOG"; then
  echo "  ok   arm recorded as SKIPPED"
else
  echo "  FAIL: no SKIPPED line in the log"; cat "$LOG"; fail=1
fi
if [ -e "$SENTINEL" ]; then
  echo "  FAIL: harness ran the model on a box it could not parse"; fail=1
else
  echo "  ok   harness did not run the model on an unparseable box"
fi

[ "$fail" -eq 0 ] && echo "ALL PASS" || echo "FAILURES PRESENT"
exit "$fail"