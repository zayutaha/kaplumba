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

echo "== live =="
LIVE=$(scripts/gpuq status 2>/dev/null)
printf '  raw: %s\n' "$(printf '%s\n' "$LIVE" | grep '^memory:')"
AVAIL=$(printf '%s\n' "$LIVE" | gpuq_mem_available_gib)
printf '  parsed: %s GiB (threshold %s)\n' "${AVAIL:-<EMPTY>}" "$MIN_UNUSED_GIB"
if [ -z "$AVAIL" ]; then echo "  FAIL: live parse empty"; fail=1
elif printf '%s\n' "$LIVE" | gpuq_mem_healthy "$MIN_UNUSED_GIB"; then echo "  live verdict: healthy"
else echo "  live verdict: unhealthy (not a failure -- the box is busy)"; fi
printf '  live PhysMem: used/wired/compressor/unused = %s\n' "$(top_physmem_live)"

echo "== verdict classifier (in-run sampling) =="
TD=$(mktemp -d); trap 'rm -rf "$TD"' EXIT
# A healthy run: low compressor, plenty unused.
printf '%s %s %s\n' 1 0.10 6.20 > "$TD/ok"
printf '%s %s %s\n' 2 0.30 3.10 >> "$TD/ok"
if mem_sample_verdict "$TD/ok" > "$TD/v"; then printf '  ok   %-26s %s\n' "healthy run VALID" "$(cat "$TD/v")"
else printf '  FAIL %-26s %s\n' "healthy run VALID" "$(cat "$TD/v")"; fail=1; fi
# An OQ5 compression run: peak 5.9 GiB compressor, unused down to 110 MiB.
printf '%s %s %s\n' 1 0.20 6.00 > "$TD/bad"
printf '%s %s %s\n' 2 5.90 0.11 >> "$TD/bad"
if mem_sample_verdict "$TD/bad" > "$TD/v2"; then printf '  FAIL %-26s %s\n' "compressed run caught" "$(cat "$TD/v2")"; fail=1
else printf '  ok   %-26s %s\n' "compressed run caught" "$(cat "$TD/v2")"; fi
# No polling at all must be INVALID, not silently VALID.
if mem_sample_verdict "$TD/none" > "$TD/v3"; then printf '  FAIL %-26s %s\n' "no-poll caught" "$(cat "$TD/v3")"; fail=1
else printf '  ok   %-26s %s\n' "no-poll caught" "$(cat "$TD/v3")"; fi

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
GPUQ_BIN="$STUB" LOG="$LOG" MIN_UNUSED_GIB=4 \
  scripts/bench_decode_splits.sh 64 1 > "$TD/out" 2>&1
if [ -e "$SENTINEL" ]; then
  echo "  FAIL: harness ran the model on a box it could not parse"; fail=1
else echo "  ok   harness did not run the model on an unparseable box"; fi
if grep -q 'SKIPPED box-unhealthy' "$LOG"; then echo "  ok   arm recorded as SKIPPED"
else echo "  FAIL: no SKIPPED line in the log"; cat "$LOG"; fail=1; fi

echo "== end-to-end positive control: a HEALTHY box must let the arm through =="
STUB2="$TD/gpuq_ok"
cat > "$STUB2" <<'STUBEOF'
#!/usr/bin/env bash
if [ "${1:-}" = "status" ]; then
  echo "memory: 9.9 GiB available (reclaimable), 4 GiB in use by processes, 0 MiB swap"
  exit 0
fi
echo "STUB-RAN-THE-BINARY ok"
exit 0
STUBEOF
chmod +x "$STUB2"
LOG2="$TD/sweep2.log"
GPUQ_BIN="$STUB2" LOG="$LOG2" MIN_UNUSED_GIB=4 DRY_RUN=1 \
  scripts/bench_decode_splits.sh 64 1 > "$TD/out2" 2>&1 || true
if grep -q 'DRY-RUN' "$LOG2"; then echo "  ok   healthy box let the arm run"
else echo "  FAIL: healthy box blocked the arm"; cat "$LOG2"; fail=1; fi

[ "$fail" -eq 0 ] && echo "ALL PASS" || echo "FAILURES PRESENT"
exit "$fail"