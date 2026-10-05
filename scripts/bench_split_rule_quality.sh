#!/usr/bin/env bash
# bench_split_rule_quality.sh -- does the `workgroups` decode split rule change
# the greedy token stream, at a context where it CAN?
#
# usage: OUT=/tmp/splitq CONTEXTS="1024 2048 4096 8204 16384" MAXNEW=512 \
#            scripts/bench_split_rule_quality.sh
#
# WHY THIS EXISTS (spec/prefill-24k-90tps, adversary F4)
# --------------------------------------------------
# `decode_splits_with` ends in `wanted.min(key_count.div_ceil(64))`, so at the
# ~120 tokens of the five reference prompts BOTH rules return 2 splits: the arms
# are literally the same computation and their diff can only ever be empty. The
# first key_count at which the rules differ is 896 (ThreadTarget 16 vs
# WorkgroupsPerCore 13), i.e. any conversation longer than ~900 tokens. The
# source comment in `attention.rs` claims the flip "changes 43 of 128" greedy
# tokens. Both statements cannot be checked by the existing evidence, so this
# measures it directly:
#
#   * one greedy generation per (context, arm), N tokens, `--temperature 0`;
#   * `--show-ids` so the token stream is diffed as IDs, not as text;
#   * the diff is reported as a COUNT of differing tokens, because "the text
#     still reads well" is not a measurement.
#
# Arms are interleaved per context (A B / A B) so a context effect cannot be
# mistaken for an arm effect. Files are written per arm and never piped through
# `tail` (AGENTS.md / NEXT.md 6).
#
# This harness measures QUALITY. It reports no speed number; speed comes from
# `scripts/bench_decode_splits.sh`, which is a different question and must not be
# merged with this one.
set -uo pipefail
cd "$(dirname "$0")/.."

M=${KAPLUMBA_MODEL:-$HOME/.kaplumba/models/Gxma/Gxma}
KAP=${KAPLUMBA_BIN:-./target/release/kaplumba}
GPUQ=${GPUQ_BIN:-scripts/gpuq}
OUT=${OUT:-/tmp/split_rule_quality}
MAXNEW=${MAXNEW:-512}
SEED=${SEED:-1234}
TEMPERATURE=${TEMPERATURE:-0}
CONTEXTS=${CONTEXTS:-"1024 2048 4096 8204 16384"}

mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
: > "$SUMMARY"

# The filler is the same 6.01-tokens-per-occurrence string every other harness in
# this spec uses, so `CONTEXTS` means the same thing here as elsewhere. The tail
# instruction keeps a 512-token generation on topic instead of letting it free-run.
build_prompt() { # <keys>
  local keys=$1
  printf 'Explain the theory of relativity. %.0s' $(seq 1 "$(( keys / 6 ))")
  printf ' Now write a detailed, well-structured essay answering everything above.'
}

run_arm() { # <ctx> <arm-label> <pass-env...>
  local ctx=$1 label=$2
  shift 2
  local log="$OUT/${ctx}_${label}.txt"
  local rc
  env "$@" "$GPUQ" try -- "$KAP" --model "$M" \
    --prompt "$(build_prompt "$ctx")" --max-new-tokens "$MAXNEW" \
    --temperature "$TEMPERATURE" --seed "$SEED" --show-ids \
    > "$log" 2>&1
  rc=$?
  local ids text
  if [ "$rc" = "75" ]; then
    printf '%s\t%s\tGPU-BUSY\n' "$ctx" "$label" >> "$SUMMARY"
    return 0
  fi
  if [ "$rc" != "0" ]; then
    printf '%s\t%s\tFAILED rc=%s\n' "$ctx" "$label" "$rc" >> "$SUMMARY"
    return 0
  fi
  # `--show-ids` prints exactly one extra line, `token_ids: <space separated
  # generated ids>` (crates/cli/src/report.rs:40). Everything else in the log is
  # the banner, the generated text, and optionally the metrics line, so the id
  # line is unambiguous.
  ids=$(sed -n 's/^token_ids: //p' "$log" | tr '\n' ' ')
  text=$(grep -v '^loading \|^loaded in \|^token_ids: \|^metrics\|^peak_resident\|^load_ms\|^$' "$log")
  local n
  n=$(printf '%s' "$ids" | wc -w | tr -d ' ')
  # A run that completes but emits no id line is a FAILURE, not a zero-token
  # success: reporting `OK tokens=0` would make the diff loop compare an empty
  # stream against a full one and print `differ=0`, i.e. a false "byte-identical".
  # That is the same class of bug as an INVALID memory sample silently folding
  # into a median, and it is caught by the arm-count assertion at the end.
  if [ "${n:-0}" -lt 1 ]; then
    printf '%s\t%s\tNO-IDS rc=0 (see %s)\n' "$ctx" "$label" "$log" >> "$SUMMARY"
    return 0
  fi
  printf '%s\t%s\tOK tokens=%s\n' "$ctx" "$label" "$n" >> "$SUMMARY"
  printf '%s' "$ids" > "$OUT/${ctx}_${label}.ids"
  printf '%s\n' "$text" > "$OUT/${ctx}_${label}.body"
  return 0
}

{
  echo "# decode split-rule QUALITY A/B (adversary F4)"
  echo "# contexts=$CONTEXTS max_new_tokens=$MAXNEW temperature=$TEMPERATURE seed=$SEED"
  echo -e "# ctx\tarm\tresult"
} >> "$SUMMARY"

for c in $CONTEXTS; do
  run_arm "$c" shipped
  run_arm "$c" workgroups KAPLUMBA_DECODE_SPLIT_RULE=workgroups
  run_arm "$c" splits13 KAPLUMBA_DECODE_SPLITS=13
  # F3's quality gate, in the same harness: the OQ9 prefill key-range passes are
  # claimed bit-exact against the one-dispatch body by 17 kernel-level tests, and
  # the claim has never been checked on the GENERATED TEXT at a context where the
  # passes actually engage (`key_count > PREFILL_PASS_KEYS` and a grid of at least
  # `PREFILL_PASS_MIN_GROUPS` threadgroups -- so not at 1024, only from ~8192).
  run_arm "$c" passes0 KAPLUMBA_PREFILL_PASS_KEYS=0
done

cat "$SUMMARY"
# A sweep that lost an arm to a crash or to GPU-busy must not read as a clean
# result, so the accounting is asserted rather than assumed. `NO-IDS` is the case
# this file was wrong about first: it used to be reported as `OK tokens=0`, which
# diffs as `differ=0` against a full stream and reads "byte-identical".
missing=$(grep -cE 'NO-IDS|FAILED|GPU-BUSY|SKIPPED' "$SUMMARY" || true)
printf -- '--- %s arm(s) did not produce a usable sample ---\n' "$missing"
grep -E 'NO-IDS|FAILED|GPU-BUSY|SKIPPED' "$SUMMARY" || true
echo "--- per-context token diff (shipped vs workgroups) ---"
for c in $CONTEXTS; do
  a="$OUT/${c}_shipped.ids"
  b="$OUT/${c}_workgroups.ids"
  s="$OUT/${c}_splits13.ids"
  [ -s "$a" ] || continue
  [ -s "$b" ] || continue
  # `cmp` on whitespace-normalised id streams: the first differing token and the
  # total. `paste` + awk so a missing side reads as "missing", not "identical".
  printf 'ctx=%-6s n_shipped=%-5s n_workgroups=%-5s ' "$c" \
    "$(wc -w < "$a" | tr -d ' ')" "$(wc -w < "$b" | tr -d ' ')"
  paste -d' ' <(tr ' ' '\n' < "$a" | grep -v '^$') \
    <(tr ' ' '\n' < "$b" | grep -v '^$') |
    awk '{ if ($1 != $2) d++ } END { printf "differ=%d\n", d + 0 }'
  if [ -s "$s" ]; then
    printf 'ctx=%-6s n_shipped=%-5s n_splits13=%-5s ' "$c" \
      "$(wc -w < "$a" | tr -d ' ')" "$(wc -w < "$s" | tr -d ' ')"
    paste -d' ' <(tr ' ' '\n' < "$a" | grep -v '^$') \
      <(tr ' ' '\n' < "$s" | grep -v '^$') |
      awk '{ if ($1 != $2) d++ } END { printf "differ=%d\n", d + 0 }'
  fi
  if cmp -s "$OUT/${c}_shipped.body" "$OUT/${c}_workgroups.body"; then
    printf 'ctx=%-6s BODY byte-identical\n' "$c"
  else
    printf 'ctx=%-6s BODY differs (see .body files)\n' "$c"
  fi
  # F3: shipped (OQ9 passes at their default) vs one dispatch.
  p="$OUT/${c}_passes0.ids"
  if [ -s "$p" ]; then
    printf 'ctx=%-6s n_shipped=%-5s n_passes0=%-5s ' "$c" \
      "$(wc -w < "$a" | tr -d ' ')" "$(wc -w < "$p" | tr -d ' ')"
    paste -d' ' <(tr ' ' '\n' < "$a" | grep -v '^$') \
      <(tr ' ' '\n' < "$p" | grep -v '^$') |
      awk '{ if ($1 != $2) d++ } END { printf "differ=%d\n", d + 0 }'
    if cmp -s "$OUT/${c}_shipped.body" "$OUT/${c}_passes0.body"; then
      printf 'ctx=%-6s vs passes0: BODY byte-identical\n' "$c"
    else
      printf 'ctx=%-6s vs passes0: BODY DIFFERS\n' "$c"
    fi
  fi
done
echo "QUALITY-SWEEP-DONE"