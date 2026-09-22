#!/usr/bin/env python3
"""Kaplumba bench harness (spec/omlx-exl3-engine T0.1, R7.1).

Measures TTFT (turn1/turn2), prefill pp/s at 1k/2k/8k, plain tg/s, peak
memory (phys_footprint + Metal peak), and prints the EXL3 roofline
(15.5 GB -> ~17.6 tok/s at 273 GB/s). Emits a stable JSON report
(schema kaplumba-bench/1) on stdout and optionally to --json PATH.

Backends:
  ponyexl3  baseline runtime (reference/PonyExl3, EXL3 3.0 bpw artifact)
  mock      deterministic fabrications for schema/tests (no model needed)
  omlx      reserved for engine integration (T1.x; stock omlx cannot
            load exl3 — see implementation_notes DV1)

Usage:
  .venv-baseline/bin/python scripts/bench.py MODEL [--json out.json]
  .venv-baseline/bin/python scripts/bench.py --backend mock --pp 1024,2048,8192
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "kaplumba-bench/1"
PROMPT_TEXT = (
    "The sliding window attention mechanism keeps a fixed local context while "
    "full attention layers preserve long-range dependencies across the "
    "sequence. Prefill cost grows with prompt length, decode cost with weight "
    "bytes moved per token, and the roofline of a 15.5 GB EXL3 checkpoint at "
    "273 GB/s is the ceiling plain single-user decode can never pass. "
)


class _RusageInfoV4(ctypes.Structure):
    """rusage_info_v4 layout from /usr/include/sys/resource.h (omlx-compatible)."""

    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
        ("ri_cpu_time_qos_default", ctypes.c_uint64),
        ("ri_cpu_time_qos_maintenance", ctypes.c_uint64),
        ("ri_cpu_time_qos_background", ctypes.c_uint64),
        ("ri_cpu_time_qos_utility", ctypes.c_uint64),
        ("ri_cpu_time_qos_legacy", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_initiated", ctypes.c_uint64),
        ("ri_cpu_time_qos_user_interactive", ctypes.c_uint64),
        ("ri_billed_system_time", ctypes.c_uint64),
        ("ri_serviced_system_time", ctypes.c_uint64),
        ("ri_logical_writes", ctypes.c_uint64),
        ("ri_lifetime_max_phys_footprint", ctypes.c_uint64),
        ("ri_instructions", ctypes.c_uint64),
        ("ri_cycles", ctypes.c_uint64),
        ("ri_billed_energy", ctypes.c_uint64),
        ("ri_serviced_energy", ctypes.c_uint64),
        ("ri_interval_max_phys_footprint", ctypes.c_uint64),
        ("ri_runnable_time", ctypes.c_uint64),
    ]


_RUSAGE_INFO_V4 = 4

_proc_pid_rusage = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        _proc_pid_rusage = _libproc.proc_pid_rusage
        _proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _proc_pid_rusage.restype = ctypes.c_int
    except OSError:
        _proc_pid_rusage = None


def roofline(weights_gb: float, bandwidth_gbps: float) -> dict[str, float]:
    if weights_gb <= 0 or bandwidth_gbps <= 0:
        raise ValueError("weights_gb and bandwidth_gbps must be positive")
    return {
        "weights_gb": weights_gb,
        "bandwidth_gbps": bandwidth_gbps,
        "tg_tps_ceiling": round(bandwidth_gbps / weights_gb, 2),
    }


def parse_pp(spec: str) -> list[int]:
    try:
        sizes = [int(x.strip()) for x in spec.split(",") if x.strip()]
    except ValueError as exc:
        raise SystemExit(f"invalid --pp: {spec!r}") from exc
    if not sizes or any(n <= 0 for n in sizes):
        raise SystemExit("--pp sizes must be positive integers")
    return sizes


def phys_footprint() -> tuple[int, int]:
    """(current, lifetime_max) phys_footprint in bytes via libproc rusage v4."""
    if _proc_pid_rusage is None:
        return 0, 0
    info = _RusageInfoV4()
    if _proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return 0, 0
    return int(info.ri_phys_footprint), int(info.ri_lifetime_max_phys_footprint)


def metal_peak() -> int | None:
    try:
        import mlx.core as mx

        if hasattr(mx, "metal") and hasattr(mx.metal, "get_peak_memory"):
            return int(mx.metal.get_peak_memory())
    except Exception:
        return None
    return None


def host_info() -> dict[str, Any]:
    mem = 0
    cpu = ""
    try:
        mem = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    except Exception:
        pass
    try:
        cpu = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu,
        "mem_bytes": mem,
        "cpu_count": os.cpu_count(),
    }


def versions() -> dict[str, str]:
    from importlib import metadata

    out: dict[str, str] = {}
    for pkg in ("mlx", "mlx-lm", "ponyexl3", "omlx"):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = "not-installed"
    return out


def _load_prompt_text(path: str | None) -> str:
    if path is not None:
        p = Path(path).expanduser()
        if not p.is_file():
            raise SystemExit(f"prompt file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if not text.strip():
            raise SystemExit(f"prompt file is empty: {p}")
        return text
    return PROMPT_TEXT


def run_ponyexl3(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from ponyexl3.cli._generate_common import (
        GenerateStack,
        build_prefill_prompt_ids,
        collect_eos_token_ids,
        load_generate_stack,
        max_position_embeddings,
        require_metal,
        validate_exl3_model_dir,
    )
    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import describe, load_model
    from mlx_lm.utils import load_tokenizer

    validate_exl3_model_dir(args.model)
    require_metal()
    prompt_text = _load_prompt_text(args.prompt_file)

    ns = argparse.Namespace(
        model=args.model,
        engine=args.engine,
        no_warm=args.no_warm,
        quiet=True,
        dflash=None,
        dflash_quant="w8",
        eagle3=None,
        mtp=args.mtp,
        draft=3,
        draft_w4=False,
    )
    tic = time.perf_counter()
    stack: GenerateStack = load_generate_stack(ns)
    load_s = time.perf_counter() - tic

    tokenizer = stack.tokenizer
    extra_eos = stack.extra_eos
    max_ctx = max_position_embeddings(stack.config)

    def gen(ids: list[int], n: int) -> tuple[float, Any, float]:
        first: dict[str, float | None] = {"t": None}

        def on_segment(_seg: str) -> None:
            if first["t"] is None:
                first["t"] = time.perf_counter()

        t0 = time.perf_counter()
        _, stats = generate_text(
            stack.model,
            tokenizer,
            prompt="",
            prompt_ids=ids,
            max_tokens=n,
            temp=args.temp,
            prefill_chunk=args.prefill_chunk,
            use_chat_template=False,
            extra_eos=extra_eos,
            mtp=stack.mtp,
            num_draft=stack.draft,
            lookup=False,
            eagle3=stack.eagle3,
            dflash=stack.dflash,
            max_context=max_ctx,
            on_segment=on_segment,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        ttft_ms = (
            (first["t"] - t0) * 1000.0
            if first["t"] is not None
            else stats.prefill_s * 1000.0
        )
        if hasattr(mx, "synchronize"):
            mx.synchronize()
        return ttft_ms, stats, wall_ms

    if args.warmup:
        warm_ids = build_prefill_prompt_ids(prompt_text, 128, tokenizer, raw=True)
        gen(warm_ids, 4)

    turns: list[dict[str, Any]] = []
    for turn in range(1, args.turns + 1):
        ids = build_prefill_prompt_ids(prompt_text, args.ttft_tokens, tokenizer, raw=True)
        ttft_ms, stats, wall_ms = gen(ids, args.ttft_gen)
        turns.append(
            {
                "turn": turn,
                "prompt_tokens": stats.prompt_tokens,
                "prefill_ms": round(stats.prefill_s * 1000.0, 3),
                "ttft_ms": round(ttft_ms, 3),
                "wall_ms": round(wall_ms, 3),
            }
        )

    prefill_rows: list[dict[str, Any]] = []
    for pp in args.pp:
        ids = build_prefill_prompt_ids(prompt_text, pp, tokenizer, raw=True)
        _, stats, _ = gen(ids, args.prefill_gen)
        prefill_rows.append(
            {
                "prompt_tokens": stats.prompt_tokens,
                "prefill_s": round(stats.prefill_s, 4),
                "pp_tps": round(
                    stats.prompt_tokens / stats.prefill_s, 2
                )
                if stats.prefill_s
                else 0.0,
                "gen_tokens": stats.gen_tokens,
            }
        )

    decode: dict[str, Any]
    if args.gen > 0:
        ids = build_prefill_prompt_ids(prompt_text, args.tg_prompt, tokenizer, raw=True)
        _, stats, _ = gen(ids, args.gen)
        decode = {
            "prompt_tokens": stats.prompt_tokens,
            "gen_tokens": stats.gen_tokens,
            "decode_s": round(stats.decode_s, 4),
            "tg_tps": round(stats.gen_tokens / stats.decode_s, 2)
            if stats.decode_s
            else 0.0,
            "finish_reason": stats.finish_reason,
        }
    else:
        decode = {
            "prompt_tokens": args.tg_prompt,
            "gen_tokens": 0,
            "decode_s": 0.0,
            "tg_tps": 0.0,
            "finish_reason": "skipped",
        }

    fp, fp_max = phys_footprint()
    mem = {
        "phys_footprint_bytes": fp,
        "lifetime_max_phys_footprint_bytes": fp_max,
        "metal_peak_bytes": metal_peak(),
        "resident_model": describe(stack.model),
    }
    del stack
    return {
        "load_s": round(load_s, 3),
        "ttft": turns,
        "prefill": prefill_rows,
        "decode": decode,
        "mem": mem,
        "engine": args.engine,
        "mtp": args.mtp,
    }


def run_mock(args: argparse.Namespace) -> dict[str, Any]:
    roof = roofline(args.weights_gb, args.bandwidth)
    turns = [
        {
            "turn": t,
            "prompt_tokens": args.ttft_tokens,
            "prefill_ms": round(args.ttft_tokens / 2.0, 3),
            "ttft_ms": round(args.ttft_tokens / 2.0 + 1.5, 3),
            "wall_ms": round(args.ttft_tokens / 2.0 + 8.0, 3),
        }
        for t in range(1, args.turns + 1)
    ]
    prefill = [
        {
            "prompt_tokens": pp,
            "prefill_s": round(pp / 2000.0, 4),
            "pp_tps": 2000.0,
            "gen_tokens": args.prefill_gen,
        }
        for pp in args.pp
    ]
    decode = {
        "prompt_tokens": args.tg_prompt,
        "gen_tokens": args.gen,
        "decode_s": round(args.gen / 17.5, 4) if args.gen else 0.0,
        "tg_tps": 17.5 if args.gen else 0.0,
        "finish_reason": "length" if args.gen else "skipped",
    }
    return {
        "load_s": 1.0,
        "ttft": turns,
        "prefill": prefill,
        "decode": decode,
        "mem": {
            "phys_footprint_bytes": 17_500_000_000,
            "lifetime_max_phys_footprint_bytes": 17_600_000_000,
            "metal_peak_bytes": 16_900_000_000,
            "resident_model": "mock",
        },
        "engine": "mock",
        "mtp": args.mtp,
        "roofline_note": round(
            100.0 * decode["tg_tps"] / roof["tg_tps_ceiling"], 1
        )
        if decode["tg_tps"]
        else 0.0,
    }


def run_omlx(_args: argparse.Namespace) -> dict[str, Any]:
    raise SystemExit(
        "omlx backend not available yet: stock omlx has no exl3 loader "
        "(implementation_notes DV1); lands with engine integration (T1.x+)"
    )


def assemble_report(
    args: argparse.Namespace, payload: dict[str, Any]
) -> dict[str, Any]:
    roof = roofline(args.weights_gb, args.bandwidth)
    tg = payload["decode"]["tg_tps"]
    return {
        "schema": SCHEMA,
        "backend": args.backend,
        "model": str(args.model),
        "temp": args.temp,
        "pp_sizes": args.pp,
        "roofline": roof,
        "tg_pct_of_roofline": round(100.0 * tg / roof["tg_tps_ceiling"], 1)
        if tg
        else 0.0,
        "host": host_info(),
        "versions": versions(),
        **payload,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="bench.py",
        description="Kaplumba bench harness: TTFT/pp/tg/mem + EXL3 roofline (T0.1).",
    )
    ap.add_argument("model", nargs="?", default="mock-model")
    ap.add_argument("--backend", choices=("ponyexl3", "mock", "omlx"), default="ponyexl3")
    ap.add_argument("--pp", default="1024,2048,8192")
    ap.add_argument("--gen", type=int, default=128, help="decode tokens for tg/s")
    ap.add_argument("--prefill-gen", type=int, default=8)
    ap.add_argument("--ttft-tokens", type=int, default=1024)
    ap.add_argument("--ttft-gen", type=int, default=16)
    ap.add_argument("--tg-prompt", type=int, default=1024)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--prefill-chunk", type=int, default=2048)
    ap.add_argument("--engine", default="exl3", choices=("exl3", "fold16"))
    ap.add_argument("--mtp", default="off", choices=("off", "auto"))
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--warmup", action="store_true")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--json", default=None, help="also write report to PATH")
    ap.add_argument("--weights-gb", type=float, default=15.5)
    ap.add_argument("--bandwidth", type=float, default=273.0)
    args = ap.parse_args(argv)

    args.pp = parse_pp(args.pp)
    if args.turns < 1:
        raise SystemExit("--turns must be >= 1")
    if args.gen < 0:
        raise SystemExit("--gen must be >= 0")

    runners = {"ponyexl3": run_ponyexl3, "mock": run_mock, "omlx": run_omlx}
    payload = runners[args.backend](args)
    report = assemble_report(args, payload)

    text = json.dumps(report, indent=2, sort_keys=True)
    if args.json:
        Path(args.json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
