import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from settings_store import get_models_dir

GIB = 1024 ** 3
MEMORY_SAFETY_MARGIN_BYTES = int(1.5 * GIB)


@dataclass
class ModelCapabilities:
    vision: bool = False
    mtp: bool = False
    fits_memory: bool = True
    estimated_bytes: int = 0
    estimated_memory: str = "0.0 GB"
    total_memory: str = "unknown"
    available_memory: str = "unknown"


@dataclass
class ModelInfo:
    name: str
    size_bytes: int
    size_gib: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)


def format_bytes_gib(num_bytes: int) -> str:
    return f"{num_bytes / GIB:.1f} GB"


def get_model_size_bytes(model_name: str) -> int:
    model_dir = get_models_dir() / model_name
    if not model_dir.exists():
        return 0
    total = 0
    for f in model_dir.glob("**/*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def get_available_memory_bytes() -> int | None:
    try:
        output = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    page_size_match = re.search(r"page size of (\d+) bytes", output)
    if not page_size_match:
        return None
    page_size = int(page_size_match.group(1))
    counts: dict[str, int] = {}
    for line in output.splitlines():
        match = re.match(r"Pages ([^:]+):\s+(\d+)\.", line.strip())
        if match:
            counts[match.group(1).strip().lower()] = int(match.group(2))
    available = (
        counts.get("free", 0)
        + counts.get("inactive", 0)
        + counts.get("speculative", 0)
        + counts.get("purgeable", 0)
    ) * page_size - MEMORY_SAFETY_MARGIN_BYTES
    return max(0, available)


def get_total_memory_bytes() -> int | None:
    try:
        output = subprocess.check_output(["hostinfo"], text=True)
    except Exception:
        return None
    match = re.search(r"Primary memory available:\s+([0-9.]+)\s+gigabytes", output)
    if not match:
        return None
    return int(float(match.group(1)) * GIB)


def estimate_model_memory_bytes(model_size_bytes: int, options: dict) -> int:
    kv_tokens = int(options.get("max_kv_size") or 8192)
    turbo_kv_bits = options.get("turbo_kv_bits")
    turbo_fp16_layers = int(options.get("turbo_fp16_layers") or 0)
    mtp_enabled = bool(options.get("mtp"))
    runtime_overhead = max(int(1.5 * GIB), int(model_size_bytes * 0.12))
    kv_cache = int(0.75 * GIB * (kv_tokens / 8192))
    if turbo_kv_bits is None:
        kv_cache = int(kv_cache * 1.8)
    elif turbo_kv_bits == 4:
        kv_cache = int(kv_cache * 1.25)
    elif turbo_kv_bits == 3:
        kv_cache = int(kv_cache * 1.0)
    elif turbo_kv_bits == 2:
        kv_cache = int(kv_cache * 0.75)
    elif turbo_kv_bits == 1:
        kv_cache = int(kv_cache * 0.55)
    fp16_overhead = int(turbo_fp16_layers * 0.12 * GIB)
    mtp_overhead = int(0.6 * GIB) if mtp_enabled else 0
    estimated = model_size_bytes + runtime_overhead + kv_cache + fp16_overhead + mtp_overhead
    return max(estimated, model_size_bytes + GIB)


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    model_dir = get_models_dir() / model_name
    caps = ModelCapabilities()
    config_path = model_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
                caps.vision = "image_token_id" in config or "vision_config" in config
        except Exception:
            pass
    caps.mtp = (
        (model_dir / "preprocessor_config.json").exists()
        or (model_dir / "video_preprocessor_config.json").exists()
    )
    return caps


def list_models(options: dict) -> list[ModelInfo]:
    models_dir = get_models_dir()
    if not models_dir.exists():
        return []

    total_memory = get_total_memory_bytes()
    available_memory = get_available_memory_bytes()
    models: list[ModelInfo] = []

    for item in sorted(models_dir.iterdir()):
        if not item.is_dir():
            continue
        size_bytes = get_model_size_bytes(item.name)
        caps = get_model_capabilities(item.name)
        estimated = estimate_model_memory_bytes(size_bytes, options)
        caps.estimated_bytes = estimated
        caps.estimated_memory = format_bytes_gib(estimated)
        caps.fits_memory = available_memory is None or estimated <= available_memory
        caps.total_memory = (
            format_bytes_gib(total_memory) if total_memory is not None else "unknown"
        )
        caps.available_memory = (
            format_bytes_gib(available_memory) if available_memory is not None else "unknown"
        )
        models.append(ModelInfo(
            name=item.name,
            size_bytes=size_bytes,
            size_gib=format_bytes_gib(size_bytes),
            capabilities=caps,
        ))

    models.sort(key=lambda m: m.capabilities.estimated_bytes)
    return models
