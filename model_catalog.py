import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from settings_store import get_models_dir

GIB = 1024 ** 3


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
    model_type: str = ""
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


def get_available_memory_bytes() -> int:
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return 0
    page = 4096
    free = inactive = speculative = purgeable = compressed = 0
    for line in out.splitlines():
        m = re.match(r"Pages ([^:]+):\s+(\d+)\.", line.strip())
        if m:
            k = m.group(1).strip().lower()
            v = int(m.group(2))
            if k == "free":
                free = v
            elif k == "inactive":
                inactive = v
            elif k == "speculative":
                speculative = v
            elif k == "purgeable":
                purgeable = v
            elif "compressed" in k:
                compressed = v
    return max(0, (free + inactive + speculative + purgeable + compressed) * page)


def get_total_memory_bytes() -> int:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip())
    except Exception:
        return 0


def estimate_model_memory_bytes(model_size_bytes: int, options: dict) -> int:
    return model_size_bytes + GIB


def get_model_type(model_name: str) -> str:
    config_path = get_models_dir() / model_name / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            return config.get("model_type", "")
        except Exception:
            pass
    return ""


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

    available_memory = get_available_memory_bytes()
    models: list[ModelInfo] = []

    for item in sorted(models_dir.iterdir()):
        if not item.is_dir():
            continue
        size_bytes = get_model_size_bytes(item.name)
        caps = get_model_capabilities(item.name)
        mt = get_model_type(item.name)
        estimated = estimate_model_memory_bytes(size_bytes, options)
        caps.estimated_bytes = estimated
        caps.estimated_memory = format_bytes_gib(estimated)
        caps.fits_memory = available_memory is None or estimated <= available_memory
        caps.available_memory = (
            format_bytes_gib(available_memory) if available_memory is not None else "unknown"
        )
        models.append(ModelInfo(
            name=item.name,
            size_bytes=size_bytes,
            size_gib=format_bytes_gib(size_bytes),
            model_type=mt,
            capabilities=caps,
        ))

    models.sort(key=lambda m: m.capabilities.estimated_bytes)
    return models
