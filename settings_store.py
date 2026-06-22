import json
from pathlib import Path


DEFAULT_MODELS_DIR = Path.home() / ".omlx" / "models"
ENV_PATH = Path.cwd() / ".env"


def get_models_dir() -> Path:
    try:
        if ENV_PATH.exists():
            for line in ENV_PATH.read_text().splitlines():
                line = line.strip()
                if line.startswith("MODELS_DIR="):
                    val = line[len("MODELS_DIR="):].strip().strip("\"'")
                    if val:
                        return Path(val).expanduser()
    except Exception:
        pass
    return DEFAULT_MODELS_DIR


def set_models_dir(path: str | Path) -> None:
    path = Path(path).expanduser().resolve()
    lines = []
    found = False
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("MODELS_DIR="):
                lines.append(f"MODELS_DIR={path}")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"MODELS_DIR={path}")
    ENV_PATH.write_text("\n".join(lines) + "\n")


SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help menu",
    "/clear": "Reset conversation",
    "/models": "Open model picker",
    "/options": "Change sampling, MTP, KV cache, and performance knobs",
    "/personality": "Switch personality: /personality <name>",
    "/search": "Web search: /search <query>",
    "/research": "Deep research: /research <topic>",
    "/think": "Send with thinking tags: /think <message>",
    "/unload": "Free GPU memory: /unload <pct>",
    "/memory": "Show GPU memory usage",
    "/mtp": "Toggle multi-token prediction",
}

NO_ARG_COMMANDS: set[str] = {
    "/help",
    "/clear",
    "/models",
    "/options",
    "/personality",
    "/memory",
    "/mtp",
}

DEFAULT_MODEL_OPTIONS = {
    "temp": 0.7,
    "top_p": 0.8,
    "top_k": 0,
    "max_tokens": 16384,
    "max_kv_size": None,
    "turbo_kv_bits": 3,
    "turbo_fp16_layers": 2,
    "mtp": True,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "enable_thinking": False,
    "prefill_step_size": 128,
}

OPTIONS_STATE_PATH = Path.home() / ".omlx" / "chat_options.json"

MODEL_CONFIGS_PATH = Path.home() / ".omlx" / "model_configs.json"

from textual_ui.personas import PERSONALITIES

PERSONALITY_CHOICES = list(PERSONALITIES.keys())

OPTION_SPECS = [
    {
        "key": "temp",
        "label": "Temperature",
        "choices": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        "description": "Lower is tighter. Higher is weirder.",
    },
    {
        "key": "top_p",
        "label": "Top-p",
        "choices": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "description": "Nucleus sampling cutoff.",
    },
    {
        "key": "top_k",
        "label": "Top-k",
        "choices": [0, 20, 40, 60, 80, 100, 120, 200],
        "description": "0 disables it. Higher keeps more candidates.",
    },
    {
        "key": "min_p",
        "label": "Min-p",
        "choices": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "description": "Minimum token probability (scaled by top token). 0 = off.",
    },
    {
        "key": "repetition_penalty",
        "label": "Rep penalty",
        "choices": [1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.5, 2.0],
        "description": "Penalizes repeated tokens. 1.0 = off.",
    },
    {
        "key": "max_tokens",
        "label": "Max tokens",
        "choices": [512, 1024, 2048, 4096, 8192, 16384, 32768],
        "description": "Response length cap.",
    },
    {
        "key": "max_kv_size",
        "label": "Context / KV",
        "choices": [None, 4096, 8192, 16384, 32768, 65536],
        "description": "KV cache cap. Bigger eats more RAM.",
    },
    {
        "key": "mtp",
        "label": "MTP",
        "choices": [True, False],
        "description": "Speculative decoding toggle.",
    },
    {
        "key": "turbo_kv_bits",
        "label": "Turbo KV bits",
        "choices": [None, 1, 2, 3, 4],
        "description": "KV compression. Less RAM, more compromise.",
    },
    {
        "key": "turbo_fp16_layers",
        "label": "FP16 layers",
        "choices": [0, 1, 2, 4, 8],
        "description": "Higher keeps more layers in FP16.",
    },
    {
        "key": "enable_thinking",
        "label": "Thinking",
        "choices": [True, False],
        "description": "Enable thinking tags for supported models.",
    },
    {
        "key": "prefill_step_size",
        "label": "Prefill step",
        "choices": [32, 64, 128, 256, 512, 1024],
        "description": "Step size for prompt prefill processing.",
    },
]


def normalize_model_options(options: dict[str, object] | None) -> dict[str, object]:
    normalized = dict(DEFAULT_MODEL_OPTIONS)
    if not options:
        return normalized
    for spec in OPTION_SPECS:
        key = spec["key"]
        if key not in options:
            continue
        value = options[key]
        if key == "top_p" and isinstance(value, (int, float)) and not (0.0 <= value <= 1.0):
            continue
        if key == "min_p" and isinstance(value, (int, float)) and not (0.0 <= value <= 1.0):
            continue
        if key == "temp" and isinstance(value, (int, float)) and value < 0.0:
            continue
        if key == "top_k" and isinstance(value, int) and value < 0:
            continue
        if key == "repetition_penalty" and isinstance(value, (int, float)) and value < 0.0:
            continue
        if key in ("max_tokens", "prefill_step_size") and isinstance(value, int) and value <= 0:
            continue
        if key == "turbo_kv_bits" and value is not None and isinstance(value, (int, float)) and not (1 <= value <= 8):
            continue
        normalized[key] = value
    return normalized


def _model_config_for(model_name: str) -> dict:
    return load_model_configs().get(model_name, {})


def get_model_options(model_name: str) -> dict[str, object]:
    return normalize_model_options(_model_config_for(model_name).get("options", {}))


def get_model_personality(model_name: str) -> str:
    return _model_config_for(model_name).get("personality", "default")


def save_model_config(model_name: str, options: dict, personality: str) -> None:
    configs = load_model_configs()
    configs[model_name] = {"options": normalize_model_options(options), "personality": personality}
    save_model_configs(configs)


def load_model_configs() -> dict:
    try:
        if MODEL_CONFIGS_PATH.exists():
            return json.loads(MODEL_CONFIGS_PATH.read_text())
    except Exception:
        pass
    return {}


def save_model_configs(configs: dict) -> None:
    MODEL_CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CONFIGS_PATH.write_text(json.dumps(configs, indent=2, sort_keys=True))
