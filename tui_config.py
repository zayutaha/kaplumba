import json
from pathlib import Path


SLASH_COMMANDS: dict[str, str] = {
    "/clear": "Clear the conversation, but keep the Kaplumba welcome screen.",
    "/models": "Open the model picker and switch models safely.",
    "/options": "Open launch options for MTP, context, sampling, and cache knobs.",
    "/personality": "Open the personality picker: default, doctor, or historian.",
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
}

OPTIONS_STATE_PATH = Path.home() / ".omlx" / "chat_options.json"

MODEL_CONFIGS_PATH = Path.home() / ".omlx" / "model_configs.json"

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
        "choices": [None, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "description": "KV compression. Less RAM, more compromise.",
    },
    {
        "key": "turbo_fp16_layers",
        "label": "FP16 layers",
        "choices": [0, 1, 2, 4, 8],
        "description": "Higher keeps more layers in FP16.",
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
        if value in spec["choices"]:
            normalized[key] = value
    return normalized


def load_saved_model_options() -> dict[str, object]:
    try:
        with open(OPTIONS_STATE_PATH) as f:
            data = json.load(f)
    except Exception:
        return dict(DEFAULT_MODEL_OPTIONS)
    return normalize_model_options(data)


def save_model_options(options: dict[str, object]) -> None:
    OPTIONS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OPTIONS_STATE_PATH, "w") as f:
        json.dump(normalize_model_options(options), f, indent=2, sort_keys=True)


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
