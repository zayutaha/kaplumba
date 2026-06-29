# Kaplumba

<img src="logo.png" width="400" alt="Kaplumba logo">

A **terminal UI** for running large language models locally on Apple Silicon. Built on [MLX](https://github.com/ml-explore/mlx) and the [mlx-lm](https://github.com/ml-explore/mlx-lm) inference engine — supports 80+ model architectures with memory-efficient features that let you run heavy models on smaller Macs.

---

## Setup

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/#installation) (install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`).

```bash
git clone https://github.com/zayutaha/kaplumba.git
cd kaplumba
uv sync          # installs all dependencies
```

Models go in `~/.omlx/models/` (or set `MODELS_DIR` in a `.env` file).  
Config files (favorites, model options, etc.) are stored in `.kaplumba/` in the project root.

## Run

```bash
uv run python chat.py
```

---

## Features

### Terminal UI
Full-featured Textual-based chat interface with model picker, options dialogs, personality selector, keyboard shortcuts, and crash recovery.

### Model Picker
Scans the model directory for available models, shows architecture type, estimated memory footprint, and live free memory — refreshes every 5 seconds so you can see memory free up as you close other apps. Models that fit are highlighted in green, models that risk OOM show in red. Press `d` to edit the model directory path.

### TurboQuant KV Cache
Compresses the KV cache to 1–4 bits per element using Gaussian-optimized codebooks with Metal-accelerated quantize/dequant kernels. At 3-bit, the cache uses ~4.6× less memory than FP16 with negligible quality loss.

Works with MTP (Multi-Token Prediction) so speculative decoding benefits from the same compression.

### Mixed-Precision KV Cache
Keeps attention-critical layers in FP16 while compressing the rest. Configure how many layers stay at full precision.

### Multi-Token Prediction (MTP)
Speculative decoding that generates 2–3 tokens per forward pass instead of 1. Combined with TurboQuant, the compressed cache leaves room for MTP heads alongside the backbone model.

### Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help overlay |
| `/clear` | Reset conversation |
| `/models` | Open model picker |
| `/options` | Change temperature, top-p, top-k, min-p, repetition penalty, max tokens, KV size, MTP, turbo KV bits, FP16 layers, thinking, prefill step |
| `/personality` | Change system prompt personality |
| `/search <query>` | Web search — generates 3 queries, scrapes 3 pages, answers concisely |
| `/research <topic>` | Deep research — gathers ~8 sources via research agent, produces structured context for follow-up |
| `/think <message>` | Send with thinking tags enabled |
| `/memory` | Show GPU cache / peak memory |
| `/unload` | Offload model from GPU (preserves KV cache, conversation continues on next prompt) |
| `/mtp` | Toggle multi-token prediction on/off |

### Personality System
Configurable system prompts that persist per-model in `.kaplumba/model_configs.json`. Bundled personalities: default (direct, honest ally), historian (gritty narrative style). Switch mid-session with `/personality` or the menu.

### Options Dialogs
Graphical selector for every sampling and performance parameter. Per-model settings persist across sessions.

### Thinking Block Detection
Detects and strips Qwen `<think>...</think>` and Gemma `<|channel>...<channel|>` blocks. If the entire response is inside a thinking block (Gemma 4 behavior), extracts the inner content.

### LaTeX Rendering
Model output is processed through a comprehensive LaTeX→Unicode converter covering Greek, math operators, fractions, integrals, matrices, cases, fonts, accents — rendered inline in the terminal.

### Kaplumbebek (Mini Chat)
A popup chat sidebar toggled with `Ctrl+O` that maintains a **completely separate conversation context** from the main chat. Ask off-topic questions, test prompts, or explore ideas without polluting your main conversation history. Uses the same loaded model — no reload needed. Has its own KV cache, system prompt (`"You are a helpful assistant"`), and persistent history across the session.

### Web Search
`/search` generates 3 search queries from the model, searches DuckDuckGo, scrapes 3 relevant pages, and injects content into context for a concise answer.

### Research Agent
`/research` deploys a multi-step agent (plan → retrieve 8 pages → extract → structure) for deep topic exploration. The structured context is loaded into the conversation for follow-up Q&A.

### Phoenix Resilience
Kaplumba runs the model in an **isolated subprocess** with its own stdin/stdout protocol. If the model crashes (transient OOM, segfault, or cosmic ray), the UI stays up. It auto-retries loading **3 times** before showing a dialog. You can switch models without restarting the app. SIGINT is relayed reliably for clean interruption. Think of it as a bulletproof vest for your inference engine.

### LiveTune Sampling
Change temperature, top-p, top-k, max tokens, and prefill step size on the fly via `/options` — **no model reload needed**. The new values take effect immediately on the next generation. Cache-affecting options like TurboQuant bits still reload, but your daily knobs are instant.

### Message Selection & Copy
**Ctrl+click** any message bubble to instantly copy its text. **Ctrl+Alt+click** copies the raw unformatted text (LaTeX source). A brief gold flash confirms the copy.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Ctrl+C` | Quit |
| `Ctrl+R` | Reload model |
| `Ctrl+\` or `/help` | Help overlay |
| `Ctrl+O` | Kaplumbebek (mini chat popup) |
| `Esc` | Close overlay |

---

## Quick Start

```sh
python chat.py
```

On first launch, pick a model from `~/.omlx/models/` and start chatting.

Model directory: `~/.omlx/models/` — drop MLX-converted models here.

---

## Configuration

Per-model config at `.kaplumba/model_configs.json`:

```json
{
  "Llama-3.2-3B-Instruct-4bit": {
    "options": {
      "temp": 0.7,
      "top_p": 0.8,
      "max_tokens": 16384,
      "mtp": true,
      "turbo_kv_bits": 3,
      "prefill_step_size": 128
    },
    "personality": "default"
  }
}
```

---

## Project Structure

```
./
├── chat.py                     # Entry point
├── tui_main.py                 # Textual ChatUI application
├── textual_ui/                 # TUI widgets
│   ├── personas.py             # Personality definitions
│   ├── latex.py                # LaTeX rendering
│   ├── styles.py               # CSS
│   └── widgets/
│       ├── chat_input.py
│       ├── model_picker.py
│       ├── options_selector.py
│       ├── personality_selector.py
│       ├── chat_selector.py
│       ├── loading_spinner.py
│       ├── slash_command_menu.py
│       ├── model_config_editor.py
│       └── kaplumbebek_popup.py
├── orchestrator.py             # UI ↔ model coordination
├── model_lifecycle.py          # Subprocess model runner
├── model_interface.py          # Async IPC protocol
├── model_catalog.py            # Model discovery & memory estimation
├── settings_store.py           # .kaplumba/ persistence
├── conversation_engine.py      # Streaming & thinking block handling
├── simple_markdown.py          # Markdown → Rich converter
├── latex_parser.py             # LaTeX → Unicode
├── scripts/                    # TurboQuant helpers
│   ├── turboquant_quantize_run.py
│   ├── turboquant_test_gen.py
│   └── turboquant_validate_weights.py
│
└── mlx_lm/                     # Inference engine (mlx-lm based)
    ├── chat.py                 # Chat REPL with slash commands, search, research
    ├── __init__.py             # Public API
    ├── generate.py             # TurboQuant, MTP, prefill_step_size
    ├── utils.py                # TurboQuant-aware loading
    ├── models/
    │   ├── turboquant_*.py     # TurboQuant Metal kernels
    │   └── mixed_quant_cache.py
    ├── quant/
    │   └── turboquant_weights.py
    ├── research_agent/         # Autonomous research framework
    ├── web_search.py           # DuckDuckGo + scraping
    ├── disk_cache.py           # Persistent prompt cache
    └── ... (80+ model architectures, tuner, server, tool parsers, etc.)
```

---

## Underlying Engine

The inference engine supports everything you'd expect from `mlx-lm`:

- **80+ model architectures**: LLaMA, Mistral, Qwen 2/3/3.5, DeepSeek V2/V3/V3.2, Gemma 1-4, Phi, Mixtral, Cohere, OLMo, Mamba, RWKV, DBRX, Jamba, and many more
- **CLI tools**: `mlx_lm.generate`, `mlx_lm.convert`, `mlx_lm.server`, `mlx_lm.lora`, `mlx_lm.evaluate`, `mlx_lm.benchmark`, and 15+ other commands
- **Python API**: `load()`, `generate()`, `stream_generate()`, `batch_generate()`, `convert()`
- **Quantization**: AWQ, GPTQ, DWQ, dynamic quantization, TurboQuant weights
- **Fine-tuning**: LoRA, DoRA, full fine-tuning with Muon optimizer, gradient checkpointing
- **Distributed inference**: tensor/pipeline parallelism, peer-to-peer weight sharing
- **HTTP server**: OpenAI-compatible, streaming, tool calling, multi-model, prompt caching

---

## License

MIT
