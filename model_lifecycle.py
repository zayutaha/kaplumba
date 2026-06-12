import asyncio
import os
import signal
from pathlib import Path

from settings_store import load_model_configs

TUI_PROMPT_MARKER = "<<KAPLUMBA_READY>> "


def build_model_command(model_path: str, options: dict, system_prompt: str) -> list[str]:
    model_name = Path(model_path).name
    configs = load_model_configs()
    model_cfg = configs.get(model_name, {})
    opts = dict(options)
    if model_cfg.get("options"):
        opts.update(model_cfg["options"])

    cmd = [
        "uv", "run", "python", "-m", "mlx_lm.chat",
        "--model", model_path,
        "--prompt-marker", TUI_PROMPT_MARKER,
        "--temp", str(opts["temp"]),
        "--top-p", str(opts["top_p"]),
        "--top-k", str(opts["top_k"]),
        "--max-tokens", str(opts["max_tokens"]),
        "--chat-template-args", '{"enable_thinking":false}',
        "--system-prompt", system_prompt,
    ]
    if opts["mtp"]:
        cmd.append("--mtp")
    if opts["max_kv_size"] is not None:
        cmd.extend(["--max-kv-size", str(opts["max_kv_size"])])
    if opts["turbo_kv_bits"] is not None:
        cmd.extend(["--turbo-kv-bits", str(int(opts["turbo_kv_bits"]))])
    if opts["turbo_fp16_layers"] is not None:
        cmd.extend(["--turbo-fp16-layers", str(opts["turbo_fp16_layers"])])
    if opts.get("prefill_step_size") is not None:
        cmd.extend(["--prefill-step-size", str(opts["prefill_step_size"])])
    return cmd


class ModelRunner:
    def __init__(self):
        self.proc = None
        self.proc_pid = None
        self.proc_pgid = None

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self, model_path: str, options: dict, system_prompt: str) -> bool:
        cmd = build_model_command(model_path, options, system_prompt)
        stderr_log = Path(f"/tmp/mlx_lm_chat_{Path(model_path).name}.log")

        try:
            self.proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=open(stderr_log, "w"),
                env={**os.environ, "PYTHONPATH": os.getcwd()},
                start_new_session=True,
            )
        except Exception:
            return False

        self.proc_pid = self.proc.pid
        self.proc_pgid = self.proc.pid

        buf = await self._read_until_prompt(timeout=60)
        if buf is None:
            return False

        token_count = 0
        buf = ""
        start = asyncio.get_event_loop().time()

        try:
            self.proc.stdin.write(b"test\n")
            await self.proc.stdin.drain()
        except Exception:
            return False

        while token_count < 4:
            if asyncio.get_event_loop().time() - start > 30:
                return False
            try:
                chunk = await asyncio.wait_for(self.proc.stdout.read(256), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                return False
            buf += chunk.decode(errors="ignore")
            token_count = len(buf.split())
            if buf.endswith(TUI_PROMPT_MARKER):
                break

        # Wait for the test generation to finish and the subprocess to
        # return to the input() prompt.  No Ctrl-D byte is written (it
        # would leak into stdin and contaminate the first real message).
        try:
            await self._read_until_prompt(timeout=10)
        except Exception:
            pass

        return True

    async def send(self, text: str) -> bool:
        try:
            self.proc.stdin.write((text + "\n").encode())
            await self.proc.stdin.drain()
            return True
        except Exception:
            return False

    async def interrupt(self):
        if self.running:
            try:
                os.killpg(self.proc_pgid, signal.SIGINT)
            except Exception:
                try:
                    self.proc.send_signal(signal.SIGINT)
                except Exception:
                    pass

    async def stop(self):
        proc = self.proc
        pgid = self.proc_pgid
        self.proc = None
        self.proc_pid = None
        self.proc_pgid = None

        if not proc or proc.returncode is not None:
            return

        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                if pgid:
                    os.killpg(pgid, sig)
                else:
                    proc.send_signal(sig)
            except ProcessLookupError:
                break
            except Exception:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=10 if sig == signal.SIGTERM else 5)
                break
            except asyncio.TimeoutError:
                continue

    async def _read_until_prompt(self, timeout=60):
        buf = ""
        start = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start > timeout:
                return None
            try:
                chunk = await asyncio.wait_for(self.proc.stdout.read(256), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                return None
            buf += chunk.decode(errors="ignore")
            if buf.endswith(TUI_PROMPT_MARKER):
                return buf[: -len(TUI_PROMPT_MARKER)]


class ModelOrchestrator:
    def __init__(self):
        self.runner = ModelRunner()
        self.crash_count = 0
        self.max_crashes = 3
        self.reloading = False

    @property
    def running(self) -> bool:
        return self.runner.running

    async def start_model(self, model_name: str, options: dict, system_prompt: str) -> bool:
        model_path = str(Path.home() / ".omlx" / "models" / model_name)
        ok = await self.runner.start(model_path, options, system_prompt)
        if ok:
            self.crash_count = 0
            return True
        self.crash_count += 1
        return False

    async def stop(self):
        await self.runner.stop()

    async def send(self, text: str) -> bool:
        return await self.runner.send(text)

    async def interrupt(self):
        await self.runner.interrupt()

    async def read_until_prompt(self, timeout=60):
        return await self.runner._read_until_prompt(timeout=timeout)

    def has_crashed_too_many(self) -> bool:
        return self.crash_count >= self.max_crashes

    def clear_crashes(self):
        self.crash_count = 0