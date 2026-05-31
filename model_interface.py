import asyncio
from typing import AsyncIterator, Protocol

from model_lifecycle import ModelRunner


class FakeModelPort:
    def __init__(self, chunks: list[str] | None = None):
        self.running = True
        self._chunks = chunks or ["Hello", " world", "!"]
        self._sent: list[str] = []

    async def start(self, model_path: str, options: dict, system_prompt: str) -> bool:
        self.running = True
        return True

    async def send_message(self, text: str) -> AsyncIterator[str]:
        self._sent.append(text)
        for chunk in self._chunks:
            yield chunk

    async def send_command(self, text: str, timeout: int = 60) -> str | None:
        self._sent.append(text)
        return None

    async def interrupt(self) -> None:
        pass

    async def stop(self) -> None:
        self.running = False


class ModelPort(Protocol):
    @property
    def running(self) -> bool: ...

    async def start(self, model_path: str, options: dict, system_prompt: str) -> bool:
        ...

    async def send_message(self, text: str) -> AsyncIterator[str]:
        ...

    async def send_command(self, text: str, timeout: int = 60) -> str | None:
        ...
    
    async def interrupt(self) -> None:
        ...

    async def stop(self) -> None:
        ...


class MLXSubprocessAdapter:
    TUI_PROMPT_MARKER = "<<KAPLUMBA_READY>> "

    def __init__(self):
        self._runner = ModelRunner()
        self._interrupted = False

    @property
    def running(self) -> bool:
        return self._runner.running

    async def start(
        self, model_path: str, options: dict, system_prompt: str
    ) -> bool:
        return await self._runner.start(model_path, options, system_prompt)

    async def send_message(self, text: str) -> AsyncIterator[str]:
        self._interrupted = False
        text = " ".join(text.split("\n"))

        # Drain any stale stdout data left from a previous interrupted read
        # so we don't confuse old marker chunks with the new response.
        await self._drain_stale_stdout()

        if not await self._runner.send(text):
            return

        marker = self.TUI_PROMPT_MARKER
        while True:
            try:
                chunk = await asyncio.wait_for(
                    self._runner.proc.stdout.read(256), timeout=0.05
                )
            except asyncio.TimeoutError:
                if self._interrupted:
                    break
                continue
            except Exception:
                break

            if not chunk:
                break

            decoded = chunk.decode(errors="ignore")
            marker_pos = decoded.find(marker)
            if marker_pos >= 0:
                yield decoded[:marker_pos]
                return

            yield decoded

        if self._interrupted:
            remaining = await self._read_until_prompt(timeout=10)
            if remaining:
                yield remaining

    async def _drain_stale_stdout(self):
        """Read and discard any data already in stdout up to the prompt marker.
        Uses a short timeout so we don't block if nothing is queued.
        """
        marker = self.TUI_PROMPT_MARKER
        try:
            while True:
                chunk = await asyncio.wait_for(
                    self._runner.proc.stdout.read(256), timeout=0.01
                )
                if not chunk:
                    break
                decoded = chunk.decode(errors="ignore")
                if marker in decoded:
                    break  # consumed up to prompt marker, stop draining
        except asyncio.TimeoutError:
            pass  # nothing stale to drain

    async def send_command(self, text: str, timeout: int = 60) -> str | None:
        await self._runner.send(text)
        return await self._read_until_prompt(timeout=timeout)

    async def interrupt(self) -> None:
        self._interrupted = True
        await self._runner.interrupt()
        # Don't drain stdout here — send_message's read loop is still running
        # and will handle the remaining output when it detects the interrupt.

    async def stop(self) -> None:
        await self._runner.stop()

    async def _read_until_prompt(self, timeout: int = 60):
        buf = ""
        start = asyncio.get_event_loop().time()
        marker = self.TUI_PROMPT_MARKER
        while True:
            if asyncio.get_event_loop().time() - start > timeout:
                return None
            try:
                chunk = await asyncio.wait_for(
                    self._runner.proc.stdout.read(256), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            if not chunk:
                return None
            buf += chunk.decode(errors="ignore")
            if buf.endswith(marker):
                return buf[: -len(marker)]
