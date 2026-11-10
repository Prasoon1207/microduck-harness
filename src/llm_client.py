"""Model-agnostic LLM client with mid-generation stop-sequence interception.

Supports:
- OpenAI (`OPENAI_API_KEY`)
- Google Gemini (`GEMINI_API_KEY` or `GOOGLE_API_KEY` via Gemini OpenAI-compat or native REST)
- Anthropic Claude (`ANTHROPIC_API_KEY` via Messages API with `stop_sequences`)
- OpenRouter (`OPENROUTER_API_KEY`)
- Custom OpenAI-compatible endpoints (`LLM_BASE_URL` + `LLM_API_KEY`, e.g. vLLM / Ollama)
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, Dict, List, Optional

import httpx


@dataclasses.dataclass
class GenerationStep:
    """Single intercepted generation segment from the language model."""

    text: str
    matched_stop: Optional[str]
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


def _reattach_stop_tag(text: str, stop_sequences: List[str], explicit_stop: Optional[str] = None) -> tuple[str, Optional[str]]:
    """Ensure the closing stop delimiter is preserved on the intercepted text segment.

    Most LLM APIs strip the matched `stop` string from the returned completion.
    Because our harness intercepts structured XML blocks (`<action_chunk>...</action_chunk>`
    or `<done>...</done>`), we detect the unclosed block (or explicit `stop_sequence`
    returned by the provider) and re-attach the closing delimiter cleanly.
    """
    for seq in stop_sequences:
        if text.rstrip().endswith(seq):
            return text.rstrip(), seq

    if explicit_stop and explicit_stop in stop_sequences:
        return text + explicit_stop, explicit_stop

    # Infer from open XML tag when provider omits `stop_reason` string (e.g. OpenAI finish_reason="stop")
    for seq in stop_sequences:
        if seq.startswith("</") and seq.endswith(">"):
            open_tag = "<" + seq[2:-1]
            last_open = text.rfind(open_tag)
            last_close = text.rfind(seq)
            if last_open != -1 and last_close < last_open:
                return text + seq, seq

    return text, None


def _load_dotenv_if_present() -> None:
    """Load key=value pairs from project `.env` without overriding explicit environment variables."""
    from pathlib import Path

    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip("\"'")
        if k and k not in os.environ:
            os.environ[k] = v


class ModelClient:
    """Thin, framework-free HTTP wrapper exposing stop-sequence generation across providers."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> None:
        _load_dotenv_if_present()
        norm_provider = None if (provider is None or provider.lower() in {"", "auto"}) else provider.lower()
        self.provider = norm_provider or self._auto_detect_provider()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or self._default_base_url(self.provider)).rstrip("/")
        self.api_key = api_key or self._resolve_api_key(self.provider)
        self.model = model or os.environ.get("LLM_MODEL") or self._default_model(self.provider)

    @staticmethod
    def _default_base_url(provider: str) -> str:
        return {
            "openai": "https://api.openai.com/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "custom": "http://localhost:11434/v1",
            "gemini": "https://generativelanguage.googleapis.com/v1beta",
            "anthropic": "https://api.anthropic.com/v1",
        }.get(provider, "https://api.openai.com/v1")

    @staticmethod
    def _auto_detect_provider() -> str:
        if os.environ.get("LLM_BASE_URL"):
            return "custom"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "anthropic"
        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            return "gemini"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"
        if os.environ.get("DEEPSEEK_API_KEY"):
            return "deepseek"
        raise RuntimeError(
            "No LLM API key detected. Set one of GEMINI_API_KEY, OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or LLM_BASE_URL, or pass --provider closed_loop."
        )

    @staticmethod
    def _resolve_api_key(provider: str) -> Optional[str]:
        if os.environ.get("LLM_API_KEY"):
            return os.environ["LLM_API_KEY"]
        mapping = {
            "openai": ["OPENAI_API_KEY"],
            "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "openrouter": ["OPENROUTER_API_KEY"],
            "deepseek": ["DEEPSEEK_API_KEY"],
            "custom": ["LLM_API_KEY", "OPENAI_API_KEY"],
        }
        for var in mapping.get(provider, []):
            if os.environ.get(var):
                return os.environ[var]
        return None

    @staticmethod
    def _default_model(provider: str) -> str:
        defaults = {
            "openai": "gpt-4.1",
            "gemini": "gemini-3.5-flash",
            "anthropic": "claude-sonnet-4-20250514",
            "openrouter": "anthropic/claude-sonnet-4",
            "deepseek": "deepseek-chat",
            "custom": "local-model",
        }
        return defaults.get(provider, "gpt-4.1")

    @staticmethod
    def _post_with_retries(
        url: str,
        *,
        json_payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, str]] = None,
        timeout: float = 90.0,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        for attempt in range(1, max_attempts + 1):
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, json=json_payload, headers=headers, params=params)
                if r.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                    time.sleep(1.5 * attempt)
                    continue
                if r.status_code >= 400:
                    raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:400]}")
                return r.json()
        raise RuntimeError(f"Exhausted retries calling {url}")

    def generate_until_stop(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
    ) -> GenerationStep:
        t0 = time.perf_counter()

        if self.provider == "anthropic":
            return self._call_anthropic(system_prompt, messages, stop_sequences, t0)
        if self.provider == "gemini":
            return self._call_gemini_native(system_prompt, messages, stop_sequences, t0)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stop": stop_sequences,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        data = self._post_with_retries(f"{self.base_url}/chat/completions", json_payload=payload, headers=headers)

        latency_s = time.perf_counter() - t0
        choice = data["choices"][0]
        raw_text = (choice.get("message") or {}).get("content") or ""
        explicit_stop = choice.get("stop_reason")
        full_text, matched_stop = _reattach_stop_tag(raw_text, stop_sequences, explicit_stop)
        usage = data.get("usage") or {}
        return GenerationStep(
            text=full_text,
            matched_stop=matched_stop,
            latency_s=latency_s,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=self.model,
        )

    def _call_anthropic(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
        t0: float,
    ) -> GenerationStep:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is missing.")
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system": system_prompt,
            "messages": messages,
            "stop_sequences": stop_sequences,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        data = self._post_with_retries("https://api.anthropic.com/v1/messages", json_payload=payload, headers=headers)

        latency_s = time.perf_counter() - t0
        blocks = data.get("content", [])
        raw_text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        explicit_stop = data.get("stop_sequence") if data.get("stop_reason") == "stop_sequence" else None
        full_text, matched_stop = _reattach_stop_tag(raw_text, stop_sequences, explicit_stop)
        usage = data.get("usage", {})
        return GenerationStep(
            text=full_text,
            matched_stop=matched_stop,
            latency_s=latency_s,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            model=self.model,
        )

    def _call_gemini_native(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        stop_sequences: List[str],
        t0: float,
    ) -> GenerationStep:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY / GOOGLE_API_KEY is missing.")
        contents = []
        for m in messages:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        model_id = self.model[len("models/") :] if self.model.startswith("models/") else self.model
        url = f"{self.base_url}/models/{model_id}:generateContent"

        def _extract_parts(resp_data: Dict[str, Any]) -> tuple[str, str]:
            cands = resp_data.get("candidates", [])
            if not cands:
                return "", ""
            pts = cands[0].get("content", {}).get("parts", [])
            vis = "".join(p.get("text", "") for p in pts if not p.get("thought", False))
            th = "".join(p.get("text", "") for p in pts if p.get("thought", False)).strip()
            return vis, th

        gen_cfg: Dict[str, Any] = {
            "temperature": self.temperature,
            "maxOutputTokens": max(self.max_tokens, 4096),
            "stopSequences": stop_sequences,
        }
        if model_id.startswith("gemini-"):
            gen_cfg["thinkingConfig"] = {"includeThoughts": True}

        payload: Dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        data = self._post_with_retries(url, json_payload=payload, params={"key": self.api_key})
        raw_text, native_thought = _extract_parts(data)

        # Generalist Gemini 3.x thinking models also match `stopSequences` inside internal `thought: true`
        # scratchpad parts. If the model mentioned literal `</action_chunk>` or `</done>` while thinking,
        # `raw_text` arrives empty; retry once without server-side stopSequences and slice at the first delimiter.
        if stop_sequences and not raw_text.strip():
            fallback_cfg: Dict[str, Any] = {
                "temperature": self.temperature,
                "maxOutputTokens": max(self.max_tokens, 4096),
            }
            if model_id.startswith("gemini-"):
                fallback_cfg["thinkingConfig"] = {"includeThoughts": True}
            fallback_payload: Dict[str, Any] = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": contents,
                "generationConfig": fallback_cfg,
            }
            data = self._post_with_retries(url, json_payload=fallback_payload, params={"key": self.api_key})
            raw_text, native_thought = _extract_parts(data)

        # Always enforce stop-sequence truncation locally as well in case fallback ran
        earliest_idx: Optional[int] = None
        earliest_seq: Optional[str] = None
        for seq in stop_sequences:
            pos = raw_text.find(seq)
            if pos != -1 and (earliest_idx is None or pos < earliest_idx):
                earliest_idx = pos
                earliest_seq = seq
        if earliest_idx is not None and earliest_seq is not None:
            raw_text = raw_text[: earliest_idx + len(earliest_seq)]

        latency_s = time.perf_counter() - t0
        full_text, matched_stop = _reattach_stop_tag(raw_text, stop_sequences, earliest_seq)
        if native_thought and "<thought>" not in full_text:
            full_text = f"<thought>\n{native_thought}\n</thought>\n{full_text}"

        usage = data.get("usageMetadata", {})
        return GenerationStep(
            text=full_text,
            matched_stop=matched_stop,
            latency_s=latency_s,
            prompt_tokens=usage.get("promptTokenCount", 0),
            completion_tokens=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            model=model_id,
        )
