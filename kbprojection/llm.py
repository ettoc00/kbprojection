import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Any, List, Optional

import httpx

from .async_runtime import AsyncRunContext, resolve_async_run_context
from .models import LLMKBResponse, NLIProblem
from .prompts import fill_prompt


PROVIDER_ALIASES = {
    "anthropic": "claude",
    "claude": "claude",
    "gemini": "gemini",
    "google": "gemini",
    "openai": "openai",
    "openrouter": "openrouter",
}

DEFAULT_MODELS = {
    "claude": "claude-sonnet-4-5",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-5-mini",
    "openrouter": "openai/gpt-5-mini",
}

# Anthropic requires an explicit output limit; other providers can omit it.
DEFAULT_PROVIDER_MAX_TOKENS = {
    "claude": 8192,
}


class LLMGenerationError(RuntimeError):
    """Raised when an LLM call fails after retries are exhausted."""


def _load_dotenv_if_present(dotenv_path: Optional[Path] = None) -> None:
    """
    Populate missing environment variables from a local .env file.

    The workspace already keeps provider keys in `.env`, but the pipeline is
    often invoked without pre-exporting them into the shell environment.
    """
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"

    if not dotenv_path.exists():
        return

    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or key in os.environ:
            continue

        if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        os.environ[key] = value


def _require_api_key(env_var_name: str) -> str:
    api_key = os.environ.get(env_var_name)
    if not api_key:
        raise ValueError(
            f"{env_var_name} is not set. Export it in your shell or place it in .env."
        )
    return api_key


_load_dotenv_if_present()


def normalize_provider(provider: Optional[str]) -> Optional[str]:
    if provider is None:
        return None

    normalized = PROVIDER_ALIASES.get(provider.strip().lower())
    if normalized is None:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Expected one of {sorted(PROVIDER_ALIASES)}."
        )
    return normalized


class AsyncGenericAIClient:
    """
    A generic AI client that wraps OpenAI, Claude, Gemini, and OpenRouter.
    It automatically detects the provider based on environment variables if not specified.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = normalize_provider(provider) if provider else self._detect_provider()
        self.client = None
        self._setup_client()

    def _detect_provider(self) -> str:
        if os.environ.get("OPENROUTER_API_KEY"):
            return "openrouter"
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini"
        if os.environ.get("ANTHROPIC_API_KEY"):
            return "claude"
        raise ValueError(
            "No API keys found. Please set OPENROUTER_API_KEY, OPENAI_API_KEY, "
            "GEMINI_API_KEY, or ANTHROPIC_API_KEY."
        )

    def _setup_client(self):
        if self.provider == "openrouter":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=_require_api_key("OPENROUTER_API_KEY"),
            )
        elif self.provider == "openai":
            from openai import AsyncOpenAI

            self.client = AsyncOpenAI(api_key=_require_api_key("OPENAI_API_KEY"))
        elif self.provider == "claude":
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                self.client = None
            else:
                self.client = AsyncAnthropic(api_key=_require_api_key("ANTHROPIC_API_KEY"))
        elif self.provider == "gemini":
            try:
                from google import genai
            except ImportError:
                self.client = None
            else:
                self.client = genai.Client(api_key=_require_api_key("GEMINI_API_KEY"))
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def generate(
        self,
        prompt: str,
        model: Optional[str],
        response_model: Any = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        model = model or DEFAULT_MODELS[self.provider]

        if self.provider in {"openai", "openrouter"}:
            if response_model:
                response = await self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_model,
                )
                return response.choices[0].message.parsed

            response = await self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content

        if self.provider == "claude":
            max_tokens = max_tokens or DEFAULT_PROVIDER_MAX_TOKENS["claude"]
            if self.client is None:
                return await self._generate_claude_http(prompt, model, response_model, max_tokens)

            if response_model:
                response = await self.client.beta.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=response_model,
                )
                return response.parsed

            response = await self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        if self.provider == "gemini":
            if self.client is None:
                return await self._generate_gemini_http(prompt, model, response_model)

            config = {}
            if response_model:
                config = {
                    "response_mime_type": "application/json",
                    "response_schema": response_model,
                }

            maybe_response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            response = await maybe_response if hasattr(maybe_response, "__await__") else maybe_response

            if response_model:
                text = response.text.strip()
                return _parse_response_model_text(text, response_model)

            return response.text

        raise ValueError(f"Provider {self.provider} not supported for generation.")

    async def _generate_claude_http(
        self,
        prompt: str,
        model: str,
        response_model: Any = None,
        max_tokens: Optional[int] = None,
    ) -> Any:
        headers = {
            "x-api-key": _require_api_key("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens or DEFAULT_PROVIDER_MAX_TOKENS["claude"],
            "messages": [{"role": "user", "content": prompt}],
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
            response.raise_for_status()
        text = response.json()["content"][0]["text"]
        if response_model:
            return _parse_response_model_text(text, response_model)
        return text

    async def _generate_gemini_http(
        self,
        prompt: str,
        model: str,
        response_model: Any = None,
    ) -> Any:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            f"?key={_require_api_key('GEMINI_API_KEY')}"
        )
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        if response_model:
            payload["generationConfig"] = {"response_mime_type": "application/json"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        text = "".join(part.get("text", "") for part in parts)
        if response_model:
            return _parse_response_model_text(text, response_model)
        return text


GenericAIClient = AsyncGenericAIClient


def _parse_response_model_text(text: str, response_model: Any) -> Any:
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    data = json.loads(stripped)
    return response_model(**data)


KB_PATTERN = re.compile(r"^\s*(isa_wn|disj)\s*\(\s*[^,]+\s*,\s*[^)]+\s*\)\s*$", re.MULTILINE)
KB_RELATION_PREFIX = re.compile(r"^\s*(isa_wn|disj)\s*\(")


def extract_kb_from_output(llm_output: str) -> List[str]:
    """
    Extract KB injection lines from LLM output.

    Supports two formats:
    1. New delimited format with [KB_START] ... [KB_END] markers
    2. Legacy format: lines starting with isa_wn( or disj(
    """
    start_marker = "[KB_START]"
    end_marker = "[KB_END]"

    start_idx = llm_output.find(start_marker)
    end_idx = llm_output.find(end_marker)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        kb_block = llm_output[start_idx + len(start_marker):end_idx].strip()
        lines = kb_block.split("\n")
    else:
        lines = llm_output.split("\n")

    kb_injections = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if KB_PATTERN.match(line):
            kb_injections.append(line)
        elif line.startswith(("isa_wn(", "disj(")) and "(" in line and ")" in line:
            kb_injections.append(line)

    return kb_injections


def _extract_validated_kb_from_output(llm_output: str) -> List[str]:
    start_marker = "[KB_START]"
    end_marker = "[KB_END]"

    start_idx = llm_output.find(start_marker)
    end_idx = llm_output.find(end_marker)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        kb_block = llm_output[start_idx + len(start_marker):end_idx].strip()
        lines = kb_block.split("\n")
    else:
        lines = llm_output.split("\n")

    kb_injections = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if KB_PATTERN.match(line):
            kb_injections.append(line)
        elif KB_RELATION_PREFIX.match(line):
            raise ValueError(f"Malformed KB relation from LLM: {line}")

    return kb_injections


async def call_llm(
    provider: Optional[str],
    model: Optional[str],
    prompt_style: str,
    prob: NLIProblem,
    max_tokens: Optional[int] = None,
    max_retries: Optional[int] = 3,
    context: Optional[AsyncRunContext] = None,
) -> List[str]:
    """
    Unified LLM call using GenericAIClient.
    """
    is_legacy = prompt_style.startswith("legacy_")
    prompt = fill_prompt(prompt_style, prob.premises, prob.hypothesis)

    response_model = None
    if is_legacy:
        prompt += (
            "\n\nOutput strictly valid JSON matching this schema:\n"
            f"{json.dumps(LLMKBResponse.model_json_schema(), indent=2)}"
        )
        response_model = LLMKBResponse

    client = AsyncGenericAIClient(provider=provider)
    resolved_context = resolve_async_run_context(context)

    retries = 0
    provider_name = client.provider
    while True:
        try:
            async with resolved_context.llm_semaphore:
                output = await client.generate(
                    prompt=prompt,
                    model=model,
                    response_model=response_model,
                    max_tokens=max_tokens,
                )

            if not is_legacy:
                return _extract_validated_kb_from_output(output)
            return [
                relation
                for kbi in output.output
                for relation in _extract_validated_kb_from_output(kbi.KB_injection)
            ]
        except Exception as e:
            if max_retries is not None and retries >= max_retries:
                raise LLMGenerationError(
                    f"LLM generation failed with {provider_name}"
                    f" using model {model or DEFAULT_MODELS[provider_name]}: {e}"
                ) from e

            retries += 1
            print(
                f"Error during LLM generation with {provider_name}: {e}. "
                f"Retrying ({retries})..."
            )
            delay = min(8.0, 0.5 * (2 ** (retries - 1))) + random.uniform(0.0, 0.25)
            await asyncio.sleep(delay)


async def inject_kb_for_example(
    prob: NLIProblem,
    model: Optional[str],
    prompt_style: str,
    provider: Optional[str] = None,
) -> List[str]:
    """
    Convenience wrapper using auto-detection unless a provider is specified.
    """
    return await call_llm(provider, model, prompt_style, prob)
