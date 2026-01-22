import os
from typing import List, Optional, Any, Union
from .prompts import fill_prompt
from .models import LLMKBResponse, NLIProblem

class GenericAIClient:
    """
    A generic AI client that wraps OpenAI, Anthropic, Google GenAI, and OpenRouter.
    It automatically detects the provider based on environment variables if not specified.
    """

    def __init__(self, provider: Optional[str] = None):
        self.provider = provider.lower() if provider else self._detect_provider()
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
        raise ValueError("No API keys found. Please set OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or ANTHROPIC_API_KEY.")

    def _setup_client(self):
        if self.provider == "openrouter":
            from openai import OpenAI
            api_key = os.environ.get("OPENROUTER_API_KEY")
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
        elif self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "claude":
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            self.client = Anthropic(api_key=api_key)
        elif self.provider == "gemini":
            from google import genai            
            api_key = os.environ.get("GEMINI_API_KEY")
            self.client = genai.Client(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(self, prompt: str, model: str, response_model: Any, max_tokens: int = 1024) -> Any:
        
        # --- OpenRouter / OpenAI ---
        if self.provider in ["openrouter", "openai"]:
            # Handle model aliases or defaults if needed
            if not model:
                model = "gpt-4o" if self.provider == "openai" else "google/gemini-flash-1.5"

            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_model,
            )
            return response.choices[0].message.parsed

        # --- Claude ---
        elif self.provider == "claude":
            if not model:
                model = "claude-sonnet-4-5"

            response = self.client.beta.messages.parse(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
            )
            return response.parsed

        # --- Gemini ---
        elif self.provider == "gemini":
            if not model:
                model = "gemini-2.5-flash"

            genai = self.gemini_module
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": response_model,  
                },
            )
            text = response.text
            # Clean markdown
            if text.startswith("```json"):
                 text = text[7:-3]
            elif text.startswith("```"):
                 text = text[3:-3]
            
            import json
            data = json.loads(text)
            return response_model(**data)

        else:
            raise ValueError(f"Provider {self.provider} not supported for generation.")


def call_llm(
    provider: Optional[str],
    model: str,
    prompt_style: str,
    prob: NLIProblem,
    max_tokens: int = 200,
) -> List[str]:

    """
    Unified LLM call using GenericAIClient.
    """
    
    # Fill prompt
    prompt = fill_prompt(prompt_style, prob.premise, prob.hypothesis)
    
    # Initialize Client (auto-detects if provider is None)
    client = GenericAIClient(provider=provider)
    
    # Generate
    try:
        parsed_response: LLMKBResponse = client.generate(
            prompt=prompt,
            model=model,
            response_model=LLMKBResponse,
            max_tokens=max_tokens
        )
        return [kbi.KB_injection for kbi in parsed_response.output]
    except Exception as e:
        print(f"Error during LLM generation with {client.provider}: {e}")
        return []

def inject_kb_for_example(prob: NLIProblem, model: str, prompt_style: str) -> List[str]:
    """
    Convenience wrapper using auto-detection.
    """
    return call_llm(None, model, prompt_style, prob)