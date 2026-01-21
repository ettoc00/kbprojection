import os
import json
from typing import List, Optional, Any, Union
from .prompts import fill_prompt
from .models import LLMKBResponse, NLIProblem
from .models import LLMKBResponse, NLIProblem
import re

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

    def generate(self, prompt: str, model: str, response_model: Any = None, max_tokens: int = 1024) -> Any:
        
        # --- OpenRouter / OpenAI ---
        if self.provider in ["openrouter", "openai"]:
            # Handle model aliases or defaults if needed
            if not model:
                model = "gpt-5-mini" if self.provider == "openai" else "google/gemini-flash-1.5"

            if response_model:
                response = self.client.beta.chat.completions.parse(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=response_model,
                )
                return response.choices[0].message.parsed
            else:
                # Raw text generation
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

        # --- Claude ---
        elif self.provider == "claude":
            if not model:
                model = "claude-sonnet-4-5"

            if response_model:
                response = self.client.beta.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=response_model,
                )
                return response.parsed
            else:
                # Raw text generation
                response = self.client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

        # --- Gemini ---
        elif self.provider == "gemini":
            if not model:
                model = "gemini-2.5-flash"

            genai = self.gemini_module
            
            config = {}
            if response_model:
                config = {
                    "response_mime_type": "application/json",
                    "response_schema": response_model,  
                }
            
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            
            if response_model:
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
                return response.text

        else:
            raise ValueError(f"Provider {self.provider} not supported for generation.")


# Pattern to match KB predicates: predicate(arg1, arg2)
KB_PATTERN = re.compile(r'^\s*(isa_wn|disj)\s*\(\s*[^,]+\s*,\s*[^)]+\s*\)\s*$', re.MULTILINE)


def extract_kb_from_output(llm_output: str) -> List[str]:
    """
    Extract KB injection lines from LLM output.
    
    Supports two formats:
    1. New delimited format with [KB_START] ... [KB_END] markers
    2. Legacy format: lines starting with isa_wn( or disj(
    
    Returns:
        List of KB injection strings (e.g., ["isa_wn(dog, animal)", "disj(sit, stand)"])
    """
    # Try new delimited format first
    start_marker = "[KB_START]"
    end_marker = "[KB_END]"
    
    start_idx = llm_output.find(start_marker)
    end_idx = llm_output.find(end_marker)
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Extract content between delimiters
        kb_block = llm_output[start_idx + len(start_marker):end_idx].strip()
        lines = kb_block.split('\n')
    else:
        # Fallback to legacy format: extract all lines that look like KB injections
        lines = llm_output.split('\n')
    
    # Filter to only valid KB injection lines
    kb_injections = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if line matches KB pattern
        if KB_PATTERN.match(line):
            kb_injections.append(line)
        # Also try to extract if line starts with predicate (for less strict matching)
        elif line.startswith(('isa_wn(', 'disj(')):
            # Try to parse it
            try:
                # We do a basic check here, actual parsing happens later in filtering
                # But creating a dummy call helps verify basic structure
                if '(' in line and ')' in line:
                    kb_injections.append(line)
            except ValueError:
                continue
    
    return kb_injections


def call_llm(
    provider: Optional[str],
    model: str,
    prompt_style: str,
    prob: NLIProblem,
    max_tokens: int = 200,
    max_retries: Optional[int] = 3,
) -> List[str]:

    """
    Unified LLM call using GenericAIClient.
    """
    
    is_legacy = prompt_style.startswith("legacy_")
    
    # Fill prompt
    prompt = fill_prompt(prompt_style, prob.premises, prob.hypothesis)
    
    response_model = None
    if is_legacy:
        # Legacy: Append expected structure and set response model
        prompt += f"\n\nOutput strictly valid JSON matching this schema:\n{json.dumps(LLMKBResponse.model_json_schema(), indent=2)}"
        response_model = LLMKBResponse
    
    # Initialize Client (auto-detects if provider is None)
    client = GenericAIClient(provider=provider)
    
    # Generate
    retries = 0
    while True:
        try:
            output = client.generate(
                prompt=prompt,
                model=model,
                response_model=response_model,
                max_tokens=max_tokens
            )
            
            if not is_legacy:
                # Output is raw text (new default prompts), parse it
                return extract_kb_from_output(output)
            else:
                # Output is parsed Pydantic object (legacy prompts)
                return [kbi.KB_injection for kbi in output.output]
                
        except Exception as e:
            if max_retries is not None and retries >= max_retries:
                print(f"Error during LLM generation with {client.provider}: {e}")
                return []
            
            retries += 1
            print(f"Error during LLM generation with {client.provider}: {e}. Retrying ({retries})...")

def inject_kb_for_example(prob: NLIProblem, model: str, prompt_style: str) -> List[str]:
    """
    Convenience wrapper using auto-detection.
    """
    return call_llm(None, model, prompt_style, prob)