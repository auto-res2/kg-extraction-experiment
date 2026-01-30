"""Gemini API client with structured output support and retry logic."""

import json
import time

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

MODEL = "gemini-3-flash-preview"


def load_api_key(env_path: str) -> str:
    """Parse .env file and return GEMINI_API_KEY."""
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1]
    raise ValueError("GEMINI_API_KEY not found in .env file")


def create_client(api_key: str) -> genai.Client:
    """Create Gemini client."""
    return genai.Client(api_key=api_key)


def call_gemini(
    client: genai.Client,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> dict:
    """Call Gemini with structured JSON output. Returns parsed dict."""
    config = GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=response_schema,
        temperature=temperature,
        thinking_config=ThinkingConfig(thinking_budget=2048),
    )

    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=user_prompt,
                config=config,
            )
            return json.loads(resp.text)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [retry {attempt+1}/{max_retries}] {e}, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
