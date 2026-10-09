from dotenv import load_dotenv
from google import genai
from google.genai import types
import os

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
client = genai.Client()  # reads GEMINI_API_KEY from the environment

#MODEL = "gemini-3.5-flash-lite"  # confirm this name in AI Studio; names change


def ask(prompt: str, temperature: float = 0.0, system: str | None = None):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=1000,
            system_instruction=system,
        ),
    )
    usage = response.usage_metadata
    return response.text, usage.prompt_token_count, usage.candidates_token_count