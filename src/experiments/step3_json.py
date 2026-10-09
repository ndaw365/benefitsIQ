"""Schema-constrained output: ask() returns a validated PolicyAnswer, not raw JSON text."""

from src.llm import ask
from src.schemas import PolicyAnswer

SYSTEM = "You are an assistant for group disability claims examiners."
PROMPT = "Explain what an elimination period is in group disability insurance."

if __name__ == "__main__":
    result = ask(PROMPT, system=SYSTEM, schema=PolicyAnswer)
    print(type(result).__name__)
    print(result.model_dump_json(indent=2))
