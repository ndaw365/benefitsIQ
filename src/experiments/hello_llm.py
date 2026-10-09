"""Compare outputs at temperature 0 vs 1. Token counts are in logs/llm_calls.jsonl."""

from src.llm import ask

PROMPT = "Explain what an elimination period is in group disability insurance."

if __name__ == "__main__":
    for temp in (0.0, 1.0):
        print(f"\n===== temperature={temp} =====")
        for i in range(3):
            text = ask(PROMPT, temperature=temp)
            print(f"\n--- run {i+1} ---")
            print(text[:400])
