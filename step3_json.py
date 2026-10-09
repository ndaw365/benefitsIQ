import json
from llm import ask

SYSTEM = """You are an assistant for group disability claims examiners.
Answer concisely (max 3 sentences).
Respond ONLY with valid JSON with exactly these keys:
"answer": string
"confidence": one of "low", "medium", "high"
"""

PROMPT = "Explain what an elimination period is in group disability insurance."

text, tin, tout = ask(PROMPT, temperature=0.0, system=SYSTEM)
print("RAW:", repr(text))
print("tokens:", tin, tout)

data = json.loads(text)
print(data["answer"])