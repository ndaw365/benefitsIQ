"""Block 1: ask plan-specific questions with NO documents, and compare to the real text.

This is the "before" picture for RAG: what does the model say when it has to rely
on what it memorized during training?
"""

from src.llm import ask
from src.schemas import PolicyAnswer

SYSTEM = "You are an assistant for group disability and life insurance claims examiners."
# A common (bad) chatbot instruction: it rewards answering over admitting uncertainty.
PRESSURE = (
    "You are a benefits assistant. Examiners are busy: always give a specific, direct "
    "answer in one or two sentences. Do not tell them to check other sources."
)

# Ground truth was copied by hand from the PDFs in data/docs/ (page = PDF page number).
CASES = [
    {
        "question": "What is the elimination period in the Lincoln Financial long-term "
        "disability plan for Moravian University?",
        "truth": "180 calendar days of Disability ... accumulated within a 360 calendar day period.",
        "source": "lincoln_moravian_ltd.pdf p.3",
    },
    {
        "question": "What benefit percentage and maximum monthly benefit does Moravian "
        "University's Lincoln Financial LTD plan pay?",
        "truth": "Benefit percentage 60%; maximum monthly benefit $10,000.",
        "source": "lincoln_moravian_ltd.pdf p.3",
    },
    {
        "question": "What is the benefit waiting period in The Standard's long-term "
        "disability plan for the University of Alabama in Huntsville?",
        "truth": "The later of 90 days, or the date accumulated sick leave payments end.",
        "source": "standard_uah_ltd.pdf p.8",
    },
    {
        "question": "Under the State of Colorado group term life certificate from "
        "Minnesota Life, how long does an employee have to convert coverage to an "
        "individual policy after group insurance ends?",
        "truth": "Apply and pay the first premium within 31 days after group insurance terminates.",
        "source": "securian_colorado_term_life.pdf p.11",
    },
]


def main() -> None:
    """Two conditions per question: a bare call, and our PolicyAnswer schema (which
    tells the model to say so instead of guessing)."""
    for i, case in enumerate(CASES, start=1):
        print(f"\n=== Q{i}: {case['question']}")
        print(f"TRUTH ({case['source']}):\n  {case['truth']}")

        bare = ask(case["question"], max_tokens=4096)
        print(f"BARE MODEL (no schema, no system prompt):\n  {bare.strip()[:600]}")

        pressured = ask(case["question"], system=PRESSURE, max_tokens=256)
        print(f"PRESSURED (must give a specific answer):\n  {pressured.strip()[:300]}")

        result = ask(case["question"], system=SYSTEM, schema=PolicyAnswer)
        print(f"WITH SCHEMA ({result.confidence}, review={result.needs_human_review}):\n"
              f"  {result.answer}")


if __name__ == "__main__":
    main()
