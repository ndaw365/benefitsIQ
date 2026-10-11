"""Blind hand-grading of a sample of judge verdicts, and judge-vs-human agreement."""

import argparse
import json
import random
from pathlib import Path

from src.eval.faithfulness import RESULTS_DIR, FaithResult, finalize, load_done
from src.eval.golden import load_golden
from src.rag.index import DEFAULT_OVERLAP, DEFAULT_SIZE, make_embedder
from src.rag.retrieve import open_collection, retrieve
from src.rag.route import detect_policies

FAITHFULNESS = ["supported", "partially_supported", "unsupported", "no_claims"]
CORRECTNESS = ["correct", "partially_correct", "incorrect", "abstained"]


def pick_sample(rows: list[dict], n: int, seed: int = 0) -> list[str]:
    """Every error first (they matter most), then a seeded random fill, so the sample is reproducible."""
    errors = [r["id"] for r in rows if r["outcome"] in ("unsafe_error", "flagged_error")]
    rest = sorted(r["id"] for r in rows if r["id"] not in errors)
    random.Random(seed).shuffle(rest)
    return (errors + rest)[:n]


def agreement(human: dict[str, dict], judge: dict[str, dict]) -> dict:
    """Share of graded items where human and judge chose the same label, per field, with the disagreements."""
    ids = [i for i in human if i in judge]
    report = {"n": len(ids)}
    for field in ("faithfulness", "correctness"):
        diffs = [(i, human[i][field], judge[i][field]) for i in ids if human[i][field] != judge[i][field]]
        report[field] = {"agreement": round(1 - len(diffs) / len(ids), 3) if ids else None, "disagreements": diffs}
    return report


def choose(prompt: str, options: list[str]) -> str:
    """Ask until the grader picks a valid option by number."""
    menu = "  ".join(f"{n}={o}" for n, o in enumerate(options, start=1))
    while True:
        raw = input(f"{prompt} [{menu}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]


def grade(run: str, n: int) -> None:
    rows = list(load_done(RESULTS_DIR / f"{run}.jsonl").values())
    labels_path = RESULTS_DIR / f"{run}_human_labels.jsonl"
    graded = load_done(labels_path)
    golden = {item.id: item for item in load_golden()}
    embed, collection = make_embedder(), open_collection(DEFAULT_SIZE, DEFAULT_OVERLAP)

    sample = pick_sample(rows, n)
    by_id = {r["id"]: r for r in rows}
    for k, item_id in enumerate(sample, start=1):
        if item_id in graded:
            continue
        item, row = golden[item_id], by_id[item_id]
        # Same routed retrieval the pipeline used (deterministic), so you see what the model and judge saw.
        hits = retrieve(item.question, embed, collection, 5, detect_policies(item.question))
        print("\n" + "=" * 100 + f"\n[{k}/{len(sample)}] {item_id}\n")
        for h in hits:
            print(f"[{h.chunk_id}]\n{h.text}\n")
        print(f"QUESTION:  {item.question}\nREFERENCE: {item.expected_answer}\nANSWER:    {row['answer']}\n")
        label = {
            "id": item_id,
            "faithfulness": choose("Faithfulness (answer vs PASSAGES only)", FAITHFULNESS),
            "correctness": choose("Correctness (answer vs REFERENCE)", CORRECTNESS),
            "note": input("Note (optional): ").strip(),
        }
        with labels_path.open("a") as f:
            f.write(json.dumps(label) + "\n")
    print(f"\nSaved to {labels_path}. Run: python -m src.eval.judge_agreement report --run {run}")


def report(run: str) -> None:
    human = load_done(RESULTS_DIR / f"{run}_human_labels.jsonl")
    judge = {i: vars(finalize(FaithResult(**r))) for i, r in load_done(RESULTS_DIR / f"{run}.jsonl").items()}
    result = agreement(human, judge)
    print(json.dumps(result, indent=2))
    for field in ("faithfulness", "correctness"):
        for item_id, h, j in result[field]["disagreements"]:
            print(f"\n{field} disagreement on {item_id}: you={h} judge={j}\n  judge's reason: {judge[item_id]['reason']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hand-grade judge verdicts blind, then measure agreement.")
    parser.add_argument("command", choices=["grade", "report"])
    parser.add_argument("--run", default="baseline")
    parser.add_argument("--n", type=int, default=8)
    args = parser.parse_args()
    grade(args.run, args.n) if args.command == "grade" else report(args.run)


if __name__ == "__main__":
    main()
