# Block 1: Hallucination demo (no documents given to the model)

Run: 2026-10-09, `gemini-3.5-flash-lite`, temperature 0, one run per question.
Script: `python -m src.experiments.hallucination_demo`. Ground truth copied by hand from `data/docs/`.

## Results

| # | Question (short) | Truth (source) | Bare model | Pressured prompt ("always give a specific answer") |
|---|---|---|---|---|
| 1 | Lincoln/Moravian LTD elimination period | 180 calendar days, accumulated within 360 days (lincoln p.3) | Declined, pointed to HR | **"90 consecutive days" (wrong)** |
| 2 | Lincoln/Moravian benefit % and max | 60%, $10,000/month (lincoln p.3) | Declined | 60%, $10,000 (correct) |
| 3 | Standard/UAH benefit waiting period | Later of 90 days or end of accumulated sick leave (standard p.8) | Declined | **"90 days" (incomplete: drops the sick-leave condition)** |
| 4 | Colorado life conversion window | Apply and pay first premium within 31 days (securian p.11) | 31 days, framed as "generally" | 31 days (correct) |

With our `PolicyAnswer` schema and no documents, all 4 abstained (`confidence=low`, `needs_human_review=true`),
but said "not in the provided documents" when none were provided: the schema descriptions acted as instructions.

## What this shows
1. **The model's behavior depends on the prompt, not just what it knows.** The same model declined when left alone,
   abstained under our schema, and answered every question when told to always be specific.
2. **Under pressure, right and wrong answers look identical.** Q1 is confidently wrong (90 vs 180 days, which
   changes when a claimant's benefits start by three months). Q3 drops a condition that matters for a claim. Q2 and Q4
   are correct, but 60%/$10,000 and 31 days are common industry defaults, so we can't tell recall from a lucky
   guess. An examiner has no way to tell which answers to trust.
3. **So the case for RAG here is verifiability, not just missing knowledge:** answers must come from the plan's own
   text with a page citation, and the system must abstain when the text isn't retrieved.

## Limitations
- 4 questions, 1 run each, one model. This is an illustration, not a measurement; the golden-set eval measures it properly.
- Results may differ on other models or model versions.
