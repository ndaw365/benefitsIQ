# BenefitsIQ

A claims and policy copilot for group disability and life insurance. It answers policy questions with page citations, extracts claim forms into validated data, estimates benefits with exact arithmetic, and hands anything uncertain to a human. Built by hand in Python (no LangChain or LlamaIndex) to learn the patterns, with evaluation as the core of the work.

> Portfolio project. Uses only public sample certificates and synthetic claim forms. Outputs are estimates for an examiner to verify, never claim decisions.

**One-page business case:** [docs/business_case.md](docs/business_case.md)

## What it does
| Capability | How |
|---|---|
| **Policy Q&A with citations** | PDF loading, page-bounded chunks, local embeddings in Chroma, policy routing, grounded answers through a schema-constrained LLM call, citation verification |
| **Claim form extraction** | Schema-constrained extraction that copies rather than corrects; plain-Python rules decide when a human must review |
| **Claims agent** | Hand-written tool-calling loop over `get_claim`, `search_policy`, `calculate_benefit`, `escalate_to_human`, with code-level guardrails |
| **Evaluation** | 32-question golden set, retrieval hit rate, LLM-as-judge faithfulness and correctness validated against human grading, extraction accuracy |
| **Serving** | Streamlit UI and a FastAPI service with API-key auth and rate limiting |

## Results
Measured on the golden set (32 questions, 29 answerable) and 6 labeled synthetic claim forms. Small samples: they show behavior, not production accuracy.

| Measure | Result |
|---|---|
| Retrieval: answer passage in top 5 (800-char chunks, policy routing) | 72.4% (21/29) |
| Answerable questions answered correctly with a citation | 20/29 |
| Safe abstentions ("not found", flagged for review) | 7/29 |
| Partially correct (a condition left out) | 2/29 (1 flagged, 1 unflagged) |
| Unanswerable questions correctly declined | 3/3 |
| Answers with claims not supported by the passages | 0 (LLM judge) |
| Judge agreement with blind human grading | 7/8 on faithfulness, 7/8 on correctness |
| Retrieval misses that produced an abstention rather than a guess (baseline run) | 8/8 |
| Claim extraction: field accuracy / planted issues caught / false flags | 98.8% / 7 of 7 / 0 |
| Typical question cost | ~1,150 input + ~85 output tokens, ~0.9 s median model latency |
| Tests | 181 pytest tests, no API calls needed |

## Architecture
```
  Streamlit UI (src/ui)              FastAPI (src/api): auth, rate limit, validation
            \                              /
             Shared core, no UI or HTTP code
   src/rag         load -> chunk -> embed -> route -> retrieve -> answer -> verify citations
   src/extraction  form text -> ClaimExtraction (schema) -> business-rule review
   src/agent       tool-calling loop + guardrails over RAG, extraction, calculator, escalation
   src/eval        golden set, retrieval, faithfulness/correctness judge, extraction, judge agreement
             |                       |                         |
   MiniLM embeddings (local)   Chroma (.chroma/)   src/llm.py -> Gemini (retries, logging, schemas)
```
Design rules: every LLM call goes through `src/llm.py`; every Pydantic schema lives in `src/schemas.py`; decisions that must be reliable (review flags, citation checks, arithmetic, eligibility rules) are plain Python, not prompts.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then set GEMINI_API_KEY, GEMINI_MODEL, BENEFITSIQ_API_KEY
python -m src.rag.index     # build the vector index once (creates .chroma/)
python -m pytest -q         # ~5 s, no API calls

python -m streamlit run src/ui/app.py            # UI  -> http://localhost:8501
python -m uvicorn src.api.main:app --reload      # API -> http://127.0.0.1:8000/docs
```
Generate an API key with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Run everything from the repo root with `python -m` so `src` imports resolve. Model names change: confirm `GEMINI_MODEL` in Google AI Studio.

**Command-line tools**
```bash
python -m src.rag.answer "What is the elimination period under the Moravian LTD policy?"
python -m src.extraction.extract data/claims/claim_06.txt
python -m src.agent.loop "What is the estimated monthly LTD benefit for claim_05?"
```

**API**: send `X-API-Key: <BENEFITSIQ_API_KEY>`. `/ask`, `/claims/extract`, and `/agent` cost 1, 1, and 5 units of a 20-per-minute budget. Errors: 401 bad key, 422 invalid input, 429 rate limited, 502 unusable model output, 503 missing server key or index.

## Evaluation
```bash
python -m src.eval.retrieval --route                 # hit rate@5 per chunk config (no LLM calls)
python -m src.eval.faithfulness --out myrun          # 64 LLM calls; resumable
python -m src.eval.judge_agreement grade             # blind hand-grading of judge verdicts
python -m src.eval.extraction                        # 6 LLM calls
```
- **Golden set** (`data/golden/golden_set.jsonl`): facts, definitions, table rows, multi-policy, paraphrased, and unanswerable questions. Each answer is labeled by page with a verbatim quote, and a test checks every quote against the PDF text.
- **Retrieval** is scored strictly (the retrieved chunk must overlap the quote) and leniently (right page). Page-level scoring overstated hit rate by about 10 points.
- **Faithfulness and correctness** are judged separately: an answer can be faithful to the wrong passage, or correct from memory. Outcomes map to correct, safe abstention, flagged error, or unsafe (wrong and unflagged).
- **The judge was validated** by blind human grading before its numbers were trusted, and judge label rules it ignored are enforced in code.

## Experiment log
Span hit@5 on 29 answerable questions, 800-char chunks unless noted. Every change was measured per item, not just in aggregate.

| Change | Result | Kept? | Why |
|---|---|---|---|
| Baseline vector search | 65.5% | | |
| Collapse dot leaders in tables | 69.0% | Yes | Removed every chunk over the embedder's 256-token limit (15 to 0) |
| 400-char chunks | 58.6% to 65.5% | No | Wins short facts, loses long definitions |
| Hybrid BM25 + reciprocal rank fusion | 65.5% | No | Equal-weight fusion dropped results only one retriever found |
| **Policy routing** (filter by policy named in question) | **72.4%** | **Yes** | Chunks don't contain the policy name; questions do |
| Strip policy names after routing | 58.6% | No | Schedule pages contain the policyholder name, so it is real signal |
| Constant document-title header on each chunk | 62.1% | No | A shared prefix compresses distances within a document |
| Prompt: include conditions on values | unsafe answers 2 to 1 | Yes | Fixed "36 months" missing "or retirement age if later" |
| Agent search skips passages already returned | 6 searches + escalation to 2 searches + correct answer | Yes | Deterministic retrieval made retries repeat themselves |
| Placeholder text ("[illegible]") counts as missing | issue recall 86% to 100% | Yes | The model copied the placeholder instead of returning null |

## Responsible AI
- **Human in the loop, by design.** The system never approves or denies claims; there is no tool that could. Decision requests, missing data, and rule violations escalate to an examiner.
- **Review flags come from rules, not model confidence.** Claim review issues, unverified citations, uncited answers, calculations made before reading the policy, and decision language all set `needs_review` in code.
- **Verifiability over fluency.** Answers cite pages; citations to passages the model was not shown are removed and flagged. The UI explains why review is needed and never shows a plain "verified" state.
- **Abstain rather than guess.** When retrieval misses, the model says the passages do not contain the answer; in the baseline eval run, all 8 retrieval misses produced abstentions.
- **Data.** Public certificates and synthetic claims only, with no real or realistic personal identifiers. The free API tier may use inputs for training, so real claims would need an approved, non-training endpoint and a privacy review.
- **Fairness and access.** Extraction was tested only on clean, English, typed synthetic forms. Accuracy on handwriting, scans, or other languages is unknown and could differ by claimant population; that must be measured before use on real claims.
- **Transparency and monitoring.** Every model call is logged with tokens, latency, and errors; agent runs and escalations are logged with full tool traces.

## Limitations
- Small evaluation sets (32 questions, 6 forms), written by the builder, one run each. One golden-set reference was trimmed after human grading found it broader than its question.
- The same model answers and judges; the judge agreed with human grading on 7 of 8 items and graded near-identical answers differently once.
- Retrieval misses multi-policy questions (0/2) and some table rows; benefit estimates simplify each policy's offset and earnings definitions.
- Synthetic claim forms are clean text, not scanned images.

## Next steps
- **Retrieval:** cross-encoder reranking, table-aware extraction (one sentence per table row), per-policy retrieval for multi-policy questions, a held-out test set written after tuning.
- **Evaluation:** a different model as judge, judge test-retest consistency, agent-level evals with the judge, more forms including scanned and handwritten ones.
- **Engineering:** Docker, CI running the test suite, cloud deployment, async model calls, a database for escalations and run logs, shared rate limiting (e.g. Redis) and real user authentication.
- **Analytics:** predictive modeling on claim outcomes (e.g. likely duration) once real, approved data is available.

## Project structure
```
src/
  llm.py            single gateway to Gemini: ask(), ToolChat, retries, logging
  schemas.py        every Pydantic schema (LLM outputs, golden set, API models)
  rag/              load, chunk, index, retrieve, route, bm25, answer
  extraction/       extract (LLM), review (rules)
  agent/            benefit calculator, tools, loop + guardrails
  eval/             golden, retrieval, faithfulness, judge_agreement, extraction
  ui/app.py         Streamlit
  api/              FastAPI app, security (auth, rate limit)
  views.py          review reasons and labels shared by UI and API
data/  docs/ (public PDFs, see data/SOURCES.md), golden/, claims/ (synthetic)
results/            saved eval outputs        notes/  hallucination demo write-up
docs/               business case             tests/  pytest suite
```
