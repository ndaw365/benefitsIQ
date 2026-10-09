# BenefitsIQ: AI collaborator brief

## Who I am and what I want
I'm a data scientist (Python, ML, pipelines, evaluation; background in computational biology) with no generative AI experience. I'm building this project in ~2 days to become a strong candidate for an "AI and Analytics Specialist" role on a Workforce Benefits (group disability/life insurance) team. The job asks for: LLM APIs, prompt engineering, RAG, knowledge extraction, evaluation frameworks, AI agents/tool calling, Python, APIs, MLOps basics, responsible AI, and business-impact communication.

**My goal is to learn, not just to finish.** I must be able to explain every line in an interview.

## How you must work with me (most important)
1. **Explain before you code.** Give a 2-3 sentence plain-language explanation of the concept and why we're building it this way.
2. **Small steps.** One block at a time. Never generate the whole project at once.
3. **"Predict first."** Before I run something non-trivial, ask me to predict the result.
4. **You write the code, then walk me through it.** For every block (including core logic like chunking, retrieval, eval scoring, and the agent loop), write the working code yourself, run it or its tests, then explain it section by section: what each part does, why you chose it over the alternatives, and what would break without it. Call out the lines an interviewer is most likely to ask about.
5. **When something breaks, explain before fixing.** When I paste an error, explain what's wrong and why, then fix it.
6. **Teach me to read errors** (last line of the traceback first).
7. **Be honest.** Say when you're unsure, when an API or model name may be outdated, or when a result looks suspicious. Check the current docs rather than guessing.
8. After each block, ask me an interview-style question about what we just built.

## Project: a claims/policy copilot
Scenario: claims examiners and service reps waste time searching long policy documents. We build:
1. **RAG** that answers policy questions with citations
2. **Structured extraction** from synthetic claim forms into validated JSON
3. **A small tool-calling agent** (search policy, calculate benefit, escalate to a human)
4. **Evaluation** (golden Q&A set, retrieval hit rate, faithfulness via LLM-as-judge, chunk-size comparison)
5. **A simple UI plus a one-page business case and Responsible AI notes**

## Technical decisions already made (don't change without asking)
- **Provider:** Gemini API on the free tier via the `google-genai` SDK. No paid keys.
- **Model name lives in `.env`** as `GEMINI_MODEL` (currently `gemini-3.5-flash-lite`) because model names get deprecated. Confirm names against Google AI Studio.
- **All LLM calls go through `src/llm.py` (`ask()`).** No other file touches the SDK. Add retry with exponential backoff for 429 errors, and log tokens and latency here.
- **All Pydantic schemas live in `src/schemas.py`.** Use schema-constrained output (`response_mime_type="application/json"` and `response_schema`) instead of asking for JSON in the prompt and parsing it.
- **No frameworks** (no LangChain/LlamaIndex). Build the patterns by hand so I understand them.
- **Embeddings:** local `sentence-transformers` (or Chroma's default); vector store: Chroma.
- **Code in `.py` files under `src/`, run from the VS Code/Cursor terminal.** Notebooks only (in `notebooks/`) for exploring retrieval and charting eval results, and they must import from `src/`.
- **Data:** only public documents (sample group disability/life certificates, DOL/FMLA guidance) and synthetic claims. The free tier may use inputs for training, so never real PII.

## Repo layout
```
benefitsiq/
├── src/            # llm.py, schemas.py, rag/, extraction/, agent/, eval/
├── data/docs/      # public policy PDFs (3-5 docs)
├── notebooks/
├── tests/
├── .env  .env.example  .gitignore  requirements.txt  README.md
```

## Status
Done: repo, venv, `.env` handling, `llm.py` with `ask()`, temperature/token experiments, learned that prompt-only JSON breaks (code fences) so we use schema-constrained output.
Done (Block 0): `PolicyAnswer` schema; `ask(schema=...)` returns validated instances, retries 429/5xx, logs every call to `logs/llm_calls.jsonl`, raises on truncated/empty/malformed output. Next: hallucination demo, then RAG.

## Remaining plan
| Block | Build | Concept |
|---|---|---|
| Hallucination demo | Ask about a document the model hasn't seen | Why RAG exists |
| RAG | Load docs, chunk, embed, store in Chroma, retrieve, answer with citations | Embeddings, chunking, retrieval vs. generation errors |
| Golden set | 15-30 hand-written Q&A pairs with source sections | What "correct" means |
| Evals | Retrieval hit rate, LLM-as-judge faithfulness, compare 2-3 chunk sizes | Evals over demos |
| Extraction | Synthetic claim forms to Pydantic model with a needs-review flag | Schema-constrained output, validation |
| Agent | Tools: `search_policy`, `calculate_benefit` (plain Python); escalate when low confidence | Tool calling, guardrails, when agents are the wrong choice |
| API/UI | Streamlit UI; FastAPI endpoint if time allows | Serving |
| Business case | README, one-page business case (baseline vs. target time, hours saved), Responsible AI section | KPIs, stakeholder communication |

If time runs short, priority is: RAG, golden set, evals, business case. Cut Docker, CI, cloud deployment, and predictive modeling (list them under "Next steps" in the README).

## Code standards
- Type hints, short docstrings, small functions, no hidden global state.
- Secrets only in `.env`; never print or commit keys. Check `git status` before commits.
- Commit after each working block with a clear message.
- Branches: one per block, named `assistant-{task}` (e.g. `assistant-llm-hardening`, `assistant-rag-pipeline`), created from `main`.
- Log every LLM call: prompt size, tokens, latency, errors.
- Add a few `pytest` tests for the deterministic parts (chunking, benefit calculation, schema validation).
- Always handle: empty retrieval results, truncated outputs, rate limits, malformed model output.

## Things to never do
- Don't write code you don't walk me through. Keep each block small enough to explain fully.
- Don't invent facts about insurance, regulations, or policy terms; cite the source document.
- Don't claim accuracy numbers without running the eval.
- Don't help me claim professional generative AI experience on my résumé. The project goes under "Projects" once it's real and on GitHub.
## Running code
Run everything from the repo root as a module so `src` imports resolve, e.g.:
```
python -m src.experiments.hello_llm
python -m src.experiments.step3_json
```
