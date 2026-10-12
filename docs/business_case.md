# Business case: policy and claims copilot for disability claims examiners

**Recommendation:** run a 6-week shadow pilot with one LTD policy family before any wider rollout. The prototype answers about 7 in 10 policy questions with a verifiable page citation and routes the rest to the examiner. The time savings below are a model built on example inputs, so the pilot's first job is to measure the real baseline.

## The problem
Examiners and service reps answer coverage questions (elimination periods, benefit percentages, offsets, exclusions) by searching 25 to 40 page certificates. Carriers word the same concept differently ("Elimination Period" vs "Benefit Waiting Period"), and the answer often carries a condition that changes the outcome ("the later of 90 days or the end of sick leave"; "36 months, or Social Security Normal Retirement Age if later"). Errors here change when and how much a claimant is paid.

## What the prototype does
- **Answers policy questions with page citations**, and says "not found" instead of guessing.
- **Extracts claim forms** into structured fields and flags missing, inconsistent, or ineligible data with rules.
- **Estimates benefits** with exact arithmetic (percent, cap, offsets, minimum), showing every step.
- **Never decides claims.** Decision requests and uncertain cases are escalated to a person.

## Measured on the evaluation set (32 hand-checked questions, 6 synthetic claim forms)
| Measure | Result |
|---|---|
| Answerable questions answered correctly with a citation | 20 of 29 (69%) |
| Answerable questions where it said "not found" and flagged review | 7 of 29 |
| Partially correct answers (a condition left out) | 2 of 29: 1 flagged, 1 not flagged |
| Unanswerable questions correctly declined | 3 of 3 |
| Answers containing claims not in the cited text (LLM judge, 7/8 agreement with human grading) | 0 |
| Planted claim-form problems caught (unsigned, date errors, salary mismatch, under minimum hours, illegible) | 7 of 7, no false flags |
| Typical cost per question | ~1,150 input + ~85 output tokens, ~1 second |

These are small samples from one prototype run. They show the behavior; they are not production accuracy estimates.

## Time savings model (example inputs: replace with pilot measurements)
Minutes saved per lookup = (share answered with citation × (baseline minutes − minutes to verify the cited page)) − (share that falls back × overhead minutes).

| Input | Example value | Source |
|---|---|---|
| Examiners | 20 | **Assumption** |
| Policy lookups per examiner per day | 8 | **Assumption** |
| Minutes per lookup today | 10 | **Assumption** |
| Minutes to verify a cited answer | 2 | **Assumption** |
| Extra minutes when it falls back to manual search | 0.5 | **Assumption** |
| Share answered with a citation | 69% | **Measured** |

With these inputs: about **5.4 minutes saved per lookup**, about **3,300 hours a year** across 36,800 lookups (roughly 1.8 full-time examiners). Model usage would be about 45 million tokens a year; multiply by the provider's current price per token to get the run cost.

**Sensitivity (hours saved per year):**

| Minutes per lookup today | 50% cited | 69% cited (measured) | 85% cited |
|---|---|---|---|
| 5 | 770 | 1,170 | 1,520 |
| 10 | 2,300 | 3,290 | 4,130 |
| 15 | 3,830 | 5,400 | 6,730 |

The savings depend far more on today's lookup time than on model accuracy, which is why the pilot must measure the baseline first.

## Risks and how the design handles them
| Risk | Control |
|---|---|
| Confident wrong answer | Citations are checked against retrieved text; low-support answers are flagged; examiners verify the cited page |
| Answer omits a condition (seen in 2 of 29) | Prompt rule to include conditions; partial answers are the main metric to watch in the pilot |
| Automated claim decisions | The system has no approve or deny capability; decision requests escalate |
| Sensitive data | Prototype uses public certificates and synthetic claims only; production needs an approved, non-training model endpoint |
| Over-trust | The UI shows why review is needed, and never shows a plain "verified" state |

## Pilot plan and success criteria
1. **Weeks 1 to 2, baseline:** time a sample of real lookups without the tool.
2. **Weeks 3 to 6, shadow mode:** examiners use the tool alongside the current process; every answer is checked.
3. **Go / no-go:** median lookup time down at least 30%; zero unflagged wrong answers in the audited sample; examiners rate citations useful; escalation rate understood and acceptable.
