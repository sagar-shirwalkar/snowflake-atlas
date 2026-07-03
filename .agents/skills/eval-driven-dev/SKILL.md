---
name: eval-driven-dev
description: Improve AI applications with evaluation-driven development. Define eval criteria, instrument the app, build golden datasets, run pixie tests, and produce actionable improvements. Use when the user asks to set up QA, add tests/evaluations, benchmark, or improve quality for any Python project that calls an LLM.
disable-model-invocation: false
---

Set up automated evaluation pipelines for Python LLM applications using `pixie-qa`. The eval exercises the **real application code** (routing, prompt assembly, LLM calls, response formatting) with controlled input data — nothing is mocked or stubbed except the external data sources the app reads.

**Rule: The app's LLM calls must go to a real LLM.** Do not replace, mock, stub, or intercept the LLM. Replacing the LLM makes the evaluation tautological.

## Leading words

- **Instrument** — Add `wrap()` calls at the app's data boundaries so the eval harness can inject controlled inputs and capture outputs. Makes the app testable without changing its logic.
- **Golden dataset** — A curated set of inputs with expected behavior characteristics, covering the app's capabilities and known failure modes.
- **Score** — An evaluator's numeric judgment of an output (LLM-as-judge, similarity, deterministic check). Scores produce pass/fail decisions.

## Phases

### PHASE 1: Understand and define

Analyze the project, identify its entry points, and define eval criteria derived from real failure modes. Produces three artifacts in `pixie_qa/`: project analysis, entry point, and eval criteria.

### PHASE 2: Instrument and reference trace

Add `wrap()` calls at data boundaries, implement a `Runnable` class that invokes the real entry point, and capture a reference trace that proves instrumentation works.

### PHASE 3: Define evaluators

Map each eval criterion to a scoring function — built-in `pixie` evaluators, LLM-as-judge agent evaluators, or manual custom functions for deterministic checks.

### PHASE 4: Build dataset

Create JSON entries that tie together the Runnable, evaluators, and use cases. Each entry specifies inputs, expected data, and which evaluators to apply.

### PHASE 5: Run tests

Execute `pixie test` and fix mechanical issues. Once tests produce real evaluator scores, proceed to analysis.

### PHASE 6: Analyze outcomes

Complete pending evaluations, analyze per-dataset and per-run results, and produce a prioritized action plan grounded in concrete data.

## Reference Files

| File | Contents |
|------|----------|
| `references/1-a-project-analysis.md` | Project analysis sub-step |
| `references/1-b-entry-point.md` | Entry point detection sub-step |
| `references/1-c-eval-criteria.md` | Eval criteria definition sub-step |
| `references/2a-instrumentation.md` | Instrumentation with wrap() |
| `references/2b-implement-runnable.md` | Runnable implementation |
| `references/2c-capture-and-verify-trace.md` | Reference trace capture |
| `references/3-define-evaluators.md` | Evaluator definition |
| `references/4-build-dataset.md` | Dataset construction |
| `references/5-run-tests.md` | Test execution |
| `references/6-analyze-outcomes.md` | Outcome analysis |
| `references/evaluators.md` | Evaluator reference |
| `references/testing-api.md` | pixie testing API reference |
| `references/wrap-api.md` | wrap() API reference |
| `resources/setup.sh` | Script to initialize pixie environment |
| `resources/verify_step6_completion.py` | Verifier for Step 6 completion |

## Detailed Workflow

The detailed 6-step workflow lives in each step's reference file. Run them in order — each step's instructions assume the previous step is complete. See the reference files for full details.
