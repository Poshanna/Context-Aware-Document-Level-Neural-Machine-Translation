# Project Rules — Context-Aware Document-Level NMT (B.E. AI & ML Research)

This file codifies **non-negotiable rules** for this academic research project.
Every contributor (including automated tooling) must follow these.

---

## 1. Academic Integrity (HARD RULES — NO EXCEPTIONS)

1.1 **Never fabricate numbers.**
  - BLEU, chrF++, BERTScore, COMET, or any evaluation metric MUST be the
    actual output of the corresponding library's function for the exact
    inputs used. If you cannot compute a metric, report it as
    `null / N/A / not-computed` with an honest explanation.

1.2 **Never claim training that did not happen.**
  - This project uses **pretrained** models (IndicTrans2, NLLB-200,
    Sentence-BERT, BERTScore mBERT). The thesis, code comments, commit
    messages, and UI MUST explicitly use the words "pretrained checkpoint"
    and MUST NOT say "we trained", "we fine-tuned", or "our model"
    (unless fine-tuning is actually performed and logged).

1.3 **Clearly distinguish baseline vs. proposed (context-aware) method.**
  - Every evaluation table, chart, and textual description MUST label each
    pipeline as either `baseline` (sentence-level, no context) or
    `context-aware` (document-level, with context retrieval).
  - Comparisons MUST show both numbers side-by-side; never show only one.

1.4 **Do not auto-download datasets.**
  - Samanantar, Flores, or other large datasets MUST be downloaded
    **manually by the user** into `data/raw/` and verified on disk before
    any evaluation script runs. Scripts MUST fail fast with a clear
    message if expected files are missing, rather than downloading.
  - `data/samples/` may hold tiny smoke-test files (< 50 sentences each)
    that ARE allowed to be committed.

1.5 **Report every computational caveat.**
  - If COMET is skipped because `ENABLE_COMET=0`, say so explicitly in
    every place that lists metrics (dashboard, CSV, thesis).
  - If you run on CPU rather than GPU, note the impact on latency.
  - If a PDF has columns that broke paragraph order, note it.

---

## 2. Secrets & Credentials

2.1 **No secrets in source code.**
  - JWT secret, DB password, Hugging Face tokens, etc. MUST only live in
    `.env` (loaded via `pydantic-settings`). `.env` is gitignored.
  - `.env.example` MUST contain only placeholder values (`replace_me...`).

2.2 **If you add a new secret,**
  - (a) add the placeholder to `.env.example`,
  - (b) add the field to `backend/app/core/config.py`,
  - (c) grep the repo to verify there is no literal value in any `*.py`,
    `*.yml`, `*.yaml`, `*.json`, `*.jsx`, `*.tsx`, or Dockerfile.

2.3 **Never print secrets to logs.**
  - `print(...)`, `logger.debug(...)`, FastAPI `dumps()` of settings, and
    error responses MUST NOT leak secrets. `Settings` class in config MUST
    mark sensitive fields with `pydantic.SecretStr` where possible.

---

## 3. Clean Architecture & Code Boundaries

3.1 **Routers have NO business logic.**
  - `backend/app/api/routers/*.py` may only do: parameter parsing +
    validation, authorization checks, service call, response shaping.
  - They MUST NOT import from `torch`, `transformers`, `sqlalchemy`
    `session.execute`, or any ML library directly.
  - They MUST NOT contain `if/else` that encodes business rules; delegate
    to a service.

3.2 **Services own business logic and orchestrate.**
  - `backend/app/services/*.py` call repositories + ML/context/preprocessing
    modules. They MUST NOT import FastAPI `Request`/`Response` objects.

3.3 **ML / Context / Preprocessing / Evaluation are pure libraries.**
  - The subpackages `translation`, `context`, `preprocessing`,
    `evaluation` MUST have ZERO imports from `app.api`, `app.models`,
    `app.db` (if a DB layer is later added). They expose ABCs first,
    implementations second.

3.4 **Persistence lives in services or a separate `repositories/` layer.**
  - If DB queries grow large, extract them into a `repositories/` layer
    rather than scattering them inside services. Do NOT put them in
    routers.

---

## 4. Modularity / Swappability

4.1 **Every external system behind an ABC.**
  - Translator → `translation/base.py`
  - ContextRetriever → `context/base.py`
  - DocumentProcessor → `preprocessing/base.py`
  - Metric → `evaluation/base.py`

4.2 **Registry pattern, not hardcoded `if/elif` chains.**
  - Adding a new translator (or retriever, metric, processor) MUST be a
    single new file + one line in the corresponding registry. No editing
    existing `translate()` functions to branch again.

---

## 5. Determinism & Reproducibility

5.1 Evaluation runs MUST be **deterministic** by default.
  - `EVAL_SEED` (default 42) is set at the start of every eval run for
    `random`, `numpy.random`, `torch`.
  - Same inputs + same pipeline + same model + same seed → identical
    metrics. A test (`TR-8.2`) enforces this.

5.2 **Never use sampling / greedy with hidden nondeterminism.**
  - Translation uses fixed `num_beams=1` (greedy) OR `do_sample=False`
    + fixed seed for reproducibility. Any change to decoding strategy is
    a configuration field saved with the run.

---

## 6. Dataset & Model Storage

6.1 **Hugging Face cache lives in a Docker volume, not in the image.**
  - `HF_HOME` maps to a named volume. Never `COPY` model weights into a
    Docker image during `docker build`.

6.2 **`data/raw/` and `data/processed/` are gitignored except `.gitkeep`.**
  - Only `data/samples/` holds committed tiny files.

---

## 7. Experiment / Artifact Hygiene

7.1 Each run in `experiments/` must contain:
  - `config.json` — full snapshot of hyperparameters, model names, seed
  - `metrics.json` — outputs of every metric
  - `system_info.json` — CPU/GPU/RAM, library versions (obtainable via
    `scripts/capture_env.py`)
  - `README.md` — one paragraph of what this run is testing.

7.2 **Do not commit large checkpoints.** Large artifacts → external storage
(e.g., Google Drive, institutional repo) referenced by URL + SHA-256 in
a committed `manifest.json`.

---

## 8. Documentation & Theses

8.1 Every method choice (context strategy, truncation policy, language
code mapping) is documented in the code AND in the thesis chapter.
  - "We chose X because Y; alternatives A/B were rejected for reason Z."

8.2 If a hypothesis fails (e.g., context-aware performs WORSE on a test
set), **report it honestly** with a plausible explanation. In research,
null results are still results.

---

## 9. Development Workflow

9.1 **Phase order:** Foundations → Auth → ML core → Retrieval → API →
Evaluation → Frontend → Polish. Never start UI before backend service
layer works in pure Python.

9.2 **No merge to main with a failing test.** Keep CI green (or if you
don't have CI, run the test suite locally before pushing big changes).

9.3 **Large dependencies are optional.**
  - `unbabel-comet` is installed only if `ENABLE_COMET=1` or via a
    requirements extra. Do NOT force all users to install it.

---

## 10. Escalation Path

If a rule conflicts with reality (e.g., you absolutely need to commit a
medium artifact, or COMET results are mandatory for the committee),
**open a discussion in the team** and:
  - record the exception here with date + justification,
  - ensure every team member / evaluator is explicitly aware,
  - document it in the thesis "Methodology / Limitations" section.
