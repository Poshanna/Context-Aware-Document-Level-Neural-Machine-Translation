# Context-Aware Document-Level Neural Machine Translation for Indian Languages

B.E. AI & ML Final-Year Project. A web-based document-level machine
translation system for **Telugu ↔ English** (with Hindi extensibility) that
compares a *sentence-level baseline* pipeline against a *context-aware*
pipeline which retrieves and injects relevant prior-sentence context to
improve co-reference, consistency, and discourse accuracy.

> **Academic honesty:** All model checkpoints used are *pretrained* (not
> trained in this project). All evaluation numbers are the actual output
> of `sacrebleu`, `bert-score`, and (optionally) `unbabel-comet`. See
> [project_rules.md](./project_rules.md).

---

## Quick Start

### Option A — Docker Compose (recommended)

```bash
# 1. Copy env template and fill in real values
cp .env.example .env
# Edit POSTGRES_PASSWORD and JWT_SECRET_KEY at minimum.

# 2. Build + start services (db, backend, frontend)
docker compose up --build

# 3. Verify
#   Backend (OpenAPI): http://localhost:8000/docs
#   Frontend:          http://localhost:5173
#   Health check:      curl http://localhost:8000/health
```

### Option B — Local development (no Docker)

See the dedicated sections below for `backend` and `frontend`.

---

## Architecture at a Glance

```
┌─────────────┐  REST   ┌────────────────────────┐  calls  ┌─────────────┐
│  React UI   │ ──────▶ │  FastAPI Routers       │ ──────▶ │  Services   │
│  (Tailwind) │         │  (validation only)     │         │  (logic)    │
└─────────────┘         └────────────────────────┘         └──────┬──────┘
                                                                 │
                  ┌────────────────┬───────────────┬─────────────┴───────────┐
                  ▼                ▼               ▼                         ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        ┌──────────────┐
        │  translation │  │   context    │  │ preprocessing│        │  evaluation  │
        │  (ML models) │  │  (retrieval) │  │  (doc parse) │        │  (metrics)   │
        └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        └──────┬───────┘
               │                 │                 │                       │
               ▼                 ▼                 ▼                       ▼
        Hugging Face       Sentence-BERT      PyMuPDF /          sacrebleu / bert-score /
        IndicTrans2 /      cosine k-NN        python-docx /     (optionally) COMET
        NLLB-200                              sentence splitter
                                                                 │
                                                        ┌────────▼────────┐
                                                        │  PostgreSQL DB  │
                                                        │ (users, jobs,   │
                                                        │  eval results)  │
                                                        └─────────────────┘
```

Full architectural details in:
  - [project_rules.md](./project_rules.md) — rules, boundaries, academic honesty
  - `docs/architecture.md` — design deep-dive (WIP)

---

## Repo Layout

```
context-aware-nmt/
├── backend/                FastAPI application (Python 3.11+)
│   ├── app/
│   │   ├── api/            HTTP routers + deps (NO business logic)
│   │   ├── core/           config, security, logging
│   │   ├── models/         SQLAlchemy ORM models
│   │   ├── schemas/        Pydantic request/response schemas
│   │   ├── services/       Business logic orchestration
│   │   ├── translation/    Translator ABC + IndicTrans2 / NLLB-200 impls
│   │   ├── context/        ContextRetriever ABC + SBERT / last-N impls
│   │   ├── preprocessing/  DocumentProcessor ABC + PDF/DOCX/TXT impls
│   │   ├── evaluation/     Metric ABC + BLEU/chrF/BERTScore/COMET impls
│   │   └── main.py         FastAPI app factory
│   ├── tests/              pytest (unit / integration / smoke)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/               React + Vite + Tailwind frontend
│   ├── src/
│   │   ├── components/     Shared UI primitives
│   │   ├── pages/          Route-level pages
│   │   ├── services/       API / helper modules
│   │   ├── hooks/          Custom React hooks
│   │   └── App.jsx
│   ├── package.json
│   └── Dockerfile
├── data/                   Datasets (raw/ and processed/ are gitignored)
│   ├── raw/                Original datasets (download manually)
│   ├── processed/          Preprocessed outputs
│   └── samples/            Committed tiny smoke-test files
├── experiments/            Reproducible run artifacts (config, metrics, logs)
│   ├── baseline/
│   ├── context_aware/
│   └── evaluation/
├── scripts/                Utility scripts (capture env, dataset prep, etc.)
├── docs/                   Documentation, architecture, thesis figures
├── .env.example            Template environment variables
├── .gitignore
├── docker-compose.yml
├── project_rules.md        NON-NEGOTIABLE project rules (read this first)
└── README.md
```

---

## Running the Backend Locally (No Docker)

```bash
cd backend
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate

pip install -r requirements.txt

# Copy vars to your shell (edit values first — or use `python-dotenv` in main.py)
# Set DATABASE_URL to `sqlite+aiosqlite:///./dev.db` for quick local development without Postgres.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit http://localhost:8000/docs for OpenAPI Swagger UI.

## Running the Frontend Locally (No Docker)

```bash
cd frontend
npm install
npm run dev      # Vite dev server on http://localhost:5173
```

Build for production:
```bash
npm run build
npm run preview
```

---

## Supported Features (Roadmap)

- [x] Skeleton + scaffolding
- [ ] Auth (register / login / JWT)
- [ ] Text translation: te↔en, baseline & context-aware, both model backends
- [ ] File translation: PDF, DOCX, TXT in → translated download
- [ ] Context visualization (which source sentences influenced each target)
- [ ] Per-user translation history + replay
- [ ] Evaluation dashboard: BLEU, chrF++, BERTScore, COMET (optional)
- [ ] Side-by-side baseline vs context-aware comparison
- [ ] Hindi hooks (en↔hi direction validation in registry)

---

## Evaluation

Evaluation is **opt-in, never auto-downloads data**. Place source/reference
files in `data/raw/` (or upload via the dashboard), then:

```bash
# Via API (recommended — records run in DB + UI)
# POST /api/evaluate/run  (see OpenAPI schema)

# Or via CLI helper script (WIP)
python scripts/run_evaluation.py \
  --src data/raw/test.te \
  --ref data/raw/test.en \
  --src-lang te --tgt-lang en \
  --model nllb200 \
  --pipelines baseline context_aware \
  --metrics bleu chrf bertscore \
  --output experiments/evaluation/run_001/
```

> **COMET:** Very GPU-heavy. Enable only if you have ≥ 16 GB VRAM and
> `ENABLE_COMET=1` in `.env`. Skipping is explicitly reported, never
> hidden or fabricated.

---

## Rules

**Everybody reads [project_rules.md](./project_rules.md).** It covers:
- Academic honesty (no fabricated metrics, no claimed training, honest comparisons)
- Secrets handling
- Clean-architecture boundaries
- Determinism / reproducibility
- Dataset / model storage policy
- Experiment hygiene
- Development workflow

---

## FAQ

**Q: Which model should I use for the demo?**
A: Start with `nllb200` (lighter, works on CPU in DEMO_MODE). For thesis
results, prefer `indictrans2` (higher Telugu quality).

**Q: I only have a laptop with no GPU — can I run it?**
A: Yes — set `FORCE_DEVICE=cpu` and `DEMO_MODE=1`. Expect ~5–10 min per
100 sentences; keep demo inputs small.

**Q: Can I add Hindi?**
A: The translator registry already validates hi↔en directions for
NLLB-200. UI dropdown + API field are the only places that need to be
unlocked later — see `app/translation/registry.py`.
