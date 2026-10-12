\# Context-Aware Document-Level Neural Machine Translation for Indian Languages



\*\*An academic research project exploring context-aware English-to-Telugu machine translation using semantic context retrieval and gated context fusion.\*\*






\## Overview



Traditional sentence-level machine translation processes each sentence independently. As a result, it can struggle with pronoun references, terminology consistency, and meanings that depend on earlier sentences.



This project investigates whether contextual information from surrounding sentences can improve translation quality. It combines a pretrained neural machine translation model, semantic context retrieval, and a lightweight trainable context adapter.



The repository includes a FastAPI backend, React frontend, dataset preparation and evaluation utilities, automated tests, experiment outputs, and IEEE research paper source files.



\*\*Primary research direction:\*\* English → Telugu.



This is an academic research prototype. Reported results are preliminary and should not be interpreted as proof of generalization across all documents, domains, or language pairs.



\## Key Features



\- Context-aware translation using information from preceding sentences.

\- Pretrained NLLB-200 translation backbone.

\- Semantic sentence embeddings using Sentence Transformers.

\- Trainable gated context adapter for context fusion.

\- FastAPI backend and React/Vite frontend.

\- Document preprocessing utilities for TXT, PDF, and DOCX workflows.

\- Evaluation utilities for BLEU, chrF++, BERTScore, and optional COMET.

\- Dataset adapters and auditing utilities for resources including DocHPLT and Samanantar.

\- Experiment reports, diagnostic outputs, tests, and research paper source.



\## System Architecture



```text

Document / Paragraph

&#x20;        |

&#x20;        v

&#x20; Sentence Segmentation

&#x20;        |

&#x20;        +----------------------------+

&#x20;        |                            |

&#x20;        v                            v

&#x20;Current Source Sentence      Previous-Sentence Context

&#x20;        |                            |

&#x20;        v                            v

&#x20;  NLLB Encoder              Semantic Context Retrieval

&#x20;        |                            |

&#x20;        |                     MiniLM Embeddings

&#x20;        |                            |

&#x20;        |                     Context Adapter

&#x20;        |                            |

&#x20;        +------- Gated Fusion -------+

&#x20;                     |

&#x20;                     v

&#x20;                NLLB Decoder

&#x20;                     |

&#x20;                     v

&#x20;             Telugu Translation

```



\### Core Models



| Component | Model / Configuration |

|---|---|

| Translation backbone | `facebook/nllb-200-distilled-600M` |

| Semantic encoder | `sentence-transformers/all-MiniLM-L6-v2` |

| Semantic embedding dimension | 384 |

| NLLB encoder hidden dimension | 1024 |

| Context adapter | Trainable gated fusion module |

| Backend | FastAPI |

| Frontend | React, Vite |



In the reported adapter experiment, the pretrained NLLB backbone and MiniLM encoder were frozen while the context adapter was trained.



\## Preliminary Evaluation Results



The following results are from a bounded validation experiment involving 100 examples.



| System | BLEU | chrF++ |

|---|---:|---:|

| NLLB sentence-level baseline | 7.35 | 25.11 |

| NLLB with gated context adapter | 13.99 | 31.94 |

| Absolute change | +6.64 | +6.83 |



\### Experimental Configuration



\- Training examples: 1,000

\- Training duration: 1 epoch

\- Trainable adapter parameters: 2,494,464

\- NLLB backbone: frozen

\- MiniLM encoder: frozen

\- Evaluation examples: 100



These are preliminary experimental results. They do not establish statistical significance or guarantee improvements on other datasets. Consult the committed experiment reports for configurations, examples, and limitations.



\## Technology Stack



\- \*\*Programming:\*\* Python, JavaScript

\- \*\*Machine Learning:\*\* PyTorch, Hugging Face Transformers

\- \*\*Translation:\*\* NLLB-200

\- \*\*Semantic Retrieval:\*\* Sentence Transformers

\- \*\*Backend:\*\* FastAPI

\- \*\*Frontend:\*\* React, Vite

\- \*\*Evaluation:\*\* BLEU, chrF++, BERTScore, optional COMET

\- \*\*Testing:\*\* pytest

\- \*\*Deployment and environment:\*\* Docker, Docker Compose



\## Repository Structure



```text

.

├── context-aware-nmt/

│   ├── backend/

│   │   ├── app/

│   │   │   ├── api/

│   │   │   ├── context/

│   │   │   ├── evaluation/

│   │   │   ├── preprocessing/

│   │   │   ├── services/

│   │   │   └── translation/

│   │   ├── tests/

│   │   └── requirements.txt

│   ├── frontend/

│   │   └── src/

│   ├── scripts/

│   ├── data/

│   │   └── samples/

│   ├── docs/

│   ├── experiments/

│   ├── docker-compose.yml

│   └── README.md

├── research\_paper\_output/

│   ├── main.tex

│   ├── references.bib

│   └── figures/

├── .trae/

│   └── specs/

└── generate\_architecture\_diagram.py

```



\## Getting Started



\### 1. Clone the repository



```bash

git clone https://github.com/Poshanna/Context-Aware-Document-Level-Neural-Machine-Translation.git



cd Context-Aware-Document-Level-Neural-Machine-Translation

```



\### 2. Open the implementation directory



```bash

cd context-aware-nmt

```



Follow the detailed setup instructions in \[`context-aware-nmt/README.md`](./context-aware-nmt/README.md).



For Docker Compose, review the environment template before starting the services:



```bash

\# Copy the environment template

copy .env.example .env



\# Start services after configuring required environment variables

docker compose up --build

```



The `copy` command above is for Windows Command Prompt. On Linux or macOS, use `cp .env.example .env`.



Never commit real API keys, database passwords, JWT secrets, or other credentials.



\### 3. Run the backend locally



The backend setup is documented in the implementation README. In general, create and activate a Python virtual environment, install the backend requirements, configure environment variables, and start the FastAPI application.



```bash

cd backend

pip install -r requirements.txt

uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

```



When configuration is complete, the API documentation is available at:



`http://localhost:8000/docs`



\### 4. Run the frontend locally



Open another terminal:



```bash

cd context-aware-nmt/frontend

npm install

npm run dev

```



Use the URL printed by Vite. The frontend may require the backend URL to be configured correctly.



\### 5. Run tests



From the backend directory, after installing the dependencies:



```bash

pytest

```



Some model-backed tests and evaluation scripts may require pretrained weights, datasets, or optional dependencies.



\## Datasets and Reproducibility



The project contains dataset adapters, preparation scripts, and audit documentation for resources including DocHPLT and Samanantar.



\- Obtain datasets from their official sources.

\- Follow the applicable dataset licenses and attribution requirements.

\- Raw corpora and large processed datasets should not be committed to Git.

\- Review the experiment configuration and provenance before interpreting reported metrics.

\- Use the committed scripts and reports to reproduce the corresponding experiments.



BLEU and chrF++ are used in the reported validation experiment. BERTScore and COMET integrations are also present, but their availability in the codebase does not imply that every experiment was evaluated with those metrics.



\## Research Paper



The LaTeX source, bibliography, and figures for the IEEE-style research paper are available in \[`research\_paper\_output/`](./research\_paper\_output/).



\- \[Paper source](./research\_paper\_output/main.tex)

\- \[References](./research\_paper\_output/references.bib)

\- \[Implementation documentation](./context-aware-nmt/README.md)



\## Limitations and Future Work



\- Expand training and evaluation datasets.

\- Evaluate across more documents and domains.

\- Investigate discourse phenomena such as coreference and terminology consistency.

\- Conduct stronger controlled ablations and statistical significance testing.

\- Validate translations with fluent Telugu speakers.

\- Verify support and quality for each additional language direction.

\- Improve reproducibility and deployment documentation.



Translation output should be reviewed by qualified speakers before research conclusions or high-stakes decisions are based on it.



\## Academic Integrity



The translation backbone and semantic encoder are pretrained models. The reported experiment trains the context adapter; it does not train the entire NLLB model from scratch.



All performance claims should be reported with their evaluation conditions and limitations. When reusing datasets or pretrained models, cite their original publications and comply with their licenses.



\## Contributors



Developed as a B.E. Artificial Intelligence and Machine Learning major project at \*\*Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad\*\*.



\---



For detailed installation instructions, configuration, API usage, and implementation-specific notes, see \[`context-aware-nmt/README.md`](./context-aware-nmt/README.md).



