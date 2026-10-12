# Evidence and Verification Report

**Project Title:** Context-Aware Document-Level Neural Machine Translation for Indian Languages Using Semantic Retrieval and Gated Context Fusion  
**Target Venue:** IEEE Conference Publication Format  
**Verification Date:** October 2026  
**Auditor:** Senior Academic Researcher & Antigravity Research Agent  

---

## 1. Executive Summary

This report establishes the ground-truth mapping between every technical claim, equation, hyperparameter, and metric value in the research paper manuscript (`main.tex`) and the actual implementation artifacts, source code, and log files in the project workspace. 

Every reported number was verified directly from local JSON artifacts; no values were estimated, smoothed, or fabricated.

---

## 2. Phase 2: Methodology and Architecture Verification

| Methodological Attribute | Verified Value / Specification | Workspace Source of Truth | Verification Status |
|---|---|---|---|
| **Translation Backbone** | `facebook/nllb-200-distilled-600M` | `backend/app/translation/nllb200.py`<br>`experiments/context_adapter/training_report.json` | **VERIFIED** |
| **Context Encoder** | `sentence-transformers/all-MiniLM-L6-v2` | `scripts/inspect_context_adapter_dimensions.py`<br>`training_report.json` | **VERIFIED** |
| **Context Embedding Dim ($d_c$)** | $384$ | `inspect_context_adapter_dimensions.py` line 121 | **VERIFIED** |
| **NLLB Hidden Dim ($d_{model}$)** | $1024$ | `inspect_context_adapter_dimensions.py` line 69<br>`context_adapter_model.py` line 21 | **VERIFIED** |
| **Trainable Context Adapter** | Linear projection ($384 \to 1024$), LayerNorm, GELU, Dropout($0.1$), Concat, Gate Linear($2048 \to 1024$), Sigmoid, Residual add | `backend/app/services/context_adapter_model.py` lines 26--94 | **VERIFIED** |
| **Gate Bias Initialization** | Constant $-4.0$ ($\sigma(-4.0) \approx 0.01798 \approx 0$) | `backend/app/services/context_adapter_model.py` lines 45--48 | **VERIFIED** |
| **Trainable Parameters** | **$2,494,464$**<br>$\bullet$ Projection: $384 \times 1024 + 1024 = 394,240$<br>$\bullet$ LayerNorm: $1024 + 1024 = 2,048$<br>$\bullet$ Gate: $2048 \times 1024 + 1024 = 2,098,176$ | `experiments/context_adapter/training_report.json` line 12<br>`scripts/train_context_adapter.py` lines 399--405 | **VERIFIED** |
| **Frozen Modules** | NLLB-200 Backbone (Encoder \& Decoder) frozen.<br>MiniLM Context Encoder frozen. | `scripts/train_context_adapter.py` lines 359--361, 381--383<br>`training_report.json` lines 19--21 | **VERIFIED** |
| **Training Steps / Epochs** | 1 Epoch, 1,000 steps, batch size 1 | `training_report.json` lines 9--10, 16 | **VERIFIED** |
| **Optimizer \& Learning Rate** | AdamW, $\text{lr} = 1 \times 10^{-4}$ | `training_report.json` line 11<br>`scripts/train_context_adapter.py` line 411 | **VERIFIED** |
| **Final Training Loss** | $1.9976804350018502$ | `training_report.json` lines 13--15 | **VERIFIED** |
| **Target Label Shift** | Native Transformers 5.18.0 M2M100 right shift (\texttt{shift\_tokens\_right}) | `training_report.json` line 23<br>`scripts/train_context_adapter.py` lines 256--265 | **VERIFIED** |
| **Primary Language Direction** | English $\to$ Telugu (`eng_Latn` $\to$ `tel_Telu`) | `training_report.json`<br>`docs/baseline_nmt.md` | **VERIFIED** |
| **Document Dataset Resource** | DocHPLT (`train-00000-of-00030.parquet`, revision `76a591ec3c014d0ac4193a3cbdfc12689616a8d3`) | `data/processed/context_adapter/dataset_stats.json`<br>`docs/dochplt_evaluation.md` | **VERIFIED** |
| **Sentence Dataset Resource** | AI4Bharat Samanantar ($4,927,517$ clean en-te pairs) | `experiments/dataset_preprocessing_report.json` | **VERIFIED** |

---

## 3. Phase 3: Experimental Results Verification

### A. Primary Validation Subset (100 Examples, $k=2$)
*Source:* `experiments/evaluation/dochplt/20261007T163620Z-final-100/summary.json`

| Metric | Sentence Baseline | Context Adapter | Absolute Delta ($\Delta$) | Relative Delta (\%) | Status |
|---|---:|---:|---:|---:|---|
| **BLEU** | 7.34970429 | 13.98655412 | +6.63684983 | +90.3009\% | **VERIFIED** |
| **chrF++** | 25.10662704 | 31.93517151 | +6.82854446 | +27.1982\% | **VERIFIED** |
| **Failures** | 0 / 100 | 0 / 100 | 0 | -- | **VERIFIED** |
| **Decoding** | Greedy (`beams=1`, `sample=False`) | Greedy (`beams=1`, `sample=False`) | -- | -- | **VERIFIED** |

### B. Manually Reviewed Diagnostic Subset (12 Examples)
*Source 1:* `experiments/evaluation/dochplt/20261007T150249Z-82cae070/summary.json` (Context Adapter run)  
*Source 2:* `experiments/evaluation/dochplt/20261006T183245Z-794973e9/summary.json` (Naive Prefix run)  

| Model Pipeline | BLEU | chrF++ | $\Delta$ vs. Baseline BLEU | $\Delta$ vs. Baseline chrF++ | Status |
|---|---:|---:|---:|---:|---|
| **NLLB Baseline** | 41.27782045 | 53.48236184 | -- | -- | **VERIFIED** |
| **Naive Prefix Concatenation** | 10.21926722 | 37.02427942 | -31.05855323 | -16.45808242 | **VERIFIED** |
| **Context Adapter (Ours)** | 43.14826471 | 63.25637242 | +1.87044426 | +9.77401058 | **VERIFIED** |

> **Critical Note:** This 12-example diagnostic subset is kept completely distinct from the 100-example validation set in the manuscript, correctly categorized as an explanatory diagnostic of input conditioning failure modes.

### C. Context-Window Diagnostic ($k \in \{0, 1, 2\}$)
*Source:* `experiments/evaluation/dochplt/20261007T161325Z-context-window/summary.json`  
*Source script:* `scripts/create_final_result_graphs.py` lines 160--180

| Configuration | BLEU | chrF++ | $\Delta$ vs. Baseline BLEU | $\Delta$ vs. Baseline chrF++ | Status |
|---|---:|---:|---:|---:|---|
| **NLLB Baseline** | 7.34970429 | 25.10662704 | -- | -- | **VERIFIED** |
| **Adapter ($k=0$)** | 14.83014381 | 28.19750966 | +7.48043952 | +3.09088262 | **VERIFIED** |
| **Adapter ($k=1$)** | 14.55826863 | 31.73087789 | +7.20856434 | +6.62425085 | **VERIFIED** |
| **Adapter ($k=2$)** | 13.98655412 | 31.93517151 | +6.63684983 | +6.82854446 | **VERIFIED** |

> **Methodological Caveat:** As explicitly documented in `summary.json` line 4438, this diagnostic evaluated the single adapter checkpoint trained with $k \le 2$ under variable runtime context restrictions. The paper explicitly warns that this does not constitute three independently trained models.

### D. Per-Example Win / Loss / Tie Distribution (100 Examples)
*Source:* `experiments/evaluation/dochplt/20261007T163620Z-final-100/error_analysis.json` lines 6--25

| Metric | Adapter Wins | Baseline Wins | Ties | Status |
|---|---:|---:|---:|---|
| **Sentence-Level BLEU** | 52 | 14 | 34 | **VERIFIED** |
| **Sentence-Level chrF++** | 52 | 30 | 18 | **VERIFIED** |
| **Both Metrics Simultaneously** | 40 | 10 | -- | **VERIFIED** |
| **Average Sentence BLEU Delta** | \multicolumn{3}{c|}{+6.97241687} | **VERIFIED** |
| **Average Sentence chrF++ Delta** | \multicolumn{3}{c|}{+7.46779414} | **VERIFIED** |

---

## 4. Phase 4: Literature and Citation Audit

All 25 citations in `references.bib` were retrieved and cross-checked against authoritative indexes (ACL Anthology, ACM Digital Library, IEEE Xplore, PMLR, and arXiv).

| Citation Key | Verified Paper Title | Venue / Year | DOI / Identifier | Status |
|---|---|---|---|---|
| `ramesh-etal-2022-samanantar` | Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages | TACL 2022 | 10.1162/tacl_a_00452 | Confirmed |
| `costajussa2022nllb` | No Language Left Behind: Scaling Human-Centered Machine Translation | arXiv 2022 | 10.48550/arXiv.2207.04672 | Confirmed |
| `obrien2025dochplt` | DocHPLT: A Massively Multilingual Document-Level Translation Dataset | WMT 2025 / arXiv | arXiv:2502.13840 | Confirmed |
| `reimers-gurevych-2019-sentence` | Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | EMNLP 2019 | 10.18653/v1/D19-1410 | Confirmed |
| `wang2020minilm` | MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression | NeurIPS 2020 | Vol. 33, pp. 5776--5788 | Confirmed |
| `bapna-firat-2019-simple` | Simple, Scalable Adaptation for Neural Machine Translation | EMNLP 2019 | 10.18653/v1/D19-1165 | Confirmed |
| `houlsby2019parameter` | Parameter-Efficient Transfer Learning for NLP | ICML 2019 | PMLR 97:2790--2799 | Confirmed |
| `voita-etal-2018-context` | Context-Aware Neural Machine Translation Learns Anaphora Resolution | ACL 2018 | 10.18653/v1/P18-1117 | Confirmed |
| `voita-etal-2019-good` | When a Good Translation is Wrong in Context... | ACL 2019 | 10.18653/v1/P19-1116 | Confirmed |
| `maruf-etal-2021-survey` | A Survey on Document-level Neural Machine Translation: Methods and Evaluation | ACM CSUR 2021 | 10.1145/3441691 | Confirmed |
| `maruf-haffari-2018-document` | Document Context Neural Machine Translation with Memory Networks | ACL 2018 | 10.18653/v1/P18-2067 | Confirmed |
| `miculicich-etal-2018-document` | Document-Level Neural Machine Translation with Hierarchical Attention Networks | EMNLP 2018 | 10.18653/v1/D18-1325 | Confirmed |
| `zhang-etal-2018-improving` | Improving the Transformer Translation Model with Document-Level Context | EMNLP 2018 | 10.18653/v1/D18-1053 | Confirmed |
| `tiedemann-scherrer-2017-neural` | Neural Machine Translation with Extended Context | DiscoMT 2017 | 10.18653/v1/W17-4811 | Confirmed |
| `bawden-etal-2018-evaluating` | Evaluating Discourse Phenomena in Neural Machine Translation | NAACL 2018 | 10.18653/v1/N18-1118 | Confirmed |
| `gala2023indictrans` | IndicTrans2: Towards High-Quality and Accessible Machine Translation Models... | TMLR 2023 | ISSN 2835-8856 | Confirmed |
| `post-2018-call` | A Call for Clarity in Reporting BLEU Scores | WMT 2018 | 10.18653/v1/W18-6319 | Confirmed |
| `popovic-2017-chrf` | chrF++: words helping character n-grams | WMT 2017 | 10.18653/v1/W17-4770 | Confirmed |
| `vaswani2017attention` | Attention is All you Need | NeurIPS 2017 | Vol. 30, pp. 5998--6008 | Confirmed |
| `tu-etal-2018-learning` | Learning to Remember Translation History with a Continuous Cache | TACL 2018 | 10.1162/tacl_a_00030 | Confirmed |
| `kim-etal-2019-document` | When and Why is Document-level Context Useful in NMT? | DiscoMT 2019 | 10.18653/v1/D19-5403 | Confirmed |
| `fernandes-etal-2021-measuring` | Measuring and Increasing Context Usage in Context-Aware Machine Translation | ACL 2021 | 10.18653/v1/2021.acl-long.505 | Confirmed |
| `jean2017does` | Does Neural Machine Translation Benefit from Larger Context? | arXiv 2017 | 10.48550/arXiv.1704.05135 | Confirmed |
| `ma-etal-2020-simple` | A Simple and Effective Unified Encoder for Document-Level Machine Translation | ACL 2020 | 10.18653/v1/2020.acl-main.321 | Confirmed |
| `muller2018large` | A Large-Scale Test Set for Context-Aware Pronoun Translation (ContraPro) | WMT 2018 | 10.18653/v1/W18-6307 | Confirmed |

---

## 5. Artifact Audit Trail

| Deliverable File | Status | Description |
|---|---|---|
| `research_paper_output/main.tex` | Complete | Full IEEE conference paper source (505 lines, 10 sections, 5 tables, 4 figures). |
| `research_paper_output/references.bib` | Complete | Verified BibTeX database (25 fully verified citations). |
| `research_paper_output/IEEEtran.cls` | Present | Official IEEEtran LaTeX document class downloaded directly from CTAN mirrors (281 KB). |
| `research_paper_output/figures/` | Complete | 4 figures: `system_architecture.png`, `system_architecture.pdf`, `final_bleu_comparison.png`, `final_chrf_comparison.png`, `context_window_diagnostic.png`. |
| `research_paper_output/evidence_and_verification.md` | Complete | This mapping report. |
| `research_paper_output/missing_information.md` | Complete | Audit of unresolved details and required student/guide inputs. |
| `research_paper_output/README.md` | Complete | Step-by-step instructions for local and Overleaf PDF compilation. |
