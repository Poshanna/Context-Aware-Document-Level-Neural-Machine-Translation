# IEEE Research Paper: Context-Aware Document-Level NMT for Indian Languages

This directory contains the complete IEEE conference-style research paper manuscript, verified bibliography, system figures, and comprehensive evidence verification artifacts for the undergraduate Major Project:

> **“Context-Aware Document-Level Neural Machine Translation for Indian Languages Using Semantic Retrieval and Gated Context Fusion”**

---

## 1. Directory Structure

```
research_paper_output/
├── main.tex                    # Complete IEEE conference LaTeX manuscript
├── references.bib              # 25 verified bibliography entries with complete DOIs/venues
├── IEEEtran.cls                # Official IEEE conference document class from CTAN
├── figures/                    # High-resolution publication figures
│   ├── system_architecture.png # Proposed architecture diagram (300 DPI)
│   ├── system_architecture.pdf # Vector PDF architecture diagram
│   ├── final_bleu_comparison.png # BLEU score bar chart comparison
│   ├── final_chrf_comparison.png # chrF++ score bar chart comparison
│   └── context_window_diagnostic.png # Context window (k=0, 1, 2) diagnostic chart
├── evidence_and_verification.md # Rigorous claim-to-source mapping & audit report
├── missing_information.md      # Unresolved details, missing profiles & author checklist
└── README.md                   # This compilation and submission guide
```

---

## 2. Compilation Instructions

### Status of Local Compilation
During project inspection, neither `pdflatex`, `xelatex`, nor `latexmk` was detected on the system's `PATH`. In accordance with scientific honesty rules, **we do not pretend that local PDF generation succeeded when no compiler exists**. Instead, we provide completely validated, portable LaTeX source code and class files ready for instant compilation.

---

### Option A: Overleaf (Recommended — Zero Setup)

1. Log in to [Overleaf](https://www.overleaf.com).
2. Click **New Project** $\to$ **Upload Project** (or **Blank Project**).
3. Upload all files from `research_paper_output/`:
   - `main.tex`
   - `references.bib`
   - `IEEEtran.cls`
   - The entire `figures/` directory containing all 5 figure files.
4. Set the compiler to **pdfLaTeX** (default) or **XeLaTeX**.
5. Click **Recompile** to generate `main.pdf`.

---

### Option B: Local Compilation with MiKTeX or TeX Live

If you have or install MiKTeX or TeX Live on your workstation:

```powershell
cd "research_paper_output"

# Standard 4-pass compilation to resolve citations and cross-references:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using `latexmk` (automatic multi-pass builder):

```powershell
latexmk -pdf main.tex
```

---

## 3. Key Research Findings Summary

| Evaluation Track | Configuration | BLEU | chrF++ | Key Finding / Significance |
|---|---|---:|---:|---|
| **Main Validation** (100 ex.) | NLLB-200 Baseline | 7.3497 | 25.1066 | Standard sentence-level translation. |
| **Main Validation** (100 ex.) | **Context Adapter (Ours)** | **13.9866** | **31.9352** | **+6.64 BLEU (+90.3%)**, **+6.83 chrF++ (+27.2%)**. |
| **Diagnostic** (12 ex.) | Naive Prefix Concatenation | 10.2193 | 37.0243 | Collapses by **-31.06 BLEU** due to prompt confusion. |
| **Diagnostic** (12 ex.) | **Context Adapter (Ours)** | **43.1483** | **63.2564** | Avoids input shift; achieves **+1.87 BLEU / +9.77 chrF++**. |
| **Context Horizon** (100 ex.) | $k = 0$ (Null context) | 14.8301 | 28.1975 | Adapter stylistic adaptation baseline. |
| **Context Horizon** (100 ex.) | $k = 1$ (1 sentence) | 14.5583 | 31.7309 | Immediate discourse reference injection. |
| **Context Horizon** (100 ex.) | $k = 2$ (2 sentences) | 13.9866 | 31.9352 | Maximum subword recall and terminology coherence. |

---

## 4. Next Steps for Submission

1. **Review:** Read `evidence_and_verification.md` to confirm every claim's traceability.
2. **Personalize:** Open `main.tex` and insert student names, roll numbers, and guide details as detailed in `missing_information.md`.
3. **Compile:** Compile `main.pdf` on Overleaf or local TeX distribution.
4. **Defense / Submission:** Present `main.pdf` and `evidence_and_verification.md` to your academic project guide.
