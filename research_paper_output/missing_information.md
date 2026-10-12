# Missing Information and Action Items for Final Submission

**Project Title:** Context-Aware Document-Level Neural Machine Translation for Indian Languages Using Semantic Retrieval and Gated Context Fusion  
**Target Venue:** IEEE Conference Publication  
**Document Status:** Complete Academic Manuscript Prepared; Remaining Pre-Submission Actions Detailed Below  

---

## 1. Remaining Author and Institutional Metadata Actions

The author names (**Durki Poshanna** and **G. Vamshi Krishna**), institution (**Chaitanya Bharathi Institute of Technology (CBIT), Hyderabad**), and faculty project guide (**Dr. G. Narayana**) have been updated in `research_paper_output/main.tex`.

Before camera-ready submission or conference upload, the following specific details must be inserted into `main.tex` (lines 30--38):

1. **Student Registration / Roll Numbers:**
   - In Line 33 of `main.tex`:
     ```latex
     Roll / Reg. No.: [Roll No. -- Durki Poshanna], [Roll No. -- G. Vamshi Krishna]\\
     ```
     Replace `[Roll No. -- Durki Poshanna]` and `[Roll No. -- G. Vamshi Krishna]` with your official CBIT roll / registration numbers (e.g., `160121729001`, `160121729002`).

2. **Student Email Addresses:**
   - In Line 34 of `main.tex`:
     ```latex
     Email: [Email -- Durki Poshanna], [Email -- G. Vamshi Krishna]
     ```
     Replace placeholders with your official college or academic email addresses.

3. **Faculty Guide Designation:**
   - In Line 35 of `main.tex`:
     ```latex
     \IEEEauthorblockA{\IEEEauthorrefmark{2}Project Guide, [Designation / Academic Title (e.g., Associate Professor)], Department of Artificial Intelligence and Machine Learning\\
     ```
     Confirm Dr. G. Narayana's exact faculty designation (*Associate Professor*, *Assistant Professor*, or *Professor*) and replace the placeholder.

4. **Faculty Guide Email Address:**
   - In Line 37 of `main.tex`:
     ```latex
     Email: [Email -- Dr. G. Narayana]
     ```
     Replace `[Email -- Dr. G. Narayana]` with his official institutional email address (e.g., `gnarayana@cbit.ac.in`).

5. **Faculty Guide Co-Authorship vs. Supervisory Attribution Confirmation:**
   - **Academic Integrity Check:** IEEE guidelines require that all listed co-authors have actively participated in manuscript drafting and explicitly consented to co-authorship.
   - **Action:** Confirm with Dr. G. Narayana and your conference guidelines whether he should be:
     - **Option A (Co-Author):** Retained in the author block with the `(Project Guide)` note removed from `\IEEEauthorblockN`.
     - **Option B (Supervisory Attribution - Current):** Retained with the explicit `(Project Guide)` label to prevent unconfirmed co-authorship attribution.
     - **Option C (Student-Only Authors):** Removed from the `\author{}` block and acknowledged in the `\section*{Acknowledgment}` section (template provided as comments in `main.tex` lines 501--504).

---

## 2. Compilation and Submission Instructions

### Local Compilation Status
No local LaTeX engine (`pdflatex`, `xelatex`, `lualatex`, `latexmk`) was detected on the system's `PATH`.

### Overleaf Compilation Steps (Recommended)
1. Navigate to [Overleaf](https://www.overleaf.com) and log in.
2. Click **New Project** $\to$ **Upload Project**.
3. Create a ZIP archive or upload all files from `research_paper_output/`:
   - `main.tex`
   - `references.bib` (25 verified entries)
   - `IEEEtran.cls`
   - `figures/` directory (all 5 files: `system_architecture.png`, `system_architecture.pdf`, `final_bleu_comparison.png`, `final_chrf_comparison.png`, `context_window_diagnostic.png`)
4. Ensure the Overleaf compiler is set to **pdfLaTeX** (default) with TeX Live 2024 or 2025.
5. Click **Recompile** to produce `main.pdf`.

---

## 3. Pre-Submission Checklist

- [x] Verified translation backbone: `facebook/nllb-200-distilled-600M` ($600\text{M}$ parameters).
- [x] Verified context encoder: `sentence-transformers/all-MiniLM-L6-v2` ($384$ dim).
- [x] Verified trainable parameters: Exactly $2,494,464$ parameters (0.4\% of backbone capacity).
- [x] Verified primary validation metrics (100 ex.): Baseline (7.35 BLEU / 25.11 chrF++) vs. Adapter (13.99 BLEU / 31.94 chrF++).
- [x] Verified diagnostic metrics (12 ex.): Baseline (41.28 BLEU) vs. Naive Prefix (10.22 BLEU) vs. Adapter (43.15 BLEU).
- [x] Verified context horizon metrics: $k=0$ (14.83), $k=1$ (14.56), $k=2$ (13.99).
- [x] Verified bibliography: All 25 citations in `main.tex` verified and mapped in `references.bib` with zero unresolved keys.
- [x] Verified figure assets: All 4 figures verified with exact paths and resolutions in `figures/`.
- [x] Verified LaTeX syntax: All environments, math delimiters, and braces verified balanced.
- [x] Set author names (**Durki Poshanna**, **G. Vamshi Krishna**) and institution (**CBIT, Hyderabad**).
- [x] Set project guide (**Dr. G. Narayana**) with explicit supervisory attribution.
- [ ] Insert actual student roll numbers in `main.tex` line 33.
- [ ] Insert actual student email addresses in `main.tex` line 34.
- [ ] Confirm and insert Dr. G. Narayana's designation in `main.tex` line 35.
- [ ] Insert Dr. G. Narayana's institutional email address in `main.tex` line 37.
- [ ] Confirm faculty guide co-authorship policy with Dr. G. Narayana and conference rules.
- [ ] Compile final `main.pdf` on Overleaf.
