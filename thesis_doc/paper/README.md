# `thesis_doc/paper/` — the journal paper (professor's template)

Built on the **professor's Elsevier `elsarticle` template** (March 2026). The
structure, macros, and rules below are *mandatory* — the professor grades to them.

**Paper first → professor approval → thesis doc.** Venue: *Information Fusion*
(aspirational) / *Expert Systems with Applications* (realistic).

## Files

| File | What |
|------|------|
| `main.tex` | the manuscript on the professor's template; front matter + abstract + RQs filled, body kept as template to fill section-by-section |
| `Bibliography.bib` | `\bibliography{Bibliography}` — published-venue anchors, flagged VERIFY |
| *(no `references.bib`)* | removed — the template uses `Bibliography.bib` |

## Hard rules (from the template's "General instructions")

- **≥30 pages** excluding references. This is a long report, not a short paper.
- **7–10 sentences per paragraph.**
- **One sentence per line** in the LaTeX source.
- **Comment, don't delete:** keep template text, comment it out, write yours below.
  (Delete only the *General instructions* section, at the very end.)
- **References: 25–50, ≥80% after 2021, journals/conferences** (IEEE, Springer,
  ACM, PLoS, Elsevier, MDPI). **Preferably NO arXiv / websites / thesis /
  ResearchGate.** BibTeX from Google Scholar.
- **Figures** in PowerPoint/Visio/Canva/DrawIO → **PDF, ≥300 ppi**, B&W-friendly;
  keep sources in a `Source/` folder, code/data in `Codes/` and `Datasets/`.
- **Tables** built in Excel → tablesgenerator.com → LaTeX.
- Every figure/table must be **referred to in the main text**; captions
  **self-explanatory**.

## Required section structure (exact)

```
Front: Title · Authors · Abstract (2 bg / 2 gap+problem / 2 method / 3 results / 1 contribution)
       · Graphical Abstract · Highlights · Keywords
1 Introduction → 1.1 Gap Analysis · 1.2 Research Questions · 1.3 Problem Statement
                 1.4 Novelty · 1.5 Significance
2 Literature Review → 2.1–2.4 Technique sections + comparison Table
3 Methodology → 3.1 Dataset · 3.2 Detailed Methodology · 3.3 Evaluation Metrics
                3.4 Experimental settings (+ config Table, architecture Fig)
4 Results → RQ1 … RQ7 subsections (results ONLY; + contemporary-method comparison)
5 Discussion → 5.1 Limitations · 5.2 Future Directions
6 Conclusion · Declarations · References
```

## Results tracker (filled vs pending)

| Part | Status |
|------|--------|
| Front matter (title, authors, affiliations, journal, keywords) | 🟢 done |
| Abstract (structure + 4 non-result sentences) | 🟢 drafted · ⬜ 3 result + 1 contribution sentences await runs |
| 1.2 Research Questions (our 7 RQs) + 1.4 Novelty bullets | 🟢 done |
| 1.1 Gap / 1.3 Problem / 1.5 Significance / Intro paragraphs | ⬜ to draft |
| 2 Literature Review (2.1–2.4 + matrix) | 🟡 with RQ1 survey |
| 3 Methodology (prose, equation drafted; figures/tables) | 🟡 prose stable; figures pending |
| 4 Results (RQ1–RQ7) | 🟡 RQ3 expl-aware arm logged (`results/RESULTS.md`); awaiting rq2 generative pair |
| 5 Discussion / 6 Conclusion / Declarations | ⬜ after results |

## Open issues to resolve

1. **Bibliography vs. arXiv rule.** Foundational VLM works (Qwen2-VL, PaliGemma)
   and the recent faithfulness papers (ChartHal, EDCT, Chart-RVR) are arXiv-only.
   Either find published versions or ask the professor whether a few foundational
   arXiv cites are acceptable. Bulk of refs should be 2022–2025 journal/conference.
2. **Figures** must be hand-made (PPT/Canva → PDF). The template's sample figures
   are neutralized (`\todo` placeholders) so it compiles; replace as we go.

## Compiling

- **Overleaf:** upload `main.tex` + `Bibliography.bib` (+ a `Figures/` folder when
  you have figures). It pulls `elsarticle`. The `\todo{}` macro marks open items.
- **Local:** `latexmk -pdf main.tex` (needs `elsarticle`, `pifont`, `adjustbox`).
