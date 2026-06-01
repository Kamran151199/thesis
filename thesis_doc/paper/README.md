# `thesis_doc/paper/` — the journal paper (living document)

The publishable manuscript distilled from the thesis. **Paper first → professor
approval → thesis doc.** Target: *Information Fusion* (aspirational) / *Expert
Systems with Applications* (realistic) — both Elsevier `elsarticle`.

## Who does what

| You | Me (the journal-keeper) |
|-----|--------------------------|
| run experiments, evals, code | own `main.tex` + `references.bib` |
| decide the science | draft prose, build tables/figures, manage the bibliography |
| hand over `results.json` / paste numbers | turn results into paper content, keep it consistent |
| read papers for related work | integrate them + add `.bib` entries |

**How to feed me a result:** after a run, point me at its
`outputs/<name>/results.json` (or paste the metrics) and tell me the config. I
update the right table/figure/prose and note what's still pending.

## Honesty rules (non-negotiable)

- Every quantitative claim carries the **1-seed variance caveat**.
- The contribution is **empirical systematization + a focused hypothesis test**,
  not a novel method — no over-claiming.
- Position against the real neighbors: ChartHal, EDCT, Chart-RVR, the 2023
  NLE-faithfulness benchmark, Multimodal-CoT.
- Lead the explanation-quality story with **Qwen2-VL** (BLIP-2 is a weak
  generator → near-zero BLEU/ROUGE; use it as an accuracy baseline).
- **Verify every `.bib` entry** against official sources before submission.

## Results tracker (filled vs pending)

| RQ | Where in paper | Artifact | Status |
|----|----------------|----------|--------|
| RQ1 | §Related Work | survey + taxonomy | 🟡 drafting (parallel) |
| RQ2 | §Results | Table: acc by alignment strategy | ⬜ pending runs |
| RQ3 | §Results | Table: acc + expl-quality; α-curve fig | ⬜ pending runs |
| RQ4 | §Results | ablation table | ⬜ pending runs |
| RQ5 | §Results / §Discussion | faithfulness table + qualitative | ⬜ pending runs |
| RQ6 | §Results | 2B-vs-7B efficiency table | ⬜ pending runs |
| RQ7 | §Results | 3×3 transfer matrix | ⬜ pending runs |
| — | §Methodology | full method (data, variants, loss, QLoRA, metrics) | 🟢 drafted |
| — | §Data | filtering yield (ScienceQA ~5k of 12.7k w/ rationale) | 🟢 noted |

Legend: 🟢 done · 🟡 in progress · ⬜ waiting on experiments.

*First real data point to produce:* the generative-vs-explanation-aware pair on
**one** dataset with **Qwen2-VL** → fills the first cells of the RQ2/RQ3 tables.

## Compiling

- **Overleaf:** New Project → Upload Project → add `main.tex` + `references.bib`
  (or link Overleaf to this GitHub repo). Recompile; it pulls the `elsarticle`
  class automatically.
- **Local:** `latexmk -pdf main.tex` (needs a TeX distribution + `elsarticle`).

`\note{}` (red) and `\todo{}` (blue) mark open items — comment out their two
`\newcommand` lines in `main.tex` to hide them for a clean PDF.
