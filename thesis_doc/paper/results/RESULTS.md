# Results ledger

Single source of truth for **§4 Results**. Every logged run is also saved as a
JSON next to this file (verbatim from the Colab artifact), so the paper builds
from version-controlled numbers, not scattered notebook cells.

**Standing caveats** (carry into every quantitative claim in prose):
- **1 seed (42)** unless a row says otherwise → no variance bars yet; say "single run".
- Scale is modest (**2000 train / 200 eval**) — a focused hypothesis test, not a
  leaderboard entry. State it plainly.
- **BLEU / ROUGE-L = surface overlap** with the gold rationale → a proxy for
  explanation *quality/style*, **not** faithfulness. Faithfulness is RQ5
  (masking-drift), reported separately. Never infer faithfulness from BLEU/ROUGE.

---

## Runs logged

| config | RQ | backbone | objective | α | data | train/eval | mc_acc (base → post) | rouge_l (base → post) | bleu (base → post) |
|--------|----|----------|-----------|---|------|-----------|----------------------|-----------------------|--------------------|
| `rq3_qwen2vl_explanation_aware_scienceqa` | RQ3 | Qwen2-VL-2B | explanation_aware | 0.5 | ScienceQA | 2000/200 | 0.460 → **0.755** (+0.295) | 0.144 → 0.653 (+0.509) | 0.014 → 0.505 (+0.490) |

Notes on this run:
- `random_baseline = 0.354` (≈2.8 choices); baseline 0.46 is **above chance** ✓,
  and post 0.755 is not near 1.0 → the likelihood-scoring eval is not gamed (no
  repeat of the old string-match "100% baseline" bug).
- The **+0.295** is *fine-tuning lift* (trained vs untrained), **NOT** yet
  "explanation-aware vs generative" — see pending below.
- Qwen is a **real generator**: BLEU/ROUGE jump from ≈0 to 0.50/0.65 (BLIP-2 gave
  near-zero here). This is the explanation-quality signal the thesis leads with.
- Tension worth a sentence in §4/§5: it writes gold-like rationales (ROUGE 0.65)
  yet still misses **24.5%** of answers → explanation surface-match ≠ correct answer.

---

## Pending to complete each RQ

- **RQ3 needs its pair — `rq2_qwen2vl_generative_scienceqa`** (α=1, *same* data /
  seed / hyperparameters). The RQ3 claim is **explanation-aware POST vs generative
  POST**. Right now we have only the explanation-aware arm; without the generative
  POST we cannot say explanation-aware *helped*. This is the immediate next run.
- **RQ2** (alignment objective): the rq2 generative arm above doubles as the RQ2
  generative data point; contrastive arm is scaffolded (`objective.contrastive`,
  needs `contrastive_features` per backbone) — in-scope cut unless time allows.
- **RQ1** (alignment evolution): survey/§2 + checkpoint-trajectory analysis — no
  run yet.
- **RQ4** (cross-domain, ChartQA): `rq4_qwen2vl_explanation_aware_chartqa` — note
  the image-token ceiling caveat (raise `max_length`/lower `max_pixels` for charts).
- **RQ5** (faithfulness): masking-drift probe runs model-agnostically today;
  needs a batched sweep over the eval set, not just one example.
- **RQ6** (scale, 2B vs 7B): `rq6_qwen2vl_7b_explanation_aware_scienceqa`.
- **RQ7** (transfer): train-on-A, eval-on-B — no run yet.

When a pair/sweep lands, I (the paper) turn it into the §4 table + prose and tick
it off here.
