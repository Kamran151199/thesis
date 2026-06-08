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
| `rq2_qwen2vl_generative_scienceqa` | RQ2 / RQ3-base | Qwen2-VL-2B | rationale+answer CE | — | ScienceQA | 2000/200 | 0.400 → **0.780** (+0.380) | post **0.693** | post **0.559** |
| `rq3_alpha_050` | RQ3 | Qwen2-VL-2B | explanation_aware | 0.5 | ScienceQA | 2000/200 | 0.400 → **0.765** (+0.365) | post 0.643 | post 0.493 |

Notes on this run:
- `random_baseline = 0.354` (≈2.8 choices); the zero-shot reference is above chance,
  and none of the trained scores are near 1.0, so the likelihood-scoring evaluation
  is not showing the old string-match "100% baseline" bug.
- The fine-tuning lift is trained vs untrained. It is not by itself evidence that
  explanation-aware training beats the rationale+answer CE control.
- Qwen2-VL generates usable rationale text, but explanation surface overlap and
  answer correctness still have to be read separately.

---

## RQ3 head-to-head — generative vs explanation-aware (the comparison)

Both arms: Qwen2-VL-2B, ScienceQA, 2000/200, seed 42, identical hyperparameters.
The ONLY difference is the loss — uniform next-token CE (generative) vs the α=0.5
answer/explanation split (explanation-aware). Post-training:

| metric | generative (rq2) | explanation-aware α=0.5 (rq3) | Δ (rq3 − rq2) |
|--------|------------------|-------------------------------|---------------|
| mc_accuracy | **0.780** | 0.765 | −0.015 |
| rouge_l     | **0.693** | 0.643 | −0.050 |
| bleu        | **0.559** | 0.493 | −0.066 |

**Finding: at α=0.5 the explanation-aware objective is close on accuracy and lower
on explanation-overlap metrics than the rationale+answer CE control.** With n=200,
this should be read as a capped single-run observation, not a final statistical
claim. Still, the direction is not the simple "balanced α is better" story.

**Mechanism (verified against the code, not hand-waved):** `masked_token_ce`
returns the **mean** CE over a span, and the generative objective is the mean over
the WHOLE target. So generative weights the answer- and explanation-span mean
losses by their TOKEN COUNTS:

    L_generative = (N_ans·mean_ans + N_exp·mean_exp) / (N_ans + N_exp)
                 ≡ explanation_aware with α_eff = N_ans / (N_ans + N_exp)

ScienceQA answers are short (a word/letter), rationales long → α_eff ≈ 0.1. So
uniform CE *already* spends ~90 % of its gradient on the explanation tokens.
Setting **α = 0.5 DOWN-weights the explanation** (0.5 answer-share vs ~0.1) → the
rationale drifts from the gold → BLEU/ROUGE fall, while the extra answer weight
buys no accuracy. The "explanation-aware" reweighting, at α=0.5, *de-emphasised*
explanations. (NB: this also means `generative` ≠ explanation_aware α=1 — α=1 is
answer-ONLY, a different thing; the explanation_aware docstring conflates the two,
flagged to fix.)

**What it motivates — the α-sweep (now the core RQ3 experiment):** map quality↔
accuracy across α ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}. Prediction from the
mechanism: **BLEU/ROUGE fall monotonically as α rises** (less explanation weight),
with the generative baseline sitting near α ≈ 0.1. If confirmed, RQ3 becomes a
clean mechanistic curve instead of a one-point null — "balanced reweighting is
counterproductive; uniform CE is a strong baseline because length already weights
the explanation." Cheap: same data, just `--set objective.alpha=X`.

---

## RQ3 α-sweep — RESULT (the curve, confirmed + refined)

Six runs, Qwen2-VL-2B / ScienceQA / 2000–200 / seed 42; only `objective.alpha`
changes. Canonical record: `rq3_alpha_sweep.json`.

| α | mc_acc | rouge_l | bleu | note |
|------|--------|---------|------|------|
| — *(frozen)* | 0.400 | — | — | zero-shot, pre-FT |
| — *(rationale+answer CE)* | 0.780 | 0.693 | 0.559 | uniform CE control |
| 0.00 | 0.725 | 0.417 | 0.299 | explanation-only (answer span unsupervised) |
| **0.10** | **0.780** | **0.695** | **0.566** | best fixed α; tied with rationale+answer CE on accuracy |
| 0.25 | 0.775 | 0.686 | 0.548 | |
| 0.50 | 0.765 | 0.643 | 0.493 | fixed balanced setting used in scale comparison |
| 0.75 | 0.690 | 0.604 | 0.443 | |
| 1.00 | 0.695 | — | — | answer-span-only inside rationale-shaped target |

**Prediction check (against lines 72–78):**
- ✓ **ROUGE/BLEU fall as α rises** (α≥0.1: 0.695→0.686→0.643→0.604). Confirmed.
- ✓ **Rationale+answer CE ≈ α = 0.1.** The best fixed point (0.780/0.695/0.566) matches the rationale+answer CE control in accuracy (0.780/0.693/0.559).
- ✗ **Accuracy is not flat.** It peaks at α=0.1 (0.780), then drops toward the high-answer-weight end (0.690 @ 0.75, 0.695 @ 1.0).
- ⚠ **Important distinction.** The α=1 row is not the true answer-only control. It is answer-span-only inside a rationale-shaped target. The true answer-only ScienceQA control uses an answer-only prompt family and reaches 0.800.

**RQ3 headline for §4/§5:** *fixed α=0.1 is the best fixed explanation-aware point
in the sweep and matches rationale+answer CE in accuracy, but the true answer-only
prompt family remains the strongest ScienceQA control.*

---

## Completed coverage and remaining caveats

- **RQ3 α-sweep: DONE** — full six-point curve logged (`rq3_alpha_sweep.json`,
  RESULT table above). Optimum at α=0.1 ≈ rationale+answer CE; the true answer-only
  prompt family is reported separately and should not be confused with α=1.
  Single seed → ideally ≥2–3 seeds at α ∈ {0.1, 0.5, 1.0} to put error bars on the
  ~10 pp gaps before the final claim is hardened.
- **RQ2 objective comparison: DONE** — BLIP-2 generative and BLIP-2
  contrastive-enhanced ScienceQA runs are logged in `Tables/generated/all_results.csv`.
  The observed gain is small (0.480 vs 0.475) and the retrieval diagnostic remains
  close to frozen/random controls.
- **RQ1 alignment evolution: DONE in the literature review** — the paper answers
  this through the related-work synthesis rather than a new experiment.
- **RQ4 domain behaviour: DONE with scope boundary** — ScienceQA/A-OKVQA use
  rationale-bearing supervision where available; ChartQA, DocVQA, and VQAv2 are
  answer-only fallback/domain rows, not explanation-aware evidence.
- **RQ5 evidence sensitivity: DONE** — masking drift is logged for completed runs
  with 30 examples per run and approximate intervals in the manuscript table.
  It is reported as evidence sensitivity, not proof of faithful rationales.
- **RQ6 scale: DONE** — Qwen2-VL-2B and Qwen2-VL-7B are compared under the same
  fixed ScienceQA explanation-aware setting.
- **RQ7 transfer: DONE** — cross-domain transfer matrix is generated and reported.

Remaining caveats are the same ones used in the manuscript: one seed, capped
evaluation, DocVQA internal split, VQAv2 subset, and no gold rationales for
ChartQA/DocVQA/VQAv2.
