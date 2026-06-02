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
| `rq2_qwen2vl_generative_scienceqa` | RQ2 / RQ3-base | Qwen2-VL-2B | generative | — | ScienceQA | 2000/200 | 0.450 → **0.765** (+0.315) | post **0.703** | post **0.570** |
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

## RQ3 head-to-head — generative vs explanation-aware (the comparison)

Both arms: Qwen2-VL-2B, ScienceQA, 2000/200, seed 42, identical hyperparameters.
The ONLY difference is the loss — uniform next-token CE (generative) vs the α=0.5
answer/explanation split (explanation-aware). Post-training:

| metric | generative (rq2) | explanation-aware α=0.5 (rq3) | Δ (rq3 − rq2) |
|--------|------------------|-------------------------------|---------------|
| mc_accuracy | **0.765** | 0.755 | −0.010 |
| rouge_l     | **0.703** | 0.653 | −0.050 |
| bleu        | **0.570** | 0.505 | −0.065 |

**Finding: at α=0.5 the explanation-aware objective is tied on accuracy and WORSE
on explanation quality — the opposite of the hypothesis.** Accuracy is a tie (1 pp,
inside the ±1 pp noise the two baselines exposed: 0.450 vs 0.460 for the same
untrained model; SE ≈ 3 pp at n=200). But ROUGE-L (−5 pp) and BLEU (−6.5 pp) fall,
and those gaps point the wrong way for an *explanation-aware* method.

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
| — *(frozen)* | 0.460 | 0.144 | 0.014 | zero-shot, pre-FT |
| — *(generative)* | 0.765 | 0.703 | 0.570 | uniform CE, α_eff ≈ 0.1 |
| 0.00 | 0.675 | 0.449 | 0.334 | explanation-only (answer span unsupervised) |
| **0.10** | **0.775** | **0.706** | **0.579** | best on all three; ≈ generative |
| 0.25 | 0.770 | 0.685 | 0.545 | |
| 0.50 | 0.740 | 0.653 | 0.505 | standalone run gave mc 0.755 → noise |
| 0.75 | 0.720 | 0.616 | 0.455 | |
| 1.00 | 0.680 | 0.000 | 0.000 | answer-only → no rationale emitted |

**Prediction check (against lines 72–78):**
- ✓ **ROUGE/BLEU fall monotonically as α rises** (α≥0.1: 0.706→0.685→0.653→0.616→0.000). Confirmed.
- ✓ **Generative ≈ α ≈ 0.1.** Best point (0.775/0.706/0.579) is within seed noise of the generative control (0.765/0.703/0.570) → α_eff ≈ 0.1 confirmed empirically; explicit answer-weighting can't beat what uniform CE already does.
- ✗ **Accuracy is NOT flat** (the one part of the pre-reg guess that was wrong). It is an inverted-U peaking at α=0.1 (0.775), dropping ~10 pp at both extremes (0.675 @ 0.0, 0.680 @ 1.0). Corrected framing: accuracy is maximised at the natural weight and degrades as α moves away in either direction.
- ⚠ **NEW — explanation supervision itself helps accuracy.** Answer-only (α=1) loses ~9.5 pp vs α=0.1 (0.680 vs 0.775) with rationale quality collapsing to 0 → the generated rationale behaves like a usable chain-of-thought at inference; starving it costs accuracy. The mirror case (explanation-only, α=0) is also weak on both axes (no answer anchor → rationale drifts off the concise gold, ROUGE 0.449).

**RQ3 headline for §4/§5:** *explicit answer-weighting cannot beat the weighting
uniform generative CE already applies (optimum = generative, α ≈ 0.1); but
supervising the explanation at all is worth ~10 pp of accuracy.* That is stronger
than the one-point "α=0.5 ties" null — a mechanistic curve with an interior optimum.

---

## Pending to complete each RQ

- **RQ3 α-sweep: DONE** — full six-point curve logged (`rq3_alpha_sweep.json`,
  RESULT table above). Optimum at α=0.1 ≈ generative; accuracy is an inverted-U;
  explanation supervision worth ~10 pp accuracy; token-count mechanism confirmed.
  Single seed → ideally ≥2–3 seeds at α ∈ {0.1, 0.5, 1.0} to put error bars on the
  ~10 pp gaps before the final claim is hardened.
- **RQ2** (alignment objective): generative arm logged (post 0.765); contrastive
  arm is scaffolded (`objective.contrastive`, needs `contrastive_features` per
  backbone) — in-scope cut unless time allows.
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
