# `src/evaluation/faithfulness/` — RQ5: grounded or hallucinated?

Accuracy says the answer is right; faithfulness asks whether the *reasoning* is
honestly tied to the image. Two complementary probes map to the proposal's two
automatic faithfulness metrics.

## 1. Evidence-masking consistency — `masking_consistency.py` (runs today)

Hide a region, watch the answer's likelihood:

```
drift(region) = score(answer | full) − score(answer | region masked)
```

```
full          mask top-left      mask bottom-right
┌──┬──┐       ┌──┬──┐            ┌──┬──┐
│░░│░░│       │▓▓│░░│            │░░│░░│
├──┼──┤   →   ├──┼──┤      →     ├──┼──┤
│░░│░░│       │░░│░░│            │░░│▓▓│
└──┴──┘       └──┴──┘            └──┴──┘
score -1.2    score -3.8         score -1.3
              drift +2.6 (relied on it)   drift +0.1 (ignored)
```

A faithful model's high-drift regions are the ones its explanation refers to. A
model whose answer barely moves when the relevant evidence is blanked is
hallucinating from the language prior. Model-agnostic — uses only masking +
`score_continuation`, so it runs on any backbone now.

## 2. Attention-alignment — `attention_alignment.py` (structured scaffold)

Compare the model's attention over image patches against where the explanation
actually refers. The aggregation/alignment math is implemented; the one
backbone-specific step — extracting per-patch attention (`output_attentions`) —
is left as `NotImplementedError` with guidance, because where cross-modal
attention lives differs across BLIP-2 (Q-Former cross-attn), Qwen2-VL and
PaliGemma. Start with BLIP-2's Q-Former cross-attentions (the cleanest).

## How they map to the thesis

| Proposal metric | Here |
|-----------------|------|
| consistency under evidence masking | `region_importance` (masking_consistency) |
| attention-alignment score | `alignment_score` + `extract_image_attention` |
| human 5-point Likert (20/domain) | manual — feed `Prediction.reasoning` to raters |
