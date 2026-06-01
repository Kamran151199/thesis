# %% [markdown]
# # 01 · Run an experiment on the framework  (RQ3 · explanation-aware · BLIP-2 · ScienceQA)
#
# A full research cycle, done as **thin calls into `src/`** — no training loop is
# rewritten here, we just *compose* the framework's pieces and **interrogate**
# them. This is the template every experiment notebook copies.
#
# What we do:
#   1. load a config (one experiment, as data)
#   2. build the four axes (model · data · objective · metrics)
#   3. **verify the masking** — decode a label row (the `# VERIFY` step)
#   4. baseline accuracy (before training)
#   5. fine-tune
#   6. measure the lift
#   7. read the model's reasoning (RQ5 raw material)
#   8. faithfulness probe — mask image regions, watch the answer move
#
# **Needs a GPU** (BLIP-2 OPT-2.7B). On Colab: run `colab/setup_colab.py` first.
# For a fast demo we shrink the run to a few dozen examples / 10 steps.

# %% Cell 1 — imports + reproducibility
from src.config import load_config
from src.data import build_collator, build_dataset, build_template
from src.evaluation import Evaluator, build_metrics
from src.evaluation.faithfulness import region_importance
from src.evaluation.scoring import generate_continuation, split_reasoning_answer
from src.models import build_model
from src.objectives import build_objective
from src.training import ConsoleCallback, Trainer
from src.utils import describe_device, set_seed

print("device:", describe_device())  # cuda:0 NVIDIA A100 … on Colab; cpu locally
set_seed(42)


# %% [markdown]
# ## 1. Load a config — one experiment is just data
# We take the RQ3 explanation-aware BLIP-2 config and **shrink it** so the demo
# runs in a couple of minutes. In a real run you'd drop these overrides (or use
# `python -m src.run --config <this file>`).

# %% Cell 2 — load + shrink for a fast demo
cfg = load_config("configs/experiments/rq3_blip2_explanation_aware_scienceqa.yaml")

cfg.data.max_train = 64        # tiny slice for a demo (real run: 2000)
cfg.data.max_eval = 16         # (real run: 200)
cfg.train.batch_size = 2
cfg.train.grad_accum_steps = 1
cfg.train.max_steps = 10       # just enough to see loss move
cfg.train.log_every = 2
cfg.wandb.enabled = False      # no tracking for a scratch run

print(f"{cfg.name}  ({cfg.rq})")
print(f"  backbone : {cfg.model.name}  ({cfg.model.pretrained})")
print(f"  objective: {cfg.objective.name}  alpha={cfg.objective.alpha}")
print(f"  data     : {cfg.data.name}  variant={cfg.data.prompt_variant}")


# %% [markdown]
# ## 2. Build the four axes
# Each `build_*` reads its slice of the config and returns a ready object. The
# collator gets `tag_spans` from the objective — explanation-aware needs the
# answer/explanation token tags, so it's switched on automatically.

# %% Cell 3 — model · data · objective · collator · metrics
wrapper = build_model(cfg.model)                       # loads BLIP-2 (+ freeze/LoRA)
objective = build_objective(cfg.objective)             # explanation-aware loss
template = build_template(cfg.data.prompt_variant)     # "Reasoning: … Answer: …"

train_ds = build_dataset(cfg.data, split=cfg.data.split_train)
eval_ds = build_dataset(cfg.data, split=cfg.data.split_eval)
collator = build_collator(cfg.data, wrapper, tag_spans=objective.requires_span_ids)
metrics = build_metrics(cfg.eval.metrics)              # [mc_accuracy, rouge_l, bleu]

print(f"train={len(train_ds)}  eval={len(eval_ds)}  trainable params={wrapper.num_trainable():,}")


# %% [markdown]
# ## 3. Sanity-check the data — DECODE the labels (`# VERIFY`)
# The single most important check before training: confirm the loss is computed
# **only on the target** (` Reasoning: … Answer: X.`) and that no image/prompt
# tokens leaked in. We decode the supervised positions of one row.

# %% Cell 4 — peek at a sample + verify masking
ex = train_ds[0]
print("QUESTION :", ex.question)
print("CHOICES  :", ex.choices)
print("ANSWER   :", ex.answer, f"(idx {ex.answer_index})")
print("RATIONALE:", (ex.explanation or "")[:160], "…\n")

batch = collator([train_ds[0], train_ds[1]])
labels0 = batch["labels"][0]
supervised = labels0[labels0 != -100]                  # drop image/prompt/pad (-100)
decoded = wrapper.processor.tokenizer.decode(supervised, skip_special_tokens=True)
print("SUPERVISED TARGET (what the loss sees):")
print(" ", repr(decoded))
assert "Question:" not in decoded, "prompt leaked into the target — masking bug!"
print("\n✓ clean: only the reasoning+answer span is supervised")


# %% [markdown]
# ## 4. Baseline — measure BEFORE any training
# Multiple-choice eval = **generate-then-score** (the model writes its reasoning,
# then each choice is likelihood-scored in that context; argmax wins).

# %% Cell 5 — baseline metrics
evaluator = Evaluator(wrapper, eval_ds, template, cfg.eval, metrics)
baseline = evaluator.evaluate()
print("BASELINE:", {k: round(v, 4) for k, v in baseline.items()})


# %% [markdown]
# ## 5. Fine-tune
# The `Trainer` is backbone/objective-agnostic — it just asks the objective for a
# loss and fires callbacks. We pass `evaluator_fn=None` here and eval manually
# afterwards so the before/after is explicit.

# %% Cell 6 — train
trainer = Trainer(
    wrapper, objective, train_ds, collator, cfg,
    callbacks=[ConsoleCallback()],
    evaluator_fn=None,
)
summary = trainer.train()
print("train summary:", summary)


# %% [markdown]
# ## 6. The lift — after vs before
# The whole RQ3 question in two numbers: did explanation-aware training move
# accuracy (and explanation quality) up? (10 steps on 64 examples won't show much
# — this is the *mechanism*, not the result.)

# %% Cell 7 — post-training metrics + lift
post = evaluator.evaluate()
print(f"{'metric':16s} {'baseline':>9s} {'post':>9s} {'Δ':>8s}")
for k in sorted(set(baseline) & set(post)):
    if k == "random_baseline":
        continue
    print(f"{k:16s} {baseline[k]:9.4f} {post[k]:9.4f} {post[k] - baseline[k]:+8.4f}")


# %% [markdown]
# ## 7. Read the model's reasoning — the RQ5 raw material
# The generated rationales are exactly what faithfulness analysis inspects. Read a
# few: are they grounded in the image, or fluent guesses?

# %% Cell 8 — qualitative chain-of-thought
for k in range(3):
    item = eval_ds[k]
    cont = generate_continuation(wrapper, item, template, max_new_tokens=cfg.eval.max_new_tokens)
    reasoning, answer = split_reasoning_answer(cont)
    print(f"Q: {item.question[:90]}")
    print(f"   reasoning: {reasoning[:200]!r}")
    print(f"   model→ {answer!r}   |   true→ {item.answer!r}\n")


# %% [markdown]
# ## 8. Faithfulness probe (RQ5) — does masking the evidence change the answer?
# Hide each image region, measure how much the gold answer's likelihood drops.
# High drift on a region ⇒ the model *relied* on it (grounded). Near-zero
# everywhere ⇒ it's answering from the language prior (hallucinating). This is the
# model-agnostic faithfulness signal — runs on any backbone today.

# %% Cell 9 — evidence-masking importance for one example
item = eval_ds[0]
result = region_importance(wrapper, item, template, grid=(2, 2))
print(f"Q: {item.question[:90]}")
print(f"region drifts (2×2, row-major): {[round(d, 3) for d in result.drifts]}")
print(f"max reliance={result.max_drift:.3f}   mean={result.mean_drift:.3f}")
print("→ higher drift = the model leaned on that region for its answer")


# %% [markdown]
# ## What you just did
#
# A complete RQ3 experiment **without writing any training/eval code** — every
# step was a call into `src`. Maps to the thesis like this:
#
# - **Cell 4** (masking verify) → the hygiene check before any real run
# - **Cells 5–7** (baseline → train → lift) → RQ3's "does explanation help?" table
# - **Cell 8** (reasoning) → RQ3 explanation-quality + RQ5 faithfulness inputs
# - **Cell 9** (masking drift) → RQ5 faithfulness metric (Table 5.1)
#
# **For a real run**, drop the Cell-2 shrink overrides and either run all cells, or
# just use the one-liner the framework provides:
#
# ```python
# from src.experiment import ExperimentRunner
# ExperimentRunner(load_config("configs/experiments/rq3_blip2_explanation_aware_scienceqa.yaml")).run()
# # → baseline-eval → train → final-eval → writes outputs/<name>/{results.json, checkpoint}
# ```
#
# Swap one string in the config (`model.name`, `data.name`, `objective.name`) to
# get a different experiment — that's the whole point of the framework.
