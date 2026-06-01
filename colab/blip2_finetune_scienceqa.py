# %% [markdown]
# # BLIP-2 Q-Former fine-tune on ScienceQA
#
# **Target compute:** Colab Pro+ A100 (40GB).
# **Runtime:** ~30-45 min for 2K examples × 1 epoch.
#
# **What this does:**
# - Loads BLIP-2 OPT-2.7B, freezes vision tower + LLM, unfreezes Q-Former + projection
# - Loads 2K ScienceQA train examples + 200 eval examples
# - Measures BASELINE accuracy (no fine-tuning yet)
# - Fine-tunes for 1 epoch
# - Measures POST-FINETUNE accuracy
# - Saves the trained Q-Former weights to Drive
#
# **Expected result:** baseline ~30-40% → post-fine-tune ~55-70% on ScienceQA.


# %% Cell 1 — Install dependencies (Colab cell)
#!pip install -q -U transformers peft accelerate bitsandbytes datasets
from google.colab import drive  # type: ignore
drive.mount("/content/drive")
import os
os.environ["HF_HOME"] = "/content/drive/MyDrive/thesis/hf_cache"

# %% Cell 2 — Imports + reproducibility
import time, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from datasets import load_dataset

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


# %% Cell 3 — Load BLIP-2 in bf16
MODEL_NAME = "Salesforce/blip2-opt-2.7b"

print(f"Loading {MODEL_NAME} ...")
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
print(f"Loaded.  device: {next(model.parameters()).device}")


# %% Cell 4 — Freeze vision tower + LLM, train only Q-Former + projection
for name, p in model.named_parameters():
    if name.startswith("vision_model") or name.startswith("language_model"):
        p.requires_grad = False
    else:
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:>15,} ({100*trainable/total:.1f}%)")
print(f"Frozen:    {total - trainable:>15,} ({100*(total-trainable)/total:.1f}%)")
print(f"Total:     {total:>15,}")

# Sanity: list what's trainable
print("\nTrainable submodules (top-level):")
for name in {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}:
    print(f"  - {name}")


# %% Cell 5 — Load ScienceQA, filter to image-only examples with rationales
print("Loading ScienceQA...")
ds_full = load_dataset("derek-thomas/ScienceQA", split="train")

# Filter: must have image AND non-empty solution (the rationale we want to learn)
ds = ds_full.filter(
    lambda x: x["image"] is not None and (x.get("solution") or "").strip() != "",
    num_proc=4,
)
print(f"ScienceQA filtered: {len(ds):,} / {len(ds_full):,} examples have image + rationale")

train_ds = ds.shuffle(seed=SEED).select(range(2000))
eval_ds = ds.shuffle(seed=SEED + 1).select(range(200))
print(f"train: {len(train_ds)},  eval: {len(eval_ds)}")

# Inspect one example
ex = train_ds[0]
print(f"\nExample 0:")
print(f"  question:  {ex['question']!r}")
print(f"  choices:   {ex['choices']}")
print(f"  answer idx:{ex['answer']}  (= {ex['choices'][ex['answer']]!r})")
print(f"  solution:  {ex['solution'][:200]!r}{'...' if len(ex['solution'])>200 else ''}")
print(f"  image:     {ex['image'].size if ex['image'] else None}")

# %%
ds[0]["image"]  # PIL image object (not yet preprocessed)
ds[0]

# %% Cell 6 — Prompt + target formatting (explanation-aware: Reasoning → Answer)
#
# Template:  "Question: {q} Options: (A).. (B).. (C).."   ← prompt (masked in loss)
#            " Reasoning: {reasoning} Answer: {answer}."   ← target (supervised)
#
# Key property: "Answer:" now directly precedes the REAL answer (not the reasoning),
# so there's no cue-word ambiguity. The model learns to produce reasoning THEN answer —
# genuine chain-of-thought / explanation-aware supervision (RQ3 / RQ5).

def format_prompt(item) -> str:
    """Question + options. Ends WITHOUT a cue word — the target supplies 'Reasoning:'."""
    opts = ", ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(item["choices"]))
    return f"Question: {item['question']} Options: {opts}."

format_prompt(ds[0])  # sample prompt (question + options only)

# %%
def format_target(item) -> str:
    """Reasoning first, then the answer. The model learns to emit BOTH cue words."""
    answer_text = item["choices"][item["answer"]]
    return f" Reasoning: {item['solution'].strip()} Answer: {answer_text}."

# %%
format_target(ds[0])  # sample target (reasoning + answer the model learns to produce)


# %%
def collate(batch_items):
    """Build a single batch for model.forward (assumes right-padding)."""
    images = [item["image"].convert("RGB") for item in batch_items]
    full_texts = [format_prompt(item) + format_target(item) for item in batch_items]

    # Process image + the FULL text (prompt + target) — model learns to predict the whole thing,
    # but we'll mask the prompt-portion of the labels so loss is only on the target.
    enc = processor(
        images=images,
        text=full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device, torch.bfloat16)

    # Labels: -100 for (a) prompt-prefix positions, (b) padding — so loss is target-only.
    #
    # CRITICAL: prompt_len must include the 32 <image> placeholder tokens the processor
    # inserts. We get that by encoding the prompt THE SAME WAY (through the processor, with
    # the image) and reading its input_ids length. Counting text tokens alone undercounts by
    # the image-token count → the prompt tail leaks into the target (a real bug we hit).
    labels = enc.input_ids.clone()
    for i, item in enumerate(batch_items):
        prompt_len = processor(
            images=item["image"].convert("RGB"),
            text=format_prompt(item),
            return_tensors="pt",
        ).input_ids.shape[1]
        labels[i, :prompt_len] = -100  # ignore image + prompt in loss
    labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore pad
    enc["labels"] = labels

    return enc

# %%
collate(train_ds.select(range(1)))  # sample batch encoding (images + input_ids + labels)


# %% Cell 7 — Baseline eval: generate-then-score (chain-of-thought eval)
#
# Because the model is trained to REASON before answering, we must let it do that at
# eval time too — we can't bare-likelihood-score a choice right after the prompt (the
# model expects "Reasoning:" there). So the eval mirrors training:
#
#   1. GENERATE the model's reasoning from the prompt (its own chain of thought).
#   2. Build context = prompt + generated-reasoning + " Answer:".
#   3. LIKELIHOOD-SCORE each choice in that context; pick the argmax.
#
# Step 3 (scoring rather than string-parsing the generated answer) avoids fragile
# text matching — the model's reasoning conditions the choice, but the final decision
# is a robust likelihood comparison over the real candidate strings.

@torch.no_grad()
def score_choice_in_context(item, context_text: str, choice_text: str) -> float:
    """Length-normalized log-likelihood of `choice_text` continuing `context_text`.
    context_text = prompt + the model's own generated reasoning + ' Answer:'."""
    image = item["image"].convert("RGB")
    enc = processor(images=image, text=context_text + " " + choice_text,
                    return_tensors="pt").to(model.device, torch.bfloat16)
    labels = enc.input_ids.clone()
    ctx_len = processor(images=image, text=context_text,
                        return_tensors="pt").input_ids.shape[1]  # includes <image> tokens
    labels[:, :ctx_len] = -100                                    # score ONLY the choice
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return -model(**enc, labels=labels).loss.item()


@torch.no_grad()
def generate_reasoning(item, max_new_tokens: int = 160) -> str:
    """Let the model produce its chain of thought; return the reasoning span (text
    before its own 'Answer:'). Handles whether generate() echoes the prompt or not."""
    prompt = format_prompt(item)
    image = item["image"].convert("RGB")
    enc = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)
    out_ids = model.generate(**enc, max_new_tokens=max_new_tokens)
    gen = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    cont = gen[len(prompt):] if gen.startswith(prompt) else gen   # strip echoed prompt if present
    # reasoning = everything the model said before it tried to answer
    reasoning = cont.split("Answer:")[0].strip()
    # the model emits its own "Reasoning:" cue — strip it so the caller can re-add exactly one
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    return reasoning


@torch.no_grad()
def evaluate(ds_to_eval, label="eval") -> float:
    correct = 0
    for item in ds_to_eval:
        reasoning = generate_reasoning(item)                       # 1. model's own CoT
        context = f"{format_prompt(item)} Reasoning: {reasoning} Answer:"   # 2. condition on it
        scores = [score_choice_in_context(item, context, c) for c in item["choices"]]  # 3. score choices
        pred = int(np.argmax(scores))
        correct += int(pred == item["answer"])
    return correct / len(ds_to_eval)


print("Measuring BASELINE accuracy (generate-then-score CoT eval)...")
t0 = time.time()
baseline_acc = evaluate(eval_ds, "baseline")
n_choices = [len(eval_ds[i]["choices"]) for i in range(len(eval_ds))]
random_baseline = sum(1 / c for c in n_choices) / len(n_choices)
print(f"baseline accuracy: {baseline_acc*100:.1f}%   "
      f"(random ≈ {random_baseline*100:.1f}%)   ({time.time()-t0:.0f}s)")


# %% Cell 8 — Training loop
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5,
    weight_decay=0.01,
)
BATCH_SIZE = 4
NUM_STEPS = len(train_ds) // BATCH_SIZE  # ~500 steps for 2K / 4
LOG_EVERY = 25
losses = []

print(f"Starting fine-tune: {NUM_STEPS} steps, batch {BATCH_SIZE}, lr 1e-5")
t0 = time.time()
model.train()

for step in range(NUM_STEPS):
    batch = [train_ds[step * BATCH_SIZE + i] for i in range(BATCH_SIZE)]
    enc = collate(batch)
    outputs = model(**enc)
    loss = outputs.loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0
    )
    optimizer.step()

    losses.append(loss.item())
    if step % LOG_EVERY == 0:
        elapsed = time.time() - t0
        avg = sum(losses[-LOG_EVERY:]) / max(1, len(losses[-LOG_EVERY:]))
        print(f"step {step:4d}/{NUM_STEPS} | loss {loss.item():.4f} | avg-25 {avg:.4f} | {elapsed:.0f}s")

print(f"\nTraining done in {time.time()-t0:.0f}s")


# %% Cell 9 — Post-fine-tune eval
model.eval()
print("Measuring POST-FINE-TUNE accuracy...")
t0 = time.time()
post_acc = evaluate(eval_ds, "post")
print(f"post accuracy: {post_acc*100:.1f}%   ({time.time()-t0:.0f}s)")

print(f"\n{'='*60}")
print(f"BASELINE:       {baseline_acc*100:>5.1f}%")
print(f"POST FINE-TUNE: {post_acc*100:>5.1f}%")
print(f"LIFT:           {(post_acc-baseline_acc)*100:>+5.1f} percentage points")
print(f"{'='*60}")


# %% Cell 10 — Save trainable weights
import os
SAVE_DIR = "/content/drive/MyDrive/thesis/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = f"{SAVE_DIR}/blip2_qformer_scienceqa_2k_v1.pt"

trainable_state = {
    name: param.detach().cpu().clone()
    for name, param in model.named_parameters()
    if param.requires_grad
}
torch.save({
    "state_dict": trainable_state,
    "losses": losses,
    "baseline_acc": baseline_acc,
    "post_acc": post_acc,
    "config": {
        "model": MODEL_NAME,
        "num_train": len(train_ds),
        "num_eval": len(eval_ds),
        "batch_size": BATCH_SIZE,
        "lr": 1e-5,
        "epochs": 1,
        "seed": SEED,
    },
}, SAVE_PATH)
print(f"Saved to {SAVE_PATH}")


# %% Cell 11 — Loss curve
plt.figure(figsize=(10, 4))
plt.plot(losses, alpha=0.25, label="raw")
window = 25
smoothed = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(losses))]
plt.plot(smoothed, color="orange", linewidth=2, label=f"smoothed (w={window})")
plt.xlabel("step")
plt.ylabel("loss")
plt.title(f"BLIP-2 Q-Former fine-tune on ScienceQA   |   {baseline_acc*100:.1f}% → {post_acc*100:.1f}%")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# %% Cell 12 — Qualitative inspection: the model's reasoning + its prediction
#
# Per example: show the model's GENERATED reasoning (RQ5 will analyze this for
# faithfulness), then the choice it picks via the same generate-then-score eval.
print("\nQualitative inspection (post fine-tune):\n")
for k in range(5):
    item = eval_ds[k]

    # 1. The model's own chain of thought.
    reasoning = generate_reasoning(item)

    # 2. Its prediction, scored in the context of that reasoning (same as evaluate()).
    context = f"{format_prompt(item)} Reasoning: {reasoning} Answer:"
    scores = [score_choice_in_context(item, context, c) for c in item["choices"]]
    pred_idx = int(np.argmax(scores))
    pred_answer = item["choices"][pred_idx]
    true_answer = item["choices"][item["answer"]]
    mark = "OK" if pred_idx == item["answer"] else "XX"

    print(f"[{mark}] Q: {item['question'][:80]}")
    print(f"     choices:   {item['choices']}")
    print(f"     reasoning: {reasoning[:200]!r}")
    print(f"     predicted: {pred_answer!r}   true: {true_answer!r}")
    print()


# %% [markdown]
# ## What you just did
#
# - Fine-tuned **BLIP-2 OPT-2.7B's Q-Former** (110M trainable params) on ScienceQA
# - Vision tower (1B) and LLM (2.7B) stayed FROZEN — only 3% of the model trained
# - Explanation-aware template: `"Question:… Options:…  Reasoning: <r> Answer: <a>."`
# - Loss masked the image + prompt prefix → supervision only on `Reasoning: … Answer: X`
# - Eval = generate-then-score: the model produces its own reasoning, then we
#   likelihood-score each choice conditioned on that reasoning (chain-of-thought eval)
# - Saved checkpoint to Drive for reuse / RQ3 ablations
#
# **What this maps to in the thesis:**
#
# - **RQ2**: BLIP-2 generative baseline on ScienceQA → goes in Table 2.1.
# - **RQ3**: this is the explanation-aware arm (reasoning + answer). For the α-sweep,
#            add an answer-only variant (`format_target` → just the answer) = α=1.0,
#            and weight the reasoning vs answer loss for intermediate α.
# - **RQ5**: the generated `reasoning` strings are exactly what attention-alignment +
#            masking-consistency will evaluate for faithfulness.
# - **RQ6**: rerun with a larger backbone via QLoRA to fill in the 2B vs 7B comparison.
#
# **Next steps:**
# 1. Re-run; confirm the target decodes cleanly (no prompt leak) and post > baseline.
# 2. If still flat with OPT-2.7b → switch to Qwen2-VL-2B (the real workhorse).
# 3. Then branch: answer-only (α arm), ChartQA (RQ4 cross-domain).

# %%
