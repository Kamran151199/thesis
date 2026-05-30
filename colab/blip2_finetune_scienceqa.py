# %% [markdown]
# # BLIP-2 Q-Former fine-tune on ScienceQA
#
# **Target compute:** Colab Pro+ A100 (40GB).
# **Runtime:** ~30-45 min for 2K examples × 1 epoch.
#
# **How to use this file in Colab:**
# 1. Push this file to your git repo, then in a Colab notebook:
#    `!git clone <your-repo>` and `%cd thesis`
# 2. Either run as `!python colab/blip2_finetune_scienceqa.py`, OR
# 3. Open VS Code → Sync to Jupyter → cells will appear in Colab interactively, OR
# 4. Use `jupytext --to ipynb colab/blip2_finetune_scienceqa.py` to convert to notebook
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


# %% Cell 6 — Prompt + target formatting (Pattern A: rationale-then-answer)
def format_prompt(item) -> str:
    """Input prompt for BLIP-2. Just the question + choices."""
    choices_str = ", ".join(f"({chr(65+i)}) {c}" for i, c in enumerate(item["choices"]))
    return f"Question: {item['question']} Choices: {choices_str} Answer:"


def format_target(item) -> str:
    """Target string: rationale first, then answer. Pattern A from our discussion."""
    answer_text = item["choices"][item["answer"]]
    return f" Solution: {item['solution'].strip()} The answer is: {answer_text}."


def collate(batch_items):
    """Build a single batch for model.forward."""
    images = [item["image"].convert("RGB") for item in batch_items]
    prompts = [format_prompt(item) for item in batch_items]
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

    # Labels: shift input_ids → -100 for (a) padding, (b) prompt-prefix positions (so loss is target-only).
    labels = enc.input_ids.clone()
    for i, prompt in enumerate(prompts):
        prompt_len = len(processor.tokenizer(prompt, add_special_tokens=False).input_ids)
        labels[i, :prompt_len] = -100  # ignore prompt in loss
    labels[labels == processor.tokenizer.pad_token_id] = -100  # ignore pad
    enc["labels"] = labels

    return enc


# %% Cell 7 — Baseline eval (BEFORE any fine-tuning)
#
# Generate answers on the eval set; loose exact-match against the answer text.
# This number is what we want to beat.

@torch.no_grad()
def evaluate(ds_to_eval, label="eval"):
    correct, total = 0, 0
    for item in ds_to_eval:
        image = item["image"].convert("RGB")
        prompt = format_prompt(item)
        enc = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)
        out_ids = model.generate(**enc, max_new_tokens=80)
        gen = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        # Loose match: did the model output the correct choice text anywhere after "answer is"?
        answer_text = item["choices"][item["answer"]].lower().strip()
        if answer_text and answer_text in gen.lower():
            correct += 1
        total += 1
    return correct / total

print("Measuring BASELINE accuracy (model BEFORE fine-tuning)...")
t0 = time.time()
baseline_acc = evaluate(eval_ds, "baseline")
print(f"baseline accuracy: {baseline_acc*100:.1f}%   ({time.time()-t0:.0f}s)")


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


# %% Cell 12 — Qualitative inspection: show 5 generations
print("\nQualitative generations (post fine-tune):\n")
for k in range(5):
    item = eval_ds[k]
    image = item["image"].convert("RGB")
    prompt = format_prompt(item)
    enc = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.bfloat16)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=80)
    gen = processor.tokenizer.decode(out_ids[0], skip_special_tokens=True)
    true_answer = item["choices"][item["answer"]]
    correct = true_answer.lower() in gen.lower()
    mark = "OK" if correct else "XX"
    print(f"[{mark}] Q: {item['question'][:80]}")
    print(f"     true:    {true_answer!r}")
    print(f"     gen:     {gen[-200:]!r}")
    print()


# %% [markdown]
# ## What you just did
#
# - Fine-tuned **BLIP-2 OPT-2.7B's Q-Former** (110M trainable params) on ScienceQA
# - Vision tower (1B) and LLM (2.7B) stayed FROZEN — only 3% of the model trained
# - Used Pattern A formatting: `"Solution: <rationale>. The answer is: <answer>."`
# - Loss masked the prompt prefix → supervision only on `Solution: ... answer: X`
# - Saved checkpoint to Drive for reuse / RQ3 ablations
#
# **What this maps to in the thesis:**
#
# - **RQ2**: BLIP-2 baseline number on ScienceQA → goes in Table 2.1.
# - **RQ3**: this notebook is one cell of the explanation-aware α-sweep (α=1.0 here:
#            we trained on rationale+answer together with equal weight). Add an α=0
#            run (answer only, no rationale) for the comparison.
# - **RQ5**: the trained model is ready for attention-alignment + masking-consistency
#            evaluation in a follow-up notebook.
# - **RQ6**: rerun with a larger backbone (e.g., `blip2-opt-6.7b` or `blip2-flan-t5-xxl`)
#            via QLoRA to fill in the 2B vs 7B comparison.
#
# **Next steps for tomorrow:**
# 1. Run this and confirm baseline → post lift (target: 30-40% → 55-70%).
# 2. Branch a copy for α=0 (answer-only) to start the RQ3 sweep.
# 3. Branch a copy for ChartQA to start RQ4 (cross-domain).
