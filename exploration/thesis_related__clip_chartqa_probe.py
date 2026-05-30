# %% [markdown]
# # thesis_related__clip_chartqa_probe
#
# **Two goals braided together in this file:**
#
# 1. **Learning goal** — internalize how HuggingFace pretrained models work end-to-end.
#    Once you finish this file you should be able to load BLIP-2 or Qwen2-VL the same way
#    with zero friction. (Files 05/06/07 will do exactly that.)
#
# 2. **Thesis goal** — produce empirical motivation for RQ2: pretrained CLIP (trained on
#    natural images) FAILS on charts. The side-by-side similarity matrix is the picture
#    that justifies why we need generative alignment (BLIP-2 / Qwen2-VL) beyond contrastive.
#
# **Roadmap (each part marked with which goal it serves):**
# ```
#   Part 1 — The universal HuggingFace contract           [learning]
#   Part 2 — Load CLIP: what `from_pretrained` does       [learning]
#   Part 3 — Peek inside the loaded model                 [learning]
#   Part 4 — Peek inside the Processor                    [learning]
#   Part 5 — Run the Processor & inspect the output       [learning]
#   Part 6 — Forward pass & inspect the model output      [learning]
#   Part 7 — Sanity check on a natural image (cat)        [learning + sanity]
#   Part 8 — Load ChartQA samples (cached locally)        [thesis]
#   Part 9 — The probe: side-by-side similarity matrix    [thesis figure]
#  Part 10 — Quantify the gap (top-1 retrieval, etc.)     [thesis numbers]
#  Part 11 — Failure-mode gallery                         [thesis qualitative]
#  Part 12 — Per-patch features (what BLIP-2 consumes)    [bridge to file 05]
#  Part 13 — Universal pattern recap                      [learning]
# ```

# %% [markdown]
# ### Imports

# %%
import pickle
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset
import torchvision


# %% [markdown]
# ## Part 1 — The universal HuggingFace contract
#
# Every HuggingFace VLM (CLIP, BLIP-2, Qwen2-VL, PaliGemma...) follows the **same
# 3-step recipe**. Internalize this shape and the rest is just renaming classes.
#
# ```
# 1.  model     = SomeModel.from_pretrained("repo/model-name")
# 2.  processor = SomeProcessor.from_pretrained("repo/model-name")
# 3.  output    = model(**processor(text=..., images=..., return_tensors="pt"))
# ```
#
# **Mental model:**
# - **Model**     = the trained neural network weights (~600MB–15GB on disk for VLMs).
# - **Processor** = the *preprocessing recipe* the model was trained with: how to
#                   resize/normalize images, how to tokenize text. Mirrors training-time
#                   preprocessing EXACTLY — that's the whole point of bundling it.
# - **from_pretrained(name)** = downloads + caches both, returns a ready-to-use object.
#
# **Cache location:** `~/.cache/huggingface/hub/`. First run downloads ~600MB for
# CLIP-base; every run after is instant. To inspect / clean: `huggingface-cli scan-cache`.


# %% [markdown]
# ## Part 2 — Load pretrained CLIP
#
# We're using `openai/clip-vit-base-patch32` — the original CLIP paper's checkpoint:
#
# - **151M params total** (vision tower ≈ 88M, text tower ≈ 63M, projections ≈ small)
# - **trained on ~400M image-text pairs** scraped from the web (natural images + alt-text)
# - **vision tower** = ViT-B/32 — a Vision Transformer with 32×32 patches (same architecture
#   you built from scratch in `03_vision_transformer.py`, but bigger)
# - **text tower** = 12-layer causal transformer (same family as your nanoGPT)
# - **shared embedding dim** = 512 (the "shared room" where image vectors meet text vectors)
#
# Vision config:     12 blocks, 12 heads, hidden=768, patch=32 → 49 patches @ 224×224 input.
# Text config:       12 blocks, 8 heads,  hidden=512, max length=77 tokens.
# Both project to:   512-dim shared space (`out.image_embeds` and `out.text_embeds`).

# %%
MODEL_NAME = "openai/clip-vit-base-patch32"

# Step 1 of the universal contract: download + load the weights.
# First run: ~600MB download. Subsequent runs: instant.
model = CLIPModel.from_pretrained(MODEL_NAME)

# Step 2 of the universal contract: download + load the preprocessing recipe.
# Tiny download — vocab file + image transform config.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# `eval()` = inference mode (disables dropout, freezes BN stats, etc.). ALWAYS call
# this when you're not training. Skipping it can change the outputs subtly.
model.eval()

print(f"Loaded: {MODEL_NAME}")
print(f"  total params:        {sum(p.numel() for p in model.parameters()):>12,}")
print(f"  vision hidden dim:   {model.config.vision_config.hidden_size:>12}")
print(f"  text hidden dim:     {model.config.text_config.hidden_size:>12}")
print(f"  shared proj dim:     {model.config.projection_dim:>12}")
print(f"  learned temperature: {model.logit_scale.exp().item():>12.2f}")
print(f"  image model config:         {model.config.vision_config.to_dict()}")
print(f"  text model config:          {model.config.text_config.to_dict()}")

# %% [markdown]
# ## Part 3 — Peek inside the loaded model
#
# `model` is just a `torch.nn.Module`. You can print it, grab its sub-modules,
# inspect any tensor inside. **This is the secret weapon for debugging VLMs.**
# When inference looks wrong, you walk the tree and check intermediate outputs.

# %%
# Top-level submodules — each is its own nn.Module.
print("Top-level submodules:")
for name, mod in model.named_children():
    n_params = sum(p.numel() for p in mod.parameters())
    print(f"  model.{name:22s}  →  {type(mod).__name__:32s}  ({n_params:>11,} params)")

# What you should see:
#   text_model           → CLIPTextTransformer        (the text tower)
#   vision_model         → CLIPVisionTransformer      (the vision tower)
#   visual_projection    → Linear                     (vision_dim → shared_dim)
#   text_projection      → Linear                     (text_dim   → shared_dim)
#
# That matches exactly the architecture you built in your mini-CLIP! Two encoders
# + two projection matrices. The shared `logit_scale` is the learned temperature.


# %%
# Drill deeper into the vision tower — it's the ViT you built from scratch.
print("\nVision tower internals:")
print(f"  patch embedding:    {type(model.vision_model.embeddings).__name__}")
print(f"  num transformer blocks: {len(model.vision_model.encoder.layers)}")
print(f"  block 0 type:       {type(model.vision_model.encoder.layers[0]).__name__}")
print(f"  final layer-norm:   {type(model.vision_model.post_layernorm).__name__}")

# Inspect the patch embedding layer (the thing that turns 224x224 images into 49+1 tokens)
patch_emb = model.vision_model.embeddings
print(f"\n  patch_size:         {patch_emb.patch_size}")
print(f"  num_patches:        {patch_emb.num_patches}")
print(f"  cls token shape:    {tuple(patch_emb.class_embedding.shape)}")
print(f"  pos emb shape:      {tuple(patch_emb.position_embedding.weight.shape)}")


# %% [markdown]
# ## Part 4 — Peek inside the Processor
#
# The Processor bundles TWO things — both are full pre-processing pipelines:
#
# ```
# CLIPProcessor
# ├── image_processor   (resize → center-crop → /255 → normalize → tensor)
# └── tokenizer         (BPE: text → input_ids + attention_mask)
# ```
#
# This is the universal HF Processor shape — BLIP-2 etc. have the same two sub-objects.

# %%
print("Image processor (the image side):")
print(f"  type:           {type(processor.image_processor).__name__}")
print(f"  do_resize:      {processor.image_processor.do_resize}")
print(f"  size:           {processor.image_processor.size}")
print(f"  do_center_crop: {processor.image_processor.do_center_crop}")
print(f"  crop_size:      {processor.image_processor.crop_size}")
print(f"  do_normalize:   {processor.image_processor.do_normalize}")
print(f"  image_mean:     {processor.image_processor.image_mean}")
print(f"  image_std:      {processor.image_processor.image_std}")
print(f"  do_rescale:     {processor.image_processor.do_rescale}  (i.e. /255)")

# Note: these are CLIP's OWN training statistics, NOT ImageNet's. Each VLM has its
# own. Don't mix them up — that's a classic "why is my fine-tune broken" bug.

print(f"\nTokenizer (the text side):")
print(f"  type:           {type(processor.tokenizer).__name__}")
print(f"  vocab size:     {processor.tokenizer.vocab_size}")
print(f"  bos token:      {processor.tokenizer.bos_token!r}  (id={processor.tokenizer.bos_token_id})")
print(f"  eos token:      {processor.tokenizer.eos_token!r}  (id={processor.tokenizer.eos_token_id})")
print(f"  pad token:      {processor.tokenizer.pad_token!r}")
print(f"  model_max_len:  {processor.tokenizer.model_max_length}")


# %% [markdown]
# ## Part 5 — Run the Processor & inspect what it returns
#
# Calling `processor(text=..., images=...)` returns a **dict-like** object
# (`BatchEncoding`) with three tensor keys:
#
# ```
# pixel_values   (B, 3, H, W)   ← processed images
# input_ids      (B, T)         ← tokenized text
# attention_mask (B, T)         ← 1 for real tokens, 0 for padding
# ```
#
# B = batch size, H/W = 224 (CLIP's fixed input size), T = max sequence length in batch.

# %%
# Grab one PIL image to play with.
_cifar = torchvision.datasets.CIFAR10(root="exploration/data", train=False, download=True)
CIFAR_CLASSES = _cifar.classes
CIFAR_CAPTIONS = [f"a photo of a {c}" for c in CIFAR_CLASSES]

cat_idx = next(i for i, t in enumerate(_cifar.targets) if t == 3)  # 3 = cat
cat_img = _cifar[cat_idx][0]
print(f"raw cat image: {type(cat_img).__name__}, size={cat_img.size}, mode={cat_img.mode}")

# Run the processor — this is the magic call.
inputs = processor(
    text=CIFAR_CAPTIONS,           # list of 10 strings
    images=cat_img,                # one PIL image
    return_tensors="pt",           # "pt" = PyTorch; could also be "np", "tf", "jax"
    padding=True,                  # right-pad shorter sequences to the longest in batch
)

print(f"\ntype of inputs:      {type(inputs).__name__}")
print(f"keys in inputs:      {list(inputs.keys())}")
print(f"\npixel_values.shape:  {tuple(inputs.pixel_values.shape)}   ← 1 image, 3 channels, 224x224")
print(f"input_ids.shape:     {tuple(inputs.input_ids.shape)}     ← 10 texts, padded to longest")
print(f"attention_mask.shape:{tuple(inputs.attention_mask.shape)}")


# %% [markdown]
# ### 5.1 — Inspect the image side: pixel_values
#
# CLIP's image processor took the 32x32 PIL cat photo and:
#   1. Resized so the shorter side = 224
#   2. Center-cropped to 224x224
#   3. Converted to a float tensor in [0, 1]   (divide by 255)
#   4. Normalized per-channel: (x - mean) / std using CLIP's training stats
#
# Result: a (1, 3, 224, 224) tensor with each channel ~zero-mean, unit-std-ish.

# %%
px = inputs.pixel_values[0]   # drop batch dim → (3, 224, 224)
print(f"pixel_values dtype:    {px.dtype}")
print(f"pixel_values shape:    {tuple(px.shape)}")
print(f"per-channel means:     {px.mean(dim=(1, 2)).tolist()}   ← should be near 0 (normalized)")
print(f"per-channel stds:      {px.std(dim=(1, 2)).tolist()}    ← should be near 1 (normalized)")
print(f"min / max:             {px.min().item():.3f}  /  {px.max().item():.3f}")


# %%
# Visualize: raw PIL on the left, processed tensor on the right (un-normalize for display).
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].imshow(cat_img)
axes[0].set_title(f"Raw PIL (size={cat_img.size})")
axes[0].axis("off")

# Un-normalize for display: multiply back by std, add mean, clamp to [0,1]
mean = torch.tensor(processor.image_processor.image_mean).view(3, 1, 1)
std = torch.tensor(processor.image_processor.image_std).view(3, 1, 1)
display = (px * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
axes[1].imshow(display)
axes[1].set_title(f"After processor (shape={tuple(px.shape)}, 224x224)")
axes[1].axis("off")

plt.suptitle("What the image_processor did to your image")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### 5.2 — Inspect the text side: input_ids
#
# Each row of `input_ids` is one caption tokenized with BPE.
# `attention_mask` tells the model which positions are real vs padding.
# Let's decode the first row back to see the BPE pieces.

# %%
print(f"Caption 0 raw text:   {CIFAR_CAPTIONS[0]!r}")
print(f"input_ids[0]:         {inputs.input_ids[0].tolist()}")
print(f"attention_mask[0]:    {inputs.attention_mask[0].tolist()}")
print()

# Decode each token id individually so you can see the BPE split.
print("Token-by-token decode of caption 0:")
for tok_id in inputs.input_ids[0].tolist():
    tok_str = processor.tokenizer.decode([tok_id])
    note = "  <PAD>" if tok_id == 0 else ""
    print(f"  id={tok_id:>6d}  →  {tok_str!r}{note}")

# Decode the whole sentence as a sanity check.
print(f"\nFull decode (skip special): {processor.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)!r}")


# %% [markdown]
# ## Part 6 — Forward pass & inspect the model output
#
# Call the model with `**inputs`. Returns a `CLIPOutput` dataclass with these fields:
#
# ```
# out.image_embeds       (B_img, 512)         ← image vectors in shared space, L2-normalized internally
# out.text_embeds        (B_txt, 512)         ← text  vectors in shared space, L2-normalized internally
# out.logits_per_image   (B_img, B_txt)       ← (image · text) * exp(logit_scale)
# out.logits_per_text    (B_txt, B_img)       ← transpose of the above
# out.vision_model_output (rich tuple)        ← intermediate vision states (if you ask for them)
# out.text_model_output  (rich tuple)         ← intermediate text states
# ```
#
# `logits_per_image.softmax(dim=-1)` → probability for each text given the image.
# That's how zero-shot classification works.

# %%
with torch.no_grad():
    out = model(**inputs)

print(f"type of out:                  {type(out).__name__}")
print(f"available fields:             {[k for k in out.keys()]}")
print()
print(f"out.image_embeds shape:       {tuple(out.image_embeds.shape)}      ← 1 image, 512-dim shared space")
print(f"out.text_embeds shape:        {tuple(out.text_embeds.shape)}     ← 10 texts, 512-dim shared space")
print(f"out.logits_per_image shape:   {tuple(out.logits_per_image.shape)}      ← image vs each text (already temp-scaled)")
print()
print(f"L2 norm of image_embeds[0]:   {out.image_embeds[0].norm().item():.6f}  ← already unit-normalized inside HF")
print(f"L2 norm of text_embeds[0]:    {out.text_embeds[0].norm().item():.6f}  ← same")
print()
print(f"logits_per_image (raw):       {out.logits_per_image[0].tolist()}")
print(f"softmax → probabilities:      {out.logits_per_image.softmax(dim=-1)[0].tolist()}")


# %% [markdown]
# ## Part 7 — Sanity check: CLIP on a natural image
#
# Feed our cat photo + all 10 CIFAR class captions. CLIP should pick "cat" with
# high probability — this confirms the model + processor are wired correctly.

# %%
probs = out.logits_per_image.softmax(dim=-1)[0]

print("CLIP zero-shot on the cat photo:")
for c, p in zip(CIFAR_CLASSES, probs.tolist()):
    star = "  <-- correct" if c == "cat" else ""
    print(f"  {c:12s}: {p:.3f}{star}")


# %%
# Sanity-check viz: ranked probability bar.
fig, ax = plt.subplots(figsize=(9, 4))
order = np.argsort(probs.tolist())[::-1]
labels = [CIFAR_CLASSES[i] for i in order]
heights = [probs[i].item() for i in order]
colors = ["#2ca02c" if c == "cat" else "#a0a0a0" for c in labels]
ax.bar(labels, heights, color=colors)
ax.set_ylabel("CLIP probability")
ax.set_title("Sanity check: pretrained CLIP correctly identifies 'cat'")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Part 8 — Load ChartQA samples
#
# ChartQA = question-answering over real-world bar/line/pie charts.
# We use the validation/test split, sample N items, cache locally so re-runs are instant.
#
# Dataset path note: the exact HF mirror + column names vary. We try a few common
# paths in order. If they all fail (gated dataset etc.), swap in another loader.

# %%
N_SAMPLES = 10
CACHE_PATH = Path("exploration/data/chartqa_probe_sample.pkl")

if CACHE_PATH.exists():
    with open(CACHE_PATH, "rb") as f:
        chart_samples = pickle.load(f)
    print(f"loaded cached: {len(chart_samples)} ChartQA items from {CACHE_PATH}")
else:
    candidates = [
        ("HuggingFaceM4/ChartQA", "val"),
        ("ahmed-masry/ChartQA", "test"),
        ("lmms-lab/ChartQA", "test"),
    ]
    ds = None
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            print(f"loaded HF dataset: {name} / {split}")
            break
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {e}")
    if ds is None:
        raise RuntimeError("ChartQA not loadable from any of the candidate mirrors.")

    print(f"columns: {ds.column_names}")
    img_col = next((c for c in ["image", "img", "image_path"] if c in ds.column_names), None)
    txt_col = next((c for c in ["query", "question", "question_text"] if c in ds.column_names), None)
    if img_col is None or txt_col is None:
        raise RuntimeError(f"can't find image/text columns in {ds.column_names}")
    print(f"using image col='{img_col}', text col='{txt_col}'")

    torch.manual_seed(0)
    idx = torch.randperm(len(ds))[:N_SAMPLES].tolist()
    chart_samples = []
    for i in idx:
        item = ds[int(i)]
        img = item[img_col]
        if isinstance(img, str):
            img = Image.open(img)
        chart_samples.append({"image": img.convert("RGB"), "question": item[txt_col]})

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(chart_samples, f)
    print(f"cached {len(chart_samples)} items -> {CACHE_PATH}")

print(f"\nfirst question:   {chart_samples[0]['question']!r}")
print(f"first image size: {chart_samples[0]['image'].size}")


# %% [markdown]
# ## Part 9 — The probe: side-by-side similarity matrices
#
# Build two N×N similarity matrices using the SAME pretrained CLIP:
#   - LEFT:  N CIFAR images   × N CIFAR captions    (CLIP's home turf)
#   - RIGHT: N chart images   × N chart questions   (out-of-distribution)
#
# Expected: bright diagonal LEFT (CLIP works), washed-out RIGHT (CLIP fails on charts).
# **This is the picture that grounds RQ2 in real numbers.**

# %%
def encode_batch(images: list, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run pretrained CLIP, return (B, B) similarity + L2-normalized embeddings.

    Notice: HF already L2-normalizes image_embeds and text_embeds for us. We
    re-normalize defensively in case a future version changes the contract.
    """
    inputs = processor(text=texts, images=images, return_tensors="pt",
                       padding=True, truncation=True)
    with torch.no_grad():
        out = model(**inputs)
        img_emb = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = out.text_embeds  / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (img_emb @ txt_emb.T) * model.logit_scale.exp()   # apply trained temperature
    return sim, img_emb, txt_emb


# CIFAR comparison batch — one image per class for a clean 10x10.
np.random.seed(42)
cifar_indices = [int(np.random.choice([i for i, t in enumerate(_cifar.targets) if t == k]))
                 for k in range(10)]
cifar_images = [_cifar[i][0] for i in cifar_indices]
cifar_texts = CIFAR_CAPTIONS
cifar_sim, cifar_ie, cifar_te = encode_batch(cifar_images, cifar_texts)
print(f"CIFAR sim shape: {tuple(cifar_sim.shape)}")

# ChartQA batch.
chart_images = [s["image"] for s in chart_samples]
chart_texts = [s["question"] for s in chart_samples]
chart_sim, chart_ie, chart_te = encode_batch(chart_images, chart_texts)
print(f"Chart sim shape: {tuple(chart_sim.shape)}")


# %%
# Side-by-side similarity matrices — Figure for the RQ2 chapter.
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Align color scales so the visual contrast is honest.
vmin = min(cifar_sim.min().item(), chart_sim.min().item())
vmax = max(cifar_sim.max().item(), chart_sim.max().item())

panels = [
    (axes[0], cifar_sim, "CIFAR (natural) — CLIP's home turf",
     CIFAR_CLASSES, CIFAR_CLASSES),
    (axes[1], chart_sim, "ChartQA — out-of-distribution",
     [f"q{i}" for i in range(N_SAMPLES)], [f"img{i}" for i in range(N_SAMPLES)]),
]
for ax, sim, title, xlabels, ylabels in panels:
    im = ax.imshow(sim.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xlabels)))
    ax.set_yticks(range(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("text")
    ax.set_ylabel("image")
    plt.colorbar(im, ax=ax, shrink=0.7)

fig.suptitle("Pretrained CLIP similarity — natural vs chart\n(same trained temperature, shared color scale)", fontsize=12)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Part 10 — Quantify the gap
#
# For each row of the similarity matrix, rank the diagonal entry among that row's
# scores. Rank 1 = correct retrieval. Then compare CIFAR vs ChartQA.

# %%
def retrieval_stats(sim: torch.Tensor) -> dict:
    N = sim.shape[0]
    ranks = []
    for i in range(N):
        row = sim[i]
        rank = (row > row[i]).sum().item() + 1   # how many off-diag beat the diag, +1
        ranks.append(rank)
    return {
        "top1_acc":     sum(r == 1 for r in ranks) / N,
        "mean_rank":    sum(ranks) / N,
        "diag_mean":    sim.diag().mean().item(),
        "offdiag_mean": (sim.sum() - sim.diag().sum()).item() / (N * N - N),
    }


cifar_stats = retrieval_stats(cifar_sim)
chart_stats = retrieval_stats(chart_sim)

print(f"{'metric':<15} {'CIFAR':>10} {'ChartQA':>10} {'gap':>10}")
print("-" * 50)
for k in ["top1_acc", "mean_rank", "diag_mean", "offdiag_mean"]:
    gap = cifar_stats[k] - chart_stats[k]
    print(f"{k:<15} {cifar_stats[k]:>10.3f} {chart_stats[k]:>10.3f}  {gap:>+9.3f}")


# %%
# Quick bar chart of the gap.
metrics = ["top1_acc", "diag_mean", "offdiag_mean"]
xs = np.arange(len(metrics))
w = 0.35
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(xs - w/2, [cifar_stats[k] for k in metrics], w, label="CIFAR (natural)", color="#2ca02c")
ax.bar(xs + w/2, [chart_stats[k] for k in metrics], w, label="ChartQA (OOD)", color="#d62728")
ax.set_xticks(xs)
ax.set_xticklabels(metrics)
ax.set_title("Pretrained CLIP: the gap between natural and chart data")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Part 11 — Failure-mode gallery
#
# For each chart, show:
#   - the image
#   - the TRUE question
#   - CLIP's top-3 retrieved questions from the bank
# If "TRUE" is missing from the top-3, CLIP confidently picked a wrong question.

# %%
N_INSPECT = 5
TOP_K = 3

fig, axes = plt.subplots(N_INSPECT, 2, figsize=(15, 3 * N_INSPECT),
                        gridspec_kw={"width_ratios": [1, 2]})
for i in range(N_INSPECT):
    ax_img, ax_text = axes[i]
    ax_img.imshow(chart_samples[i]["image"])
    ax_img.set_xticks([]); ax_img.set_yticks([])
    ax_img.set_title(f"chart {i}", fontsize=10)

    scores = chart_sim[i]
    top_idx = scores.topk(TOP_K).indices.tolist()
    true_q = chart_samples[i]["question"]
    if len(true_q) > 80:
        true_q = true_q[:77] + "..."

    ax_text.axis("off")
    ax_text.text(0.0, 0.95, f"TRUE: {true_q}",
                 fontsize=8, family="monospace", color="darkgreen", weight="bold",
                 transform=ax_text.transAxes)
    ax_text.text(0.0, 0.78, "CLIP top-3:",
                 fontsize=8, family="monospace",
                 transform=ax_text.transAxes)

    y = 0.65
    for rank, j in enumerate(top_idx):
        q = chart_samples[j]["question"]
        if len(q) > 70:
            q = q[:67] + "..."
        tag = "  <- TRUE" if j == i else ""
        ax_text.text(0.0, y, f"  #{rank+1} ({scores[j]:.2f})  {q}{tag}",
                     fontsize=7.5, family="monospace",
                     color=("darkgreen" if j == i else "black"),
                     transform=ax_text.transAxes)
        y -= 0.15

fig.suptitle("Pretrained CLIP — chart retrieval failures", fontsize=12)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Part 12 — Per-patch features (what BLIP-2 / Qwen2-VL consume)
#
# CLIP normally returns ONE pooled vector per image (the CLS-projected embedding —
# `out.image_embeds`). But generative VLMs like BLIP-2 and Qwen2-VL bypass that pooling
# entirely — they consume the **full per-patch hidden state** straight from the vision tower.
# That's how they achieve fine-grained patch↔token alignment that CLIP cannot.
#
# For ViT-B/32 @ 224×224:
#   - patch grid = 224/32 = 7 → 7×7 = 49 patches  + 1 CLS  =  **50 tokens**
#   - hidden dim = 768
#   - per-image feature tensor shape = **(B, 50, 768)**
#
# **How to extract them:** instead of calling `model(**inputs)` (which gives you the
# pooled output), call `model.vision_model(pixel_values=..., output_hidden_states=True)`
# and grab `.last_hidden_state`. This is the "internal access" trick — works for every
# HF VLM where you want unpooled features.

# %%
def get_patch_features(pil_image) -> torch.Tensor:
    """Per-patch hidden states from CLIP's vision tower. Shape (1, 50, 768)."""
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        vision_out = model.vision_model(
            pixel_values=inputs.pixel_values,
            output_hidden_states=True,    # ask the model to keep intermediate states
        )
    # last_hidden_state = output of the final transformer block (before any pooling).
    # hidden_states (tuple) = output of EVERY block, including the input embedding.
    return vision_out.last_hidden_state


cat_patches   = get_patch_features(cat_img)
chart_patches = get_patch_features(chart_samples[0]["image"])
print(f"cat patch features:   {tuple(cat_patches.shape)}    (token 0 = CLS, tokens 1..49 = patches)")
print(f"chart patch features: {tuple(chart_patches.shape)}")
print()
print("Token 0     = CLS — the pooled vector CLIP normally returns to you")
print("Tokens 1..N = per-patch features — what BLIP-2's Q-Former queries against")
print()
print("Files 05/06/07 will feed exactly this tensor into their bridges.")


# %% [markdown]
# ## Part 13 — Universal pattern recap
#
# **You now know enough to load any HF VLM.** The pattern is identical across the field:
#
# ```python
# # CLIP                              # BLIP-2                          # Qwen2-VL
# from transformers import (          from transformers import (         from transformers import (
#     CLIPModel,                          Blip2ForConditionalGeneration,    Qwen2VLForConditionalGeneration,
#     CLIPProcessor,                      Blip2Processor,                   Qwen2VLProcessor,
# )                                   )                                  )
#
# model     = CLIPModel.from_pretrained(           model     = Blip2For…from_pretrained(    model     = Qwen2VL…from_pretrained(
#     "openai/clip-vit-base-patch32")               "Salesforce/blip2-opt-2.7b")            "Qwen/Qwen2-VL-2B-Instruct")
# processor = CLIPProcessor.from_pretrained(...)   processor = Blip2Processor.from_pretrained(...)  processor = Qwen2VLProcessor.from_pretrained(...)
# model.eval()                                      model.eval()                                      model.eval()
#
# inputs = processor(text=..., images=..., return_tensors="pt", padding=True)
# out = model(**inputs)
# ```
#
# **What changes between models:**
# - The model class (different architecture: dual-encoder vs Q-Former vs decoder-fused).
# - The output dataclass fields (CLIPOutput vs Blip2Output vs Qwen2VLOutput).
# - The image preprocessing constants (image_mean/std are model-specific!).
# - For generative models: instead of `out = model(**inputs)`, you call
#   `model.generate(**inputs)` to autoregressively sample text.
#
# **What stays the same:**
# - The 3-step contract: `from_pretrained` × 2 + `model(**processor(...))`.
# - The processor returns `pixel_values` / `input_ids` / `attention_mask`.
# - You can always peek inside via `model.named_children()` and friends.
#
# **You're now ready for `05_blip2_study.py`** — same pattern, bigger model,
# different output.


# %% [markdown]
# ### What this file contributes to the thesis
#
# - **Side-by-side similarity matrix** (Part 9)  → motivation figure for RQ2 chapter.
# - **Gap numbers** (Part 10)                    → seed values for Table 2.1.
# - **Failure gallery** (Part 11)                → qualitative section header for RQ2.
# - **Per-patch feature extraction** (Part 12)   → bridge to BLIP-2 (file 05).
# - **HF workflow muscle memory** (Parts 1-7)    → all of Weeks 2-6 reuse this pattern.
#
# **Scale-up for actual thesis numbers:** raise `N_SAMPLES` to ~500, add DocVQA
# and ScienceQA via the same loader, compare the gap across domains (this becomes
# the table that grounds RQ2 + RQ4 empirically).

# %%
for i in model.children():
    print(i)
# %%
