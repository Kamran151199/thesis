# %% [markdown]
# # thesis_related__blip2_probe
#
# **Goal:** the same way we probed CLIP, now probe BLIP-2. Understand the ONE
# new architectural idea (the Q-Former), then run real inference on natural +
# chart images.
#
# **What's different from CLIP / LLaVA:**
#
# ```
# CLIP    : image → ViT → CLS token → SCORE vs caption                      (contrastive)
# LLaVA   : image → ViT → 576 patch tokens → Linear projector → LLM        (generative)
# BLIP-2  : image → ViT → 576 patch tokens → Q-FORMER → 32 query tokens → LLM   (generative)
#                                                ↑↑↑↑↑↑↑↑
#                                       NEW: a tiny transformer that COMPRESSES
#                                       576 patches into 32 learned "summary tokens"
# ```
#
# The Q-Former is BLIP-2's whole contribution. It's a 110M-parameter "querying
# transformer" with 32 learnable query vectors that cross-attend to image
# features and produce a 32-token compressed image representation.
#
# **Trade-off vs LLaVA:**
# - LLaVA sends 576 image tokens to the LLM → richer, more expensive
# - BLIP-2 sends only 32 → cheaper, slight info loss
# - Q-Former is also useful as a learned alignment objective during pretraining

# %% [markdown]
# ## Part 1 — Setup & imports

# %%
import pickle
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor
import torchvision


# %% [markdown]
# ## Part 2 — Load BLIP-2
#
# We use `Salesforce/blip2-opt-2.7b` — BLIP-2's smallest variant.
# About 3.7 GB download. Fits on a laptop with float16.
#
# Architecture: EVA-CLIP-ViT-G/14 (vision, frozen) + Q-Former (110M, trainable)
#              + OPT-2.7B (LLM, frozen).
#
# Total params ~3.7B. The Q-Former is 110M of them — that's the only thing
# that gets trained during BLIP-2's published pretraining.

# %%
MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# Load in float16 to save memory. First run: ~3.7 GB download.
# device_map="auto" puts it on MPS (Mac) or CUDA if available, else CPU.
processor = Blip2Processor.from_pretrained(MODEL_NAME)
model = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()

print(f"Loaded: {MODEL_NAME}")
print(f"  total params:      {sum(p.numel() for p in model.parameters()):>14,}")
print(f"  device:            {next(model.parameters()).device}")
print(f"  vision tower:      {type(model.vision_model).__name__}")
print(f"  q-former:          {type(model.qformer).__name__}")
print(f"  language model:    {type(model.language_model).__name__}")


# %% [markdown]
# ## Part 3 — Inspect the submodules
#
# Same pattern as the CLIP probe. BLIP-2's top-level children are the three
# pieces (vision, q-former, llm) plus a couple of projectors.

# %%
print("Top-level submodules:")
for name, mod in model.named_children():
    n_params = sum(p.numel() for p in mod.parameters())
    pct = 100 * n_params / sum(p.numel() for p in model.parameters())
    print(f"  model.{name:20s}  →  {type(mod).__name__:35s}  ({n_params:>13,} params, {pct:>5.1f}%)")

# Expected output:
#   vision_model           → Blip2VisionModel              (~1.0B, ~27%)
#   qformer                → Blip2QFormerModel             (~110M,  ~3%)
#   language_projection    → Linear                        (~2M,    <1%)
#   language_model         → OPTForCausalLM                (~2.7B, ~70%)
#
# So 97% of params are FROZEN (vision + LLM). Only the Q-Former + tiny projector trains.


# %% [markdown]
# ## Part 4 — Peek inside the Q-Former (the new concept)
#
# The Q-Former is itself a transformer with two attention types:
#   1. **Self-attention**: queries attend to each other (32 → 32)
#   2. **Cross-attention**: queries attend to the image features (32 → 576)
#
# Plus the 32 learnable query tokens themselves — those are the "summary slots."

# %%
print("Q-Former internals:")
qformer = model.qformer

# The 32 query tokens are stored in the parent model as `query_tokens`
print(f"  query_tokens shape:  {tuple(model.query_tokens.shape)}     (1, 32, qformer_hidden)")
print(f"  → 32 learnable 'summary slots' that will attend to the image")
print()

# Inspect Q-Former layers
print(f"  Q-Former layers:      {len(qformer.encoder.layer)} blocks")
print(f"  First block type:     {type(qformer.encoder.layer[0]).__name__}")
print()

# A Q-Former layer has BOTH self-attention AND cross-attention
layer0 = qformer.encoder.layer[0]
print(f"  Block 0 components:")
for name, _ in layer0.named_children():
    print(f"    - {name}")
# You'll see: attention (self), crossattention (cross to image), intermediate (FFN), output


# %% [markdown]
# ## Part 5 — Q-Former data flow, drawn
#
# ```
# image (B, 3, 224, 224)
#    ↓ vision_model (EVA-CLIP-ViT-G/14)
# patch features (B, 257, 1408)              ← 256 patches + 1 CLS
#    │
#    ├──────── cross-attention KV ──────┐
#    │                                   │
#    ↓                                   ↓
# query_tokens (1, 32, 768)              ↓
#    ↓ Q-Former blocks                   │
#    │   ┌── self-attn (queries ↔ queries) ──┐
#    │   ├── cross-attn (queries ← patches) ─┘
#    │   ├── ffn
#    │   └── (repeat × 12 blocks)
#    ↓
# query outputs (B, 32, 768)
#    ↓ language_projection (Linear: 768 → 2560)
# "image tokens in LLM space" (B, 32, 2560)
#    ↓
# CONCAT with text prompt tokens
#    ↓
# OPT-2.7B language model → autoregressive text generation
# ```
#
# **The compression**: 576 patch tokens → 32 query tokens, learned by the Q-Former
# attention pattern. Each query attends to a different aspect of the image.
#
# **Compare to LLaVA**: LLaVA sends ALL 576 to the LLM (each projected independently
# by a simple MLP). BLIP-2 first lets 32 queries pick what's important.


# %% [markdown]
# ## Part 6 — The Processor
#
# Same shape as CLIP's processor — image_processor + tokenizer bundle.
# The image_processor uses CLIP's mean/std (since BLIP-2 wraps CLIP's ViT).

# %%
print("Image processor:")
print(f"  type:         {type(processor.image_processor).__name__}")
print(f"  size:         {processor.image_processor.size}")
print(f"  image_mean:   {processor.image_processor.image_mean}")
print(f"  image_std:    {processor.image_processor.image_std}")
print()
print("Tokenizer:")
print(f"  type:         {type(processor.tokenizer).__name__}")
print(f"  vocab size:   {processor.tokenizer.vocab_size}")
print(f"  pad token:    {processor.tokenizer.pad_token!r}")
print(f"  bos token:    {processor.tokenizer.bos_token!r}")


# %% [markdown]
# ## Part 7 — Inference: caption a CIFAR image
#
# BLIP-2 is generative — `model.generate(...)` autoregressively samples a caption.
# This is what your mini LLaVA does, just with a real pretrained model.

# %%
_cifar = torchvision.datasets.CIFAR10(root="exploration/data", train=False, download=True)
cat_idx = next(i for i, t in enumerate(_cifar.targets) if t == 3)  # 3 = cat
cat_img = _cifar[cat_idx][0]
print(f"cat image size: {cat_img.size}")

# Inference: just images, no text prompt (pure captioning)
inputs = processor(images=cat_img, return_tensors="pt").to(model.device, torch.float16)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=30)
caption = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
print(f"\nBLIP-2 caption: {caption!r}")


# %% [markdown]
# ## Part 8 — Inference: VQA on a chart
#
# Now the interesting part. Load a ChartQA image from your previous probe cache
# and ask BLIP-2 a question about it.
#
# Prompt format for BLIP-2: `"Question: <q> Answer:"` — model autocompletes the answer.

# %%
CACHE_PATH = Path("exploration/data/chartqa_probe_sample.pkl")
with open(CACHE_PATH, "rb") as f:
    chart_samples = pickle.load(f)

chart = chart_samples[0]
print(f"chart question: {chart['question']!r}")

prompt = f"Question: {chart['question']} Answer:"
inputs = processor(images=chart["image"], text=prompt, return_tensors="pt").to(model.device, torch.float16)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=30)

# generate() returns both the prompt and the answer; strip prompt
generated = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
answer = generated[len(prompt):].strip() if generated.startswith(prompt) else generated
print(f"\nBLIP-2 answer: {answer!r}")


# %% [markdown]
# ## Part 9 — Extract the 32 Q-Former query outputs
#
# These are the "compressed image" vectors that the LLM actually sees.
# Inspecting them gives intuition for what the Q-Former extracted.

# %%
with torch.no_grad():
    inputs = processor(images=chart["image"], return_tensors="pt").to(model.device, torch.float16)
    # Run just vision + Q-Former, stop before LLM
    vision_out = model.vision_model(pixel_values=inputs.pixel_values).last_hidden_state
    image_attention_mask = torch.ones(vision_out.size()[:-1], dtype=torch.long).to(model.device)

    # Q-Former forward — expects (image features + image_attention_mask) and the 32 learnable queries
    query_tokens = model.query_tokens.expand(vision_out.shape[0], -1, -1)
    qformer_out = model.qformer(
        query_embeds=query_tokens,
        encoder_hidden_states=vision_out,
        encoder_attention_mask=image_attention_mask,
    ).last_hidden_state

    # Project to LLM space
    llm_input_tokens = model.language_projection(qformer_out)

print(f"vision output:      {tuple(vision_out.shape)}        ← (B, 257, 1408)  patch features")
print(f"query tokens (in):  {tuple(query_tokens.shape)}      ← (B, 32, 768)    learnable queries")
print(f"qformer output:     {tuple(qformer_out.shape)}       ← (B, 32, 768)    Q-Former summary")
print(f"LLM input tokens:   {tuple(llm_input_tokens.shape)}  ← (B, 32, 2560)   projected to OPT-2.7B space")
print(f"\n→ These 32 vectors are 'the image' as far as the LLM is concerned.")


# %% [markdown]
# ## Part 10 — Family comparison
#
# Put the three architectures side-by-side. Same pretrained vision tower
# underneath — different bridges + different decoders.

# %%
comparison = """
┌──────────────┬─────────────────────┬───────────────────────┬───────────────────┐
│              │ CLIP                │ LLaVA-1.5             │ BLIP-2            │
├──────────────┼─────────────────────┼───────────────────────┼───────────────────┤
│ vision       │ CLIP ViT-B/32       │ CLIP ViT-L/14         │ EVA-CLIP-ViT-G/14 │
│ vision out   │ 1 CLS vector        │ 576 patch tokens      │ 257 patch tokens  │
│ bridge       │ Linear projection   │ 2-layer MLP projector │ Q-Former (110M)   │
│ tokens to LM │ N/A (no LM)         │ 576                    │ 32                │
│ LM           │ none (text encoder) │ LLaMA / Vicuna 7B-13B │ OPT-2.7B / FlanT5 │
│ output       │ similarity score    │ generated text        │ generated text    │
│ trainable    │ everything          │ projector + LLM       │ Q-Former + proj   │
└──────────────┴─────────────────────┴───────────────────────┴───────────────────┘
"""
print(comparison)


# %% [markdown]
# ## What this file gave you
#
# - Loaded and ran real BLIP-2 OPT-2.7B inference on both natural and chart images
# - Saw the Q-Former's role: compressing 576 patches → 32 summary tokens
# - Extracted the 32 query outputs (what the LLM actually sees)
# - Family comparison: CLIP vs LLaVA vs BLIP-2 in one table
#
# ### Next step: FINE-TUNE BLIP-2 on Colab Pro+
#
# Now that you understand the architecture, you can fine-tune the Q-Former
# (the small trainable bit) on a chart-QA or document-QA task. That's the
# **next file** — a Colab notebook for actual fine-tuning.
#
# Architecture you'll fine-tune:
#
# ```
# FROZEN:        vision_model    (1.0B params)  — saves 95% of memory
# TRAINABLE:     qformer         (110M params)  — the Q-Former
# TRAINABLE:     language_projection  (~2M)      — the small linear adapter
# FROZEN:        language_model  (2.7B params)  — saves more memory
# ```
#
# With QLoRA, you can also LoRA-tune the LLM if needed (RQ6). For tonight,
# train just the Q-Former on a slice of ScienceQA.

# %%
