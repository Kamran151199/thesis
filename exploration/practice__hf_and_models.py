# %% [markdown]
# # Practice — HuggingFace & Pretrained Models (5 Tasks)
#
# **Self-test file.** Each task pushes a different muscle:
# - HuggingFace mechanics (Processor / Model / generate())
# - Model internals (hidden states, attention, learned embeddings)
# - Matplotlib analytics (heatmaps, overlays, distributions)
# - torch ops (slicing, masking, reductions, similarity)
# - Basic stats (mean / std / cosine similarity / accuracy)
#
# **Rules:**
# - No peeking at `thesis_related__clip_chartqa_probe.py` unless you're stuck.
# - When stuck, try to derive from first principles before grabbing reference code.
# - Each task has:
#     - **Goal**     — the visible/numeric output you should produce
#     - **Hints**    — tools/methods worth considering
#     - **Success**  — what "done" looks like
#     - **Stretch**  — optional harder version
#
# Estimated total time: ~5-7 hours across all 5. Do them in any order.
#
# **Setup is shared** — run the next two cells once, then jump to whichever task.

# %% [markdown]
# ## Setup — shared imports and model

# %%
from matplotlib import axes
import torch
from torch import return_types, tensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10
from transformers import CLIPModel, CLIPProcessor
import torchvision

torch.manual_seed(0)
np.random.seed(0)


MODEL_NAME = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

# CIFAR for the image-side tasks
cifar = torchvision.datasets.CIFAR10(root="exploration/data", train=False, download=True)
CIFAR_CLASSES = cifar.classes
CIFAR_CAPTIONS = [f"a photo of a {c}" for c in CIFAR_CLASSES]

print(f"loaded model: {sum(p.numel() for p in model.parameters()):,} params")
print(f"cifar test images: {len(cifar)}")
print(f"classes: {CIFAR_CLASSES}")


# %% [markdown]
# # Task 1 — Zero-shot CIFAR accuracy at scale (~30 min)
#
# **Warm-up task.** The probe ran on 1 cat photo. Now scale it up properly.
#
# ### Goal
# - Run pretrained CLIP zero-shot classification on **500 random CIFAR test images**.
# - Print final **top-1 accuracy**.
# - Plot a **10×10 confusion matrix** (row = true class, col = predicted class).
#
# ### Hints
# - Encode the 10 captions ONCE (you don't re-encode for every image).
# - Batch the image encoding (e.g., 64 images at a time) — running them one-by-one will be slow.
# - For confusion: a simple `torch.zeros(10, 10)` accumulator works fine; no sklearn needed.
# - Row-normalize the confusion before plotting (so colors aren't dominated by class imbalance).
#
# ### Success
# - You see a number like "Top-1 accuracy: 58.2%" (expect 50-70% — CLIP-base on CIFAR is decent but not great).
# - Diagonal of the confusion matrix glows brighter than off-diagonal.
# - One or two off-diagonal hotspots (cat↔dog, car↔truck, deer↔horse).
#
# ### Stretch
# - Try **3 different caption templates** and report which wins:
#   - `"a photo of a {class}"`
#   - `"{class}"`
#   - `"a low-resolution photo of a {class}"` (matches CIFAR's actual quality)
# - This is the famous "prompt engineering" CLIP effect — wording moves accuracy by 2-5%.

# %% [markdown]
# ### Task 1 — workspace
# 

# %% [markdown]
# #### Utils

# %%
import random

def target_idx_class_map(targets: list[int]) -> dict[int, int]:
    out = {}
    for idx, c in enumerate(targets):
        out[idx] = c
    return out


def target_per_class(target_class_map: dict[int, int]):
    num_cls = len(cifar.class_to_idx.values())
    seen = set()
    shuffled_idx = random.sample(list(target_class_map.keys()), len(target_class_map.keys()))
    out = []
    for idx in shuffled_idx:
        if target_class_map[idx] in seen:
            continue
        out.append(idx)
        seen.add(target_class_map[idx])
        if len(seen) == num_cls:
            break 
    return out


def make_balanced_batch(
    cifar: CIFAR10,
    batch_size: int,
    for_training: bool = False,
) -> tuple:
    if for_training and batch_size:
        raise ValueError("Training batch-size must be inferred from the distinct class number")
    iters = int(np.ceil(batch_size / len(cifar.class_to_idx.values())))

    chosen_idxs = []
    idx_class_map = target_idx_class_map(cifar.targets)
    
    for _ in range(iters):
        chosen_idxs.extend(target_per_class(idx_class_map))

    out_idx = chosen_idxs[:batch_size]

    batch_images = cifar.data[out_idx]
    batch_texts = list(map(lambda x: cifar.classes[x], np.array(cifar.targets)[out_idx]))
    return batch_images, batch_texts


def draw_balanced_batch(sample_imgs, sample_texts):
    _, ax = plt.subplots(len(sample_imgs), 2, figsize=(len(sample_imgs), len(sample_imgs) * 2), layout="constrained")

    for i in range(len(sample_imgs)):
        img_panel: axes.Axes = ax[i, 0]
        text_panel: axes.Axes = ax[i, 1]
        
        img_panel.imshow(sample_imgs[i])
        img_panel.set_xticks([]); img_panel.set_xlabel(""); img_panel.set_yticks([])

        text_panel.text(0.5, 0.5, sample_texts[i], fontsize=12, ha='center', va='center')
        text_panel.set_xticks([]); text_panel.set_yticks([]); text_panel.set_xlabel(""); text_panel.set_ylabel("");
        text_panel.spines['top'].set_visible(False)
        text_panel.spines['right'].set_visible(False)
        text_panel.spines['bottom'].set_visible(False)
        text_panel.spines['left'].set_visible(False)

# NOTE: uncomment if you wanna have a look
# sample_imgs, sample_texts = make_balanced_batch(cifar, 5)
# draw_balanced_batch(sample_imgs, sample_texts)

def make_batch(cifar: CIFAR10, batch_size: int = 64):
    idx = torch.randint(0, len(cifar.data), (batch_size, ))
    imgs = cifar.data[idx]
    labels = torch.tensor(cifar.targets)[idx]
    return imgs, labels


# %% [markdown]
# #### Task itself
sample_text_inputs = processor(
    text=cifar.classes,
    return_tensors="pt",
    padding=True
)
text_embeddings: torch.Tensor = model.get_text_features(**sample_text_inputs).pooler_output
normalized_embs = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

processed = 0
batch_size = 64
conf_matrix = torch.zeros((10, 10))
correct = 0

while processed != 500:
    if processed + batch_size > 500:
        batch_size = 500 - processed
    batch_imgs, batch_labels = make_batch(cifar, batch_size)
    inputs = processor(
        images=batch_imgs,
        return_tensors="pt",
        padding=True
    )
    image_embs: torch.Tensor = model.get_image_features(inputs.pixel_values).pooler_output
    assert image_embs.shape == (batch_size, normalized_embs.shape[-1])
    
    normalized_img_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

    sim = normalized_img_embs @ normalized_embs.T
    predicted = torch.argmax(sim, dim=-1)
    correct += (predicted == batch_labels).sum()
    conf_matrix.index_put_(
      (batch_labels, predicted),
      torch.ones_like(batch_labels, dtype=conf_matrix.dtype),
      accumulate=True,
    )
    processed += batch_size

acc = correct / 500


# %% [markdown]
# #### Plot results

# %%
def plot_results(conf_matrix: torch.Tensor, acc: torch.Tensor):
    normalized = conf_matrix / conf_matrix.norm(dim=-1, keepdim=True)
    _, axes = plt.subplots(figsize=(10, 10))
    axes.set_title(f"Confusion Matrix — CLIP zero-shot on CIFAR-10  (acc = {acc.item()*100:.1f}%)")    
    im = axes.imshow(normalized, cmap="Blues")
    axes.set_ylabel("true")
    axes.set_xlabel("predicted")
    axes.set_yticks(list(cifar.class_to_idx.values()), cifar.classes)
    axes.set_xticks(list(cifar.class_to_idx.values()), cifar.classes, rotation=45, ha="right")

    for i in range(10):
        for j in range(10):
            val = normalized[i, j]
            axes.text(j, i, f"{val:.2f}", ha="center", va="center",
                      color="white" if val > 0.05 else "black", fontsize=8) 
    plt.colorbar(im, ax=axes, shrink=0.7)
    plt.tight_layout()
    plt.show()
plot_results(conf_matrix, acc)


# %% [markdown]
# # Task 2 — Position embedding has hidden 2D structure (~1 hr)
#
# **You'll dig into a single learned tensor and prove the model figured out spatial layout from scratch.**
#
# CLIP's vision tower has a learned **position embedding** of shape `(50, 768)`:
#   - Index 0 = CLS position
#   - Indices 1..49 = the 49 patch positions (arranged as a 7×7 grid)
#
# **Hypothesis:** positions that are spatial neighbors (e.g., patch (3,3) and patch (3,4))
# should have **more similar embeddings** than positions far apart (e.g., (0,0) and (6,6)) —
# even though the model was never told this; it learned it from gradient descent alone.
#
# ### Goal
# Test the hypothesis. Produce:
# 1. A **7×7 heatmap** showing cosine similarity between **one chosen center position**
#    (e.g., the center patch at grid position (3, 3) → flat index `1 + 3*7 + 3 = 22`)
#    and every other patch position. If the model learned 2D structure, you'll see a bright
#    spot at the center fading outward — like a Gaussian blob.
# 2. A **3×3 grid of heatmaps** for 9 different center positions: the 4 corners, 4 edge-midpoints,
#    and the dead center. Each should "glow" around its own location.
#
# ### Hints
# - The position embedding lives at:
#   `pos = model.vision_model.embeddings.position_embedding.weight`  → shape `(50, 768)`
# - Drop index 0 (CLS): `patch_pos = pos[1:]` → `(49, 768)`.
# - Cosine similarity to position `k`: `F.cosine_similarity(patch_pos, patch_pos[k:k+1], dim=-1)` → `(49,)`.
# - Reshape `(49,)` → `(7, 7)` to put it back on the 2D grid.
# - Use `plt.imshow(...)` with `cmap='hot'` or `'viridis'`.
#
# ### Success
# - The 7×7 heatmap shows a clear bright spot at your chosen position.
# - The 3×3 grid shows the bright spot **moves** with the chosen position — confirming the
#   model learned a 2D coordinate system inside that tensor.
#
# ### Stretch
# - Compute the **average similarity between adjacent patches** (4-connectivity) vs **diagonal patches**
#   vs **far-apart patches**. Numbers should decrease monotonically with distance.
# - This recreates one of the iconic figures from the original ViT paper (Dosovitskiy 2021).

# %% Task 2 — workspace
# write your code here


# %% [markdown]
# # Task 3 — The modality gap (~1 hr)
#
# **Famous CLIP phenomenon, surprising even to the authors:** image embeddings and text embeddings,
# even after L2-normalization in the shared space, don't fully overlap. They form **two distinct
# clusters** separated by a measurable "gap." Liang et al. 2022 ("Mind the gap") wrote a whole paper on this.
#
# You're going to **see** and **measure** this gap.
#
# ### Goal
# 1. Encode **100 random CIFAR test images** → image embeddings (100, 512), L2-normalized.
# 2. Encode **100 random captions** (e.g., reuse the 10 CIFAR captions, repeated, or generate variations) → text embeddings (100, 512), L2-normalized.
# 3. Compute and **print** three numbers:
#     - mean cosine similarity **within** images (image-image, off-diagonal of a 100×100 sim matrix)
#     - mean cosine similarity **within** texts  (text-text, off-diagonal)
#     - mean cosine similarity **across** modalities (image-text, off-diagonal)
#   You should see: within > across (the gap).
# 4. **Visualize** the 200 embeddings in 2D using PCA:
#     - Stack to `(200, 512)`, compute PCA to 2D (use `torch.pca_lowrank` or `sklearn.decomposition.PCA`),
#       scatter plot, color by modality (images blue, texts red).
#     - You should see **two visibly separated clusters**.
#
# ### Hints
# - Use `out = model(**inputs)`, then `out.image_embeds` and `out.text_embeds` (already L2-normalized internally).
# - For pairwise similarity, `emb @ emb.T` gives you the full similarity matrix.
# - To exclude self-similarity (diagonal = 1) when computing means, mask it out:
#   `mask = ~torch.eye(N, dtype=bool); sim_mean = sim[mask].mean()`
# - For PCA, the quickest is `torch.pca_lowrank(stacked, q=2)`.
#
# ### Success
# - Three printed numbers, with within-modality clearly > cross-modality.
# - A scatter plot where blue and red are visually separable.
# - You can quote a single-number "modality gap" = `1 - cos(mean_image_emb, mean_text_emb)`.
#
# ### Stretch
# - Try **subtracting** the mean image embedding from all image embeddings, and the mean text
#   embedding from all text embeddings. Re-run zero-shot CIFAR classification with the
#   centered embeddings. Does accuracy improve? (This is "modality-gap correction" — a known trick.)

# %% Task 3 — workspace
# write your code here


# %% [markdown]
# # Task 4 — Where does class information emerge in the vision tower? (~1.5 hr)
#
# **The deepest task in this set.** You'll peer inside the 12 transformer blocks of CLIP's
# vision tower and figure out at which layer "this is a cat" becomes a usable feature.
#
# ### Goal
# Pick **2 visually-different CIFAR classes** (e.g., `cat` vs `automobile`). Grab **10 images of each**.
#
# For each image, extract the **CLS token's hidden state at every transformer block** —
# that is, 13 vectors per image (input embedding + 12 block outputs).
#
# At each layer `l`, compute:
# - mean cosine similarity of **same-class pairs** (cat-cat AND auto-auto)   → `same_class_sim[l]`
# - mean cosine similarity of **cross-class pairs**   (cat-auto)              → `cross_class_sim[l]`
#
# Plot **two curves** on the same axes: x = layer index (0..12), y = cosine similarity.
# Add a separation curve: `same_class_sim - cross_class_sim`.
#
# ### Hints
# - Run the vision tower with `output_hidden_states=True`:
#   ```python
#   vis_out = model.vision_model(pixel_values=..., output_hidden_states=True)
#   vis_out.hidden_states  # tuple of length 13 = input + 12 block outputs
#   ```
#   Each element is shape `(B, 50, 768)`. The CLS token is `[:, 0, :]`.
# - For mean over multiple pair-similarities, organize same-class and cross-class as boolean masks
#   over a `(20, 20)` similarity matrix.
# - L2-normalize CLS vectors before computing cosine sim.
#
# ### Success
# - A plot where the **same-class curve** rises faster than the **cross-class curve** as you go deeper.
# - The **separation** (same - cross) starts near 0 and grows monotonically toward the final layer.
# - You can answer: "class-level structure becomes meaningfully separable at layer ~X."
# - Hard rule of thumb: expect separation to emerge sharply around layers 6-10 in CLIP-base.
#
# ### Stretch
# - Repeat with `cat` vs `dog` (visually SIMILAR classes). Where does the curve diverge differently?
# - Compare with the **post-projection** embeddings (after `visual_projection` is applied to CLS).
#   Does the gap widen or narrow after projection? Why?

# %% Task 4 — workspace
# write your code here


# %% [markdown]
# # Task 5 — Patch occlusion saliency (~1.5 hr)
#
# **What part of the image does CLIP actually care about?** Without using attention maps,
# you can answer this by *occlusion*: blank out a region of the image and see how the
# embedding changes. Big change = important region.
#
# ### Goal
# Pick **one chart image** (load from `chartqa_probe_sample.pkl`) or **one CIFAR cat image**.
# Pre-process it to 224×224 and **encode** to get the "original" image embedding (call it `e_orig`).
#
# For each of the **49 patch positions** in the 7×7 grid:
# 1. Make a copy of the 224×224 image.
# 2. **Zero out** the 32×32 pixel block at that position.
# 3. Encode → `e_masked[p]`.
# 4. Compute `saliency[p] = 1 - cos(e_orig, e_masked[p])` (= how much the embedding changes).
#
# Reshape `saliency` from `(49,)` to a `(7, 7)` grid. Visualize:
# - Side-by-side: the original 224×224 image on the left, the 7×7 saliency heatmap (upsampled to
#   224×224 via `np.kron` or `repeat_interleave`) overlaid in semi-transparency on the right.
#
# ### Hints
# - Easiest: use `inputs.pixel_values` (already 224×224, already normalized), mask in normalized space.
#   Setting a region to 0 in normalized space is roughly "set to mean color in pixel space."
# - Loop over 49 positions, but **batch** the 49 masked variants into one forward pass.
#   So you build a `(49, 3, 224, 224)` tensor of masked variants, call the model once.
# - For overlay, see the matplotlib pattern in the probe file Part 12 (the CLS-attention overlay
#   in `03_vision_transformer.py` also uses `repeat_interleave` for upsampling).
#
# ### Success
# - The saliency heatmap **highlights structurally meaningful regions**:
#     - Cat photo: bright on face, fur, body.
#     - Chart: bright on the bars/lines, dark on whitespace.
# - One number to report: which single patch produces the largest embedding change?
#
# ### Stretch
# - Compare saliency under **two different captions** — encode both `"a photo of a cat"` and
#   `"a photo of a dog"` as queries, and define saliency relative to the *contrast* in cosine
#   similarity to each caption. This is a poor man's "Grad-CAM for CLIP."
# - Run the same occlusion experiment on a **chart with text labels** — does CLIP get *most*
#   sensitive to the text regions? That would be evidence CLIP is reading the text, not the visualization.

# %% Task 5 — workspace
# write your code here


# %% [markdown]
# ### When you're done
#
# Re-rank yourself on these 5 axes by how confident you felt:
# 1. **HF mechanics** (Processor / Model / `from_pretrained`)
# 2. **Model internals** (`output_hidden_states`, named sub-modules, parameter introspection)
# 3. **Matplotlib analytics** (heatmaps, overlays, side-by-side panels)
# 4. **Torch ops** (slicing, masking, batched matmul, `F.cosine_similarity`)
# 5. **Statistical reasoning** (when to mask diagonals, what means/stds tell you)
#
# Whichever scored lowest is what to drill on next.
