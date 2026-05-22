# %% [markdown]
# # SUPPLEMENT: The Two Spaces — In ML / AI Land
#
# (companion to `00_supplement_two_spaces.py`. Same duality, but now every
#  example is from ML: weights, neurons, attention, LoRA, CLIP.)
#
# ## What you should walk away knowing
#
# > **Rows of a weight matrix are neurons (probes living in input space).**
# > **Columns of a weight matrix are feature contributions (vectors in output space).**
# > **Same weight entry W[i, j] does double duty — both readings are
# >  the same number answering the same question two ways.**
#
# We'll see this in:
#  1. A tiny **linear classifier** (cat vs dog from physical features)
#  2. **Attention** — `Q @ K.T` as a grid of probe·probe alignments
#  3. **LoRA** — the duality written out explicitly: input probes × output landings
#  4. **CLIP / shared embedding space** — both rows of image_emb and text_emb live
#     in the SAME space and dot-product against each other

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

np.set_printoptions(precision=2, suppress=True)
np.random.seed(7)

# Helpers
def setup_ax_2d(ax, lim=5.0, title="", xlabel=None, ylabel=None):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)

def draw_arrow_2d(ax, vec, origin=(0, 0), color="black", label=None,
                  lw=2, label_offset=(0.15, 0.15), label_size=10):
    ax.annotate("", xy=(origin[0] + vec[0], origin[1] + vec[1]),
                xytext=origin,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label:
        tx = origin[0] + vec[0] + label_offset[0]
        ty = origin[1] + vec[1] + label_offset[1]
        ax.text(tx, ty, label, color=color, fontsize=label_size, fontweight="bold")

def draw_arrow_3d(ax, vec, origin=(0, 0, 0), color="black", label=None, lw=2):
    ax.quiver(origin[0], origin[1], origin[2],
              vec[0], vec[1], vec[2],
              color=color, lw=lw, arrow_length_ratio=0.12)
    if label:
        ax.text(origin[0] + vec[0] * 1.05,
                origin[1] + vec[1] * 1.05,
                origin[2] + vec[2] * 1.05,
                label, color=color, fontsize=9, fontweight="bold")

# %% [markdown]
# ## 1. The tiniest meaningful neural net — a linear classifier
#
# Imagine you want to classify an animal as **cat** or **dog** from three
# physical measurements:
#
# ```
#   input features (R³, our input space):
#     x[0] = whisker_length     (cats have long whiskers)
#     x[1] = bark_volume        (dogs are loud)
#     x[2] = fur_length         (both have fur)
#
#   output scores (R², our output space):
#     y[0] = cat_score
#     y[1] = dog_score
# ```
#
# A simple linear layer `y = W @ x` does the job with `W` shape `(2, 3)`:
#
# ```
#       whisker  bark    fur
#       _length  _vol    _length
#   W = [[ +2.0, -1.0,  +0.5 ],   ← row 0 = "CAT detector" (a probe in feature space)
#        [ -1.0, +2.0,  +1.0 ]]   ← row 1 = "DOG detector" (a probe in feature space)
#
#       ↑       ↑       ↑
#    col 0    col 1   col 2   ← each column = "what this feature contributes
#                                              to (cat_score, dog_score)"
# ```

# %%
W = np.array([[+2.0, -1.0, +0.5],     # cat detector
              [-1.0, +2.0, +1.0]])    # dog detector

# A sample input: long whiskers, quiet, medium fur — clearly a cat-ish profile
x = np.array([3.0, 1.0, 2.0])

y = W @ x
print("W =\n", W)
print("\nx (input features) =", x,
      "   ← whisker_length=3, bark_volume=1, fur_length=2")
print("\ny = W @ x =", y,
      "   ← cat_score, dog_score")
print(f"\nPrediction: {'CAT' if y[0] > y[1] else 'DOG'}  "
      f"(cat={y[0]:.1f} vs dog={y[1]:.1f})")

# Verify both readings give the same y
row_reading = np.array([W[0] @ x, W[1] @ x])
col_reading = x[0] * W[:, 0] + x[1] * W[:, 1] + x[2] * W[:, 2]
print("\nRow reading (each output = dot of probe row · input):", row_reading)
print("Col reading (output = sum of feature contributions):    ", col_reading)
# Same answer. Two stories.

# %% [markdown]
# ## 2. Rows of W ARE neurons — visualizing them in INPUT space
#
# Each row of `W` is **literally a neuron** — a single direction in the
# 3D feature space. When an input vector is closely aligned with the
# "cat-detector row," the cat neuron fires hard. That's how the network
# "looks for" cat-ness.
#
# **Row 0 = `[+2.0, -1.0, +0.5]`** says:
#   - +2 weight on whisker_length → "I love long whiskers"
#   - −1 weight on bark_volume    → "loudness hurts my score"
#   - +0.5 weight on fur_length   → "fur is mildly positive"
#
# So the cat-detector is a direction that POINTS toward "high whisker,
# low bark, medium fur." Inputs that resemble that direction fire it.
#
# Let's draw both detector rows AND the input vector in 3D feature space.

# %%
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 4); ax.set_ylim(-2, 4); ax.set_zlim(-1, 4)
ax.set_xlabel("whisker_length", fontsize=10)
ax.set_ylabel("bark_volume", fontsize=10)
ax.set_zlabel("fur_length", fontsize=10)
ax.set_title("INPUT SPACE (R³, the feature world)\n"
             "Rows of W are neurons living HERE as probes",
             fontsize=12)

# The two probe rows
draw_arrow_3d(ax, W[0], color="crimson",
              label=f"row 0 (CAT detector)\n= {W[0]}", lw=2.5)
draw_arrow_3d(ax, W[1], color="navy",
              label=f"row 1 (DOG detector)\n= {W[1]}", lw=2.5)

# The input vector
draw_arrow_3d(ax, x, color="black",
              label=f"x = {x}\n(input animal)", lw=3)

# Annotations
ax.text2D(0.02, 0.95,
          f"cat_score = dot(row 0, x) = dot({W[0]}, {x}) = {W[0] @ x:.1f}",
          transform=ax.transAxes, fontsize=10, color="crimson",
          fontweight="bold")
ax.text2D(0.02, 0.91,
          f"dog_score = dot(row 1, x) = dot({W[1]}, {x}) = {W[1] @ x:.1f}",
          transform=ax.transAxes, fontsize=10, color="navy",
          fontweight="bold")
ax.text2D(0.02, 0.02,
          "The input aligns more with the CAT row than the DOG row,\n"
          "so its dot product with row 0 is bigger → predicts cat.",
          transform=ax.transAxes, fontsize=10, style="italic")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Key takeaway**: a neuron is *just a direction in input space*. The
# whole "what does this neuron fire on?" interpretability question is
# just asking *"in which direction does this row point?"*

# %% [markdown]
# ## 3. Columns of W are feature contributions — visualizing them in OUTPUT space
#
# Now flip the lens. Each **column** of `W` says *"if I nudge this one
# input feature, how does the output (cat_score, dog_score) change?"*
#
# ```
#   col 0 = [+2.0, -1.0]  ← "1 unit of whisker_length adds +2 to cat, −1 to dog"
#   col 1 = [−1.0, +2.0]  ← "1 unit of bark_volume   adds −1 to cat, +2 to dog"
#   col 2 = [+0.5, +1.0]  ← "1 unit of fur_length    adds +0.5 to cat, +1 to dog"
# ```
#
# These are arrows you can draw in the **2D output space** (cat_score, dog_score).

# %%
fig, ax = plt.subplots(figsize=(9, 7))
setup_ax_2d(ax, lim=8, title="OUTPUT SPACE (R², the score world)\n"
                              "Columns of W are feature contributions HERE",
            xlabel="cat_score", ylabel="dog_score")

# The three column arrows from origin
draw_arrow_2d(ax, W[:, 0], color="tab:purple",
              label=f"col 0 = {W[:,0]}\n(whisker contribution per unit)",
              label_offset=(0.15, -0.5))
draw_arrow_2d(ax, W[:, 1], color="tab:olive",
              label=f"col 1 = {W[:,1]}\n(bark contribution per unit)",
              label_offset=(-3.5, 0.1))
draw_arrow_2d(ax, W[:, 2], color="tab:cyan",
              label=f"col 2 = {W[:,2]}\n(fur contribution per unit)",
              label_offset=(0.15, -0.3))

# The full output vector
draw_arrow_2d(ax, y, color="black", lw=3,
              label=f"y = W·x = {y}\n(total output)",
              label_offset=(0.15, 0.15))

ax.text(0, -7.5,
        f"y = x[0]·col 0 + x[1]·col 1 + x[2]·col 2\n"
        f"  = 3·{W[:,0]} + 1·{W[:,1]} + 2·{W[:,2]}\n"
        f"  = {x[0]*W[:,0]} + {x[1]*W[:,1]} + {x[2]*W[:,2]} = {y}",
        ha="center", fontsize=10, family="monospace", color="darkblue")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Notice:** column 0 has a POSITIVE cat_score component but a NEGATIVE
# dog_score component. That makes biological sense — whiskers favor cat,
# hurt dog. Column 1 (bark) does the opposite. Column 2 (fur) helps both
# a bit. The columns *encode each feature's policy* in output space.
#
# **This is exactly the "sensitivity" reading.** Want to know what
# happens if you increase whisker_length by 1? Add column 0 to your
# current output. That's it. The columns ARE the sensitivities.

# %% [markdown]
# ## 4. The double duty of every weight (the ML version of the climax)
#
# Pick any single weight, say `W[0, 1] = -1.0` (the cat-detector's
# coefficient on bark_volume). That number does **two jobs at once**:
#
# ```
#   NEURON reading (row 0, position 1):
#     "The cat detector has weight −1.0 on bark_volume.
#      Loud animals get penalized by the cat neuron."
#
#   SENSITIVITY reading (column 1, position 0):
#     "Increasing bark_volume by 1 decreases cat_score by 1."
# ```
#
# **Same number. Two questions. Same answer.** This is the
# matrix-duality, but stated in ML language.

# %%
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

# Draw the matrix as a grid with one cell highlighted
cell_w, cell_h = 1.0, 0.8
origin_x, origin_y = 3.5, 3.0
labels_col = ["whisker", "bark", "fur"]
labels_row = ["cat", "dog"]

for i in range(2):
    for j in range(3):
        color = "yellow" if (i == 0 and j == 1) else "lavender"
        ax.add_patch(Rectangle((origin_x + j * cell_w, origin_y - i * cell_h),
                               cell_w, cell_h,
                               facecolor=color, edgecolor="black", lw=1.5))
        ax.text(origin_x + j * cell_w + cell_w / 2,
                origin_y - i * cell_h + cell_h / 2,
                f"{W[i, j]:+.1f}",
                ha="center", va="center", fontsize=14, fontweight="bold")
# Column labels
for j, name in enumerate(labels_col):
    ax.text(origin_x + j * cell_w + cell_w / 2, origin_y + cell_h + 0.15,
            name, ha="center", fontsize=10, color="darkblue")
# Row labels
for i, name in enumerate(labels_row):
    ax.text(origin_x - 0.2, origin_y - i * cell_h + cell_h / 2,
            name, ha="right", va="center", fontsize=10,
            color="darkred", fontweight="bold")

ax.text(origin_x + 1.5 * cell_w, origin_y + 1.4,
        "W (weights)", ha="center", fontsize=13, fontweight="bold")

# Annotations from both sides
highlighted_x = origin_x + 1 * cell_w + cell_w / 2
highlighted_y = origin_y - 0 * cell_h + cell_h / 2

ax.annotate(
    "NEURON reading (row 0):\n"
    "  Row 0 = CAT detector = [+2.0, −1.0, +0.5].\n"
    "  The cat neuron weights bark_volume by −1.\n"
    "  → 'Cat neuron hates loud animals.'",
    xy=(highlighted_x, highlighted_y),
    xytext=(8, 5),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
    color="crimson", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="crimson"))

ax.annotate(
    "SENSITIVITY reading (column 1):\n"
    "  Col 1 = bark contribution = [−1.0, +2.0].\n"
    "  Adding 1 to bark_volume changes\n"
    "  cat_score by −1 and dog_score by +2.\n"
    "  → 'Louder dog, less catty.'",
    xy=(highlighted_x, highlighted_y),
    xytext=(8, 1),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5),
    color="tab:blue", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="aliceblue", edgecolor="tab:blue"))

ax.annotate(
    "W[0, 1] = −1.0",
    xy=(highlighted_x, highlighted_y),
    xytext=(1.0, 0.7), fontsize=12, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    color="black", ha="center")

ax.text(6, 0.1,
        "ONE weight number. TWO ML questions. SAME answer.\n"
        "This is why 'understand a neuron's row' = 'understand a feature's sensitivity column' — same data, two lenses.",
        ha="center", fontsize=11, fontweight="bold", color="darkgreen")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Attention — the killer example of the dot product view
#
# In a transformer block, the most important matrix multiplication is:
#
# ```
#   scores = Q @ K.T        # shape (T, T)
# ```
#
# where `Q` (queries) and `K` (keys) both have shape `(T, d)` — `T` tokens,
# each represented as a `d`-dimensional vector in some shared embedding space.
#
# - **Each row of `Q` is a query** — a probe in embedding space.
#   It says *"I'm token i, looking for tokens that match this direction."*
# - **Each row of `K` is a key** — also a probe in the same embedding
#   space. It says *"I'm token j, presenting this direction to be matched."*
# - **Each entry `scores[i, j] = dot(q_i, k_j)`** — the alignment of
#   token i's query with token j's key.
#
# The whole attention matrix is **a grid of dot products** between probes
# living in the same embedding space. That's *exactly* the dual-view
# pattern, but now with ML semantics.

# %%
# Toy attention example: 4 tokens, 2D embeddings (for visualization)
T, d = 4, 2

# Queries: 4 tokens each looking for something
Q = np.array([
    [ 1.5,  0.5],   # token 0's query
    [ 0.0,  1.5],   # token 1's query
    [-1.0,  1.0],   # token 2's query
    [ 1.0, -1.0],   # token 3's query
])

# Keys: 4 tokens each offering something
K = np.array([
    [ 1.0,  0.0],   # token 0's key
    [ 0.5,  1.0],   # token 1's key
    [-1.0,  0.5],   # token 2's key
    [ 0.8, -0.8],   # token 3's key
])

# The attention scores grid
scores = Q @ K.T
print("Q (4 queries, each a 2D direction):\n", Q)
print("\nK (4 keys, each a 2D direction):\n", K)
print("\nQ @ K.T (alignment grid):\n", scores)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# LEFT: queries and keys as arrows in the SAME embedding space
ax = axes[0]
setup_ax_2d(ax, lim=2.5, title="EMBEDDING SPACE — queries and keys live HERE\n"
                                "Both are probes in the same R^d")
for i in range(T):
    draw_arrow_2d(ax, Q[i], color=f"C{i}", lw=2.5,
                  label=f"q_{i}", label_offset=(0.05, 0.05), label_size=11)
    draw_arrow_2d(ax, K[i], color=f"C{i}", lw=1.5,
                  label=f"k_{i}", label_offset=(0.05, -0.2), label_size=11)
ax.text(0, -2.3,
        "Solid arrows = queries (q_i).  Thinner arrows = keys (k_i).\n"
        "Both are arrows in the SAME embedding space.",
        ha="center", fontsize=10, style="italic")

# RIGHT: attention score heatmap with one entry highlighted
ax = axes[1]
ax.imshow(scores, cmap="RdBu_r", aspect="equal", vmin=-3, vmax=3)
for i in range(T):
    for j in range(T):
        ax.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(scores[i, j]) > 1.5 else "black",
                fontsize=12, fontweight="bold")
# Highlight entry (0, 1) — query 0 against key 1
ax.add_patch(Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor="lime", lw=4))
ax.set_xticks(range(T)); ax.set_yticks(range(T))
ax.set_xticklabels([f"k_{j}" for j in range(T)])
ax.set_yticklabels([f"q_{i}" for i in range(T)])
ax.set_xlabel("KEY (column)")
ax.set_ylabel("QUERY (row)")
ax.set_title("Attention scores = Q @ K.T\n"
             f"Highlighted: scores[0, 1] = dot(q_0, k_1) = {scores[0, 1]:.2f}",
             fontsize=11)

plt.tight_layout()
plt.show()

print(f"\nVerifying scores[0, 1] = dot(q_0, k_1) = dot({Q[0]}, {K[1]}) = {Q[0] @ K[1]:.2f}")
print("That single number lives at the intersection of row 0 (query 0) and column 1 (key 1).")
print("\nDuality reading:")
print("  Row reading:    'how much does k_1 contribute to q_0's attention pattern?'")
print("  Column reading: 'how much attention does q_0 give to k_1?'")
print("  → SAME answer. The Q @ K.T grid is the duality made operational.")

# %% [markdown]
# **Why this matters for transformers:** after `softmax(scores)`, each
# row of the result is a probability distribution that says "this is how
# query i splits its attention among the keys." The whole mechanism is a
# dot-product grid — every (query, key) pair gets an alignment score, in
# parallel, in one matmul. That's the entire reason attention is fast on GPUs.

# %% [markdown]
# ## 6. LoRA — the duality made EXPLICIT in the architecture
#
# LoRA (Low-Rank Adaptation) is fine-tuning with a deliberately decomposed
# weight update:
#
# ```
#   W_finetuned = W_frozen + ΔW
#
#   ΔW = B @ A
#
#   A.shape = (rank, in_dim)     ← rows of A live in INPUT space  (probes)
#   B.shape = (out_dim, rank)    ← cols of B live in OUTPUT space (landings)
# ```
#
# This is literally the "rows in input space, columns in output space"
# picture written into the architecture. `A` provides `rank` carefully
# chosen **input-space directions worth listening to**. `B` provides
# `rank` **output-space directions worth producing**. Their matmul
# weaves them into a full weight matrix that has rank at most `rank`.
#
# The whole point of LoRA: you compress the update by separating the
# "what input directions matter" question from the "what output
# directions matter" question, and learning each with `rank ≪ min(in, out)`
# parameters.

# %%
in_dim, out_dim, rank = 5, 3, 2     # tiny example
A = np.random.randn(rank, in_dim)   # rows = input-space probes
B = np.random.randn(out_dim, rank)  # cols = output-space landings
delta_W = B @ A                     # shape (out_dim, in_dim)

print(f"A shape: {A.shape}   ← {rank} rows, each a probe in {in_dim}-dim input space")
print(f"B shape: {B.shape}   ← {rank} columns, each a landing in {out_dim}-dim output space")
print(f"\nΔW = B @ A, shape: {delta_W.shape}")
print(f"Rank of ΔW: {np.linalg.matrix_rank(delta_W)}   (at most {rank}, by construction)")
print(f"\nTotal LoRA parameters: A entries + B entries = "
      f"{A.size} + {B.size} = {A.size + B.size}")
print(f"Equivalent full-rank ΔW parameters: {delta_W.size}")
print(f"Compression ratio: {delta_W.size / (A.size + B.size):.2f}×")

# Visualize the decomposition
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, mat, name, cmap in zip(
        axes, [B, A, delta_W],
        [f"B  ({out_dim}×{rank})\ncols = output landings",
         f"A  ({rank}×{in_dim})\nrows = input probes",
         f"ΔW = B @ A  ({out_dim}×{in_dim})\nfull update, low rank"],
        ["Blues", "Reds", "Purples"]):
    ax.imshow(mat, cmap=cmap, aspect="equal")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                    color="white" if abs(mat[i, j]) > 1.0 else "black",
                    fontsize=10)
    ax.set_title(name, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
plt.show()

# %% [markdown]
# **The mental model**: LoRA explicitly separates *"which `r` input
# directions should I listen to?"* (matrix `A`) from *"which `r` output
# directions should I write into?"* (matrix `B`). The full update is
# their outer-product weave. The architecture *forces* you to think in
# the two-spaces lens.

# %% [markdown]
# ## 7. CLIP — when two encoders dump into the SAME embedding space
#
# CLIP has two encoders:
#
# ```
#   image_encoder(image) → image_embedding ∈ R^d
#   text_encoder(text)   → text_embedding  ∈ R^d
# ```
#
# Both outputs live in the **same** d-dimensional embedding space. That's
# the whole point — alignment requires they live in the same world.
#
# At training time you compute a similarity matrix between every image
# and every text in the batch:
#
# ```
#   similarities = image_embeddings @ text_embeddings.T
#
#   image_embeddings   shape (B, d)   ← each row = one image as a probe
#   text_embeddings.T  shape (d, B)   ← each column = one text as a landing
#   similarities       shape (B, B)   ← grid of dot products
# ```
#
# Diagonal entries (i, i) = matching pair → should be high.
# Off-diagonal entries (i, j) for i≠j = mismatched → should be low.
# Train so the rows of the similarity matrix look like one-hot
# distributions concentrated on the diagonal.

# %%
B_size, d = 4, 3
image_emb = np.array([        # 4 images, each a 3D embedding (toy)
    [+0.9, +0.1, +0.1],       # photo of a cat
    [+0.1, +0.9, +0.1],       # photo of a dog
    [+0.1, +0.1, +0.9],       # photo of a bird
    [+0.6, +0.6, +0.2],       # photo of cat+dog together
])
text_emb = np.array([
    [+0.8, +0.2, +0.1],       # caption: "a cat"
    [+0.2, +0.8, +0.1],       # caption: "a dog"
    [+0.1, +0.1, +0.9],       # caption: "a bird"
    [+0.5, +0.5, +0.2],       # caption: "a cat and a dog"
])

sim = image_emb @ text_emb.T

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
for i in range(B_size):
    for j in range(B_size):
        ax.text(j, i, f"{sim[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(sim[i, j]) > 0.5 else "black",
                fontsize=11, fontweight="bold")
# Highlight the diagonal (correct pairs)
for i in range(B_size):
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                           edgecolor="lime", lw=3))
ax.set_xticks(range(B_size)); ax.set_yticks(range(B_size))
ax.set_xticklabels(["'cat'", "'dog'", "'bird'", "'cat+dog'"])
ax.set_yticklabels(["img:cat", "img:dog", "img:bird", "img:cat+dog"])
ax.set_xlabel("TEXT embeddings (cols of T)")
ax.set_ylabel("IMAGE embeddings (rows of I)")
ax.set_title("CLIP similarity: I @ T.T\n"
             "Each cell = dot(image_i, text_j) in the shared embedding space",
             fontsize=11)
plt.colorbar(im, ax=ax, label="cosine-like similarity")
plt.tight_layout()
plt.show()

print("\nReading the matrix two ways:")
print("  ROW reading:    'image i's similarity profile across all texts'")
print("  COLUMN reading: 'text j's similarity profile across all images'")
print("  Diagonal entries should be largest → matching pair.")
print("\nThis is literally the same dot-product duality as attention,")
print("but here the two sets of probes (images vs texts) come from")
print("DIFFERENT encoders that learned to agree on a shared space.")

# %% [markdown]
# ## 8. The mental flips you should now be able to do
#
# When you see ANY weight matrix in a paper or codebase, ask both
# questions:
#
# | When you see... | Row reading (what does each output care about?) | Column reading (what does each input do?) |
# |---|---|---|
# | `nn.Linear(in, out)` weight `W` shape `(out, in)` | Each row = one neuron's preferred input direction | Each column = how nudging one input feature affects every output |
# | Attention `Q` of shape `(T, d)` | Each row = one token's query (what it's looking for) | Each column = a single embedding axis, summed over tokens |
# | Attention `K` of shape `(T, d)` | Each row = one token's key (what it offers) | Each column = a single embedding axis, summed over tokens |
# | `Q @ K.T` shape `(T, T)` | Row i = query i's alignment with every key | Column j = how key j gets attended to by every query |
# | LoRA `A` shape `(r, in)` | Each row = one low-rank input direction worth listening to | (less useful for A) |
# | LoRA `B` shape `(out, r)` | (less useful for B) | Each column = one low-rank output direction worth writing to |
# | CLIP `I` shape `(B, d)` | Each row = one image embedding | Each column = one embedding axis, image-wise |
# | Embedding table `E` shape `(vocab, d)` | Each row = one word's vector in embedding space | Each column = "the meaning of axis k" across all words |
#
# **Practice**: pick a weight matrix from a paper and write down both
# readings before you continue. Two sentences per matrix. Do this 20
# times and it becomes automatic.

# %% [markdown]
# ## 9. Interpretability: where the duality earns its keep
#
# Modern interpretability research lives or dies on this lens:
#
# - **"What does neuron k of layer L detect?"** → read **row k** of that
#   layer's weight matrix. Inspect the highest-magnitude input dimensions
#   in that row — those are the input features the neuron cares about
#   most. (For deep nets, you visualize what input maximizes this row's
#   dot product with the input — that's "feature visualization.")
#
# - **"What does input feature j control?"** → read **column j** of the
#   weight matrix. Find the largest-magnitude entries — those are the
#   neurons most sensitive to that input feature.
#
# - **"Which neurons fire together?"** → look for similar ROWS (cosine
#   similarity between rows of W). Neurons with similar preferred
#   directions detect similar things.
#
# - **"Which inputs are interchangeable?"** → look for similar COLUMNS.
#   Inputs with similar effect profiles across outputs are redundant.

# %% [markdown]
# ## The unforgettable ML summary
#
# ```
#   ┌────────────────────────────────────────────────────────────────────┐
#   │  A weight matrix W of shape (out_dim, in_dim) is ONE map with TWO  │
#   │  homes for its arrows:                                             │
#   │                                                                    │
#   │     ROWS (out_dim of them)        live in INPUT space              │
#   │       = neurons / detectors / probes                               │
#   │       = directions in the feature world                            │
#   │       = what each output "looks for"                               │
#   │                                                                    │
#   │     COLUMNS (in_dim of them)       live in OUTPUT space            │
#   │       = sensitivities / contributions / landings                   │
#   │       = directions in the score world                              │
#   │       = how each input feature "writes into" outputs               │
#   │                                                                    │
#   │     ENTRY W[i, j] does double duty:                                │
#   │       row i, position j  =  what input direction j does neuron i  │
#   │                              listen to?                            │
#   │       column j, position i = how does input feature j affect       │
#   │                              output neuron i?                      │
#   │       → SAME number, same answer to two questions.                 │
#   └────────────────────────────────────────────────────────────────────┘
# ```
#
# ### Hey-dude one-liner
#
# > Hey dude, in ML, **rows of weights ARE neurons** (probes in the input
# > embedding space), and **columns of weights ARE feature contributions**
# > (sensitivities in the output space). Attention is the duality in
# > action: `Q @ K.T` is a grid of probe·probe alignments. LoRA literally
# > separates input probes (A) from output landings (B). CLIP makes two
# > encoders dump into one shared space so their probes can dot-product
# > against each other. Once you see the two-spaces picture, transformer
# > papers stop being magic.
#
# **ML payoff:** every weight matrix becomes readable as either a stack
# of neurons (row lens) or a stack of feature contributions (column lens).
# Pick the lens that matches the question you're asking. Transpose to
# swap them. That single mental move makes most ML internals legible.

# %%
