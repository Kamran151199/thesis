# %% [markdown]
# # SUPPLEMENT: Matrix Multiplication has TWO Faces
#
# ## The confusion that needs jolting
#
# You already know one story about matrix multiplication:
#
# > **Story A (transformation view, your default):**
# > "A matrix is a transformation. Its columns tell you where the basis
# >  vectors land. Matrix × matrix = compose two transformations."
#
# But people keep saying another thing:
#
# > **Story B (dot product view):**
# > "Matrix × matrix is a grid of dot products. Each output entry is a
# >  row of the left matrix · a column of the right matrix."
#
# These sound like two different operations. THEY ARE NOT. They are the
# SAME arithmetic, read with two different mental highlights. This file
# is the bridge — by the end you should see both faces of the same number.
#
# We'll go:
#   1. The arithmetic, brutally explicit — 2×3 @ 3×2 with every step shown
#   2. Read it as "where do basis vectors land" (your view)
#   3. Read it as "what does each row probe in each column" (the dual view)
#   4. The proof they're identical (same formula, just regrouped)
#   5. The visual aha — one entry as a row-column dot product
#   6. Why neural nets read it the second way (rows = neurons)

# %%
import numpy as np
import matplotlib.pyplot as plt

# Helpers from your usual conventions
def setup_ax_2d(ax, lim=4.0, title=""):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)

def draw_arrow_2d(ax, vec, color="black", label=None, lw=2):
    ax.annotate("", xy=tuple(vec), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label:
        ax.text(vec[0] * 1.1, vec[1] * 1.1, label, color=color, fontsize=11)

np.set_printoptions(precision=2, suppress=True)

# %% [markdown]
# ## 1. The arithmetic — naked and explicit
#
# Let's pick two matrices and compute `A @ B` by hand. Small numbers, no magic.

# %%
A = np.array([[1, 2, 3],
              [4, 5, 6]])      # shape (2, 3)

B = np.array([[7,  8],
              [9,  10],
              [11, 12]])       # shape (3, 2)

# Expected shape: (2, 3) @ (3, 2) → (2, 2)
C = A @ B
print("A @ B =")
print(C)

# %% [markdown]
# Let's compute every single entry of `C` BY HAND.
#
# The formula (whatever interpretation you like) is:
#
# ```
#  C[i, j]  =  Σ  A[i, k] · B[k, j]
#             k
# ```
#
# So entry by entry:

# %%
for i in range(2):
    for j in range(2):
        terms = [f"{A[i, k]}·{B[k, j]}" for k in range(3)]
        value = sum(A[i, k] * B[k, j] for k in range(3))
        print(f"C[{i},{j}] = {' + '.join(terms)}  =  {value}")

# %% [markdown]
# Notice the formula has TWO indices that are FREE (`i`, `j`) and ONE that
# is SUMMED (`k`). That's the entire mystery: depending on which one you
# emphasize, you get a different reading of the same arithmetic.

# %% [markdown]
# ## 2. Story A — "Where do basis vectors land?" (your default)
#
# In this view, we **fix one column of `B` at a time** and watch what `A`
# does to it. Recall: a matrix-times-vector means "linear combination of
# the matrix's columns weighted by the vector's entries."
#
# `B`'s columns are:
#
# ```
#   col_0(B) = [7, 9, 11]ᵀ
#   col_1(B) = [8, 10, 12]ᵀ
# ```
#
# So:
#
# ```
#   A @ col_0(B) = 7 · A[:,0] + 9 · A[:,1] + 11 · A[:,2]
#                = 7 · [1, 4] + 9 · [2, 5] + 11 · [3, 6]
#                = [7 + 18 + 33,  28 + 45 + 66]
#                = [58,  139]    ← this is column 0 of C
#
#   A @ col_1(B) = 8 · A[:,0] + 10 · A[:,1] + 12 · A[:,2]
#                = 8 · [1, 4] + 10 · [2, 5] + 12 · [3, 6]
#                = [8 + 20 + 36,  32 + 50 + 72]
#                = [64,  154]    ← this is column 1 of C
# ```

# %%
print("Manual columns of C using transformation view:")
print("col 0:", 7 * A[:, 0] + 9 * A[:, 1] + 11 * A[:, 2])
print("col 1:", 8 * A[:, 0] + 10 * A[:, 1] + 12 * A[:, 2])

print("\nC again, for comparison:")
print(C)
# BOOM — same numbers.

# %% [markdown]
# **Reading**: each column of `C` is "where the corresponding column of `B`
# landed after `A` transformed it." That's the transformation story.

# %% [markdown]
# ## 3. Story B — "What does each row probe in each column?"
#
# Now let's emphasize the **rows** of `A` instead. The same formula:
#
# ```
#   C[i, j]  =  Σ  A[i, k] · B[k, j]
#             k
# ```
#
# can be read as: pick row `i` of `A`, pick column `j` of `B`, multiply
# element-wise, and sum. **THAT IS THE DOT PRODUCT** of those two vectors.

# %%
# Row 0 of A:
row_0_A = A[0, :]      # [1, 2, 3]
row_1_A = A[1, :]      # [4, 5, 6]

# Columns of B:
col_0_B = B[:, 0]      # [7, 9, 11]
col_1_B = B[:, 1]      # [8, 10, 12]

print("Entries of C, computed as dot products:")
print(f"C[0, 0] = dot({row_0_A}, {col_0_B}) = {row_0_A @ col_0_B}")
print(f"C[0, 1] = dot({row_0_A}, {col_1_B}) = {row_0_A @ col_1_B}")
print(f"C[1, 0] = dot({row_1_A}, {col_0_B}) = {row_1_A @ col_0_B}")
print(f"C[1, 1] = dot({row_1_A}, {col_1_B}) = {row_1_A @ col_1_B}")

# BOOM — every entry of C is one dot product of a row · column pair.

# %% [markdown]
# **Reading**: each entry `C[i, j]` measures *"how aligned is row i of A
# with column j of B?"* — and the whole matrix multiplication is a `2×2`
# grid of those alignment numbers.

# %% [markdown]
# ## 4. The bridge — why both stories give the same numbers
#
# Here's the part that needs to click. Let's stare at one entry, say `C[0, 0]`.
#
# **Story A says:**
#
# ```
#   col_0(C) = 7·A[:,0] + 9·A[:,1] + 11·A[:,2]      ← linear combination
#                                                     of A's columns
#
#   So C[0, 0]  = 7·A[0,0] + 9·A[0,1] + 11·A[0,2]
#               = 7·1 + 9·2 + 11·3
#               = 58
# ```
#
# **Story B says:**
#
# ```
#   C[0, 0] = dot( row_0(A) , col_0(B) )
#           = dot( [1, 2, 3] , [7, 9, 11] )
#           = 1·7 + 2·9 + 3·11
#           = 58
# ```
#
# Compare the two sums:
#
# ```
#   Story A:   7·1  +  9·2  +  11·3
#   Story B:   1·7  +  2·9  +  3·11
# ```
#
# **SAME NUMBERS. Just multiplied in the opposite order.** Multiplication
# is commutative, so these are bit-for-bit identical.
#
# The trick: the formula `Σₖ A[i,k]·B[k,j]` doesn't care if you read it
# "column-by-column of A weighted by entries of B" or
# "row-of-A dotted with column-of-B". It's the same sum.

# %% [markdown]
# ### One more way to see it: the four readings of the same formula
#
# Stare at this and don't read past until the symmetry hits:
#
# ```
#                    ┌── free index → picks which row of C
#                    │   ┌── free index → picks which column of C
#                    │   │
#   C[i, j]  =   Σ   A[i, k] · B[k, j]
#                k
#                ↑
#                summed-out index ("contracted")
# ```
#
# - **Fix j first:** "column j of C is A applied to column j of B" → transformation story
# - **Fix i first:** "row i of C is row i of A applied to all columns of B" → row-as-probe story
# - **Fix (i, j):** "entry [i, j] of C is row i of A · column j of B" → dot product story
# - **Sum over k first:** "C is a sum over k of outer products A[:,k] ⊗ B[k,:]" → outer-product story
#
# All four are literally the same formula. None is more "fundamental" —
# you choose the reading that helps you think about the problem at hand.

# %% [markdown]
# ## 5. The visual aha — see the dot product LIVE inside the matmul

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# --- Left panel: matrix A with row 0 highlighted ---
ax = axes[0]
ax.set_title("A  (2×3) — row 0 highlighted", fontsize=11)
ax.imshow(A, cmap="Blues", aspect="equal", vmin=0, vmax=15)
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        ax.text(j, i, str(A[i, j]), ha="center", va="center",
                color="white" if A[i, j] > 8 else "black", fontsize=14, fontweight="bold")
ax.add_patch(plt.Rectangle((-0.5, -0.5), 3, 1, fill=False, edgecolor="red", lw=3))
ax.text(1, -1, "row 0 of A = [1, 2, 3]", color="red", ha="center", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# --- Middle panel: matrix B with column 0 highlighted ---
ax = axes[1]
ax.set_title("B  (3×2) — column 0 highlighted", fontsize=11)
ax.imshow(B, cmap="Greens", aspect="equal", vmin=0, vmax=15)
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        ax.text(j, i, str(B[i, j]), ha="center", va="center",
                color="white" if B[i, j] > 8 else "black", fontsize=14, fontweight="bold")
ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 3, fill=False, edgecolor="red", lw=3))
ax.text(0, 3.2, "col 0 of B = [7, 9, 11]", color="red", ha="center", fontsize=11)
ax.set_xticks([]); ax.set_yticks([])

# --- Right panel: C with entry [0, 0] highlighted ---
ax = axes[2]
ax.set_title("C = A @ B   (2×2) — entry [0,0] is the dot product", fontsize=11)
ax.imshow(C, cmap="Oranges", aspect="equal", vmin=0, vmax=160)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        ax.text(j, i, str(C[i, j]), ha="center", va="center",
                color="white" if C[i, j] > 80 else "black", fontsize=14, fontweight="bold")
ax.add_patch(plt.Rectangle((-0.5, -0.5), 1, 1, fill=False, edgecolor="red", lw=3))
ax.text(0.5, 2.0,
        "C[0,0] = 1·7 + 2·9 + 3·11 = 58\n"
        "       = dot(row 0 of A, col 0 of B)",
        color="red", ha="center", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.show()

# %% [markdown]
# Stare at that picture. **One entry of `C` is built by reaching across
# the red row of `A`, reaching down the red column of `B`, and summing
# their elementwise products.** Do that for all 4 (row, col) pairs and
# you've built the entire result matrix.

# %% [markdown]
# ## 6. Same matmul, visualized as transformation (your favorite view)
#
# To prove these are the same operation, let's also picture it as
# transformation. Pick a smaller square example so we can draw it in 2D.

# %%
A2 = np.array([[1.0, 0.5],
               [0.5, 1.0]])     # 2×2 — a mild shear-ish transformation

B2 = np.array([[2.0, 0.0],
               [0.0, 3.0]])     # 2×2 — a stretch (x*2, y*3)

C2 = A2 @ B2
print("A2 =\n", A2)
print("B2 =\n", B2)
print("C2 = A2 @ B2 =\n", C2)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: B2's columns — where i_hat and j_hat go under B2
ax = axes[0]
setup_ax_2d(ax, lim=4, title="Step 1: B2's columns\n= where î, ĵ land after B2")
draw_arrow_2d(ax, B2[:, 0], color="tab:blue",  label="B2·î = [2, 0]")
draw_arrow_2d(ax, B2[:, 1], color="tab:orange", label="B2·ĵ = [0, 3]")

# Panel 2: apply A2 to each of B2's columns — these become C2's columns
ax = axes[1]
setup_ax_2d(ax, lim=4, title="Step 2: A2 applied to each\ncolumn of B2 → columns of C2")
col0_C = A2 @ B2[:, 0]
col1_C = A2 @ B2[:, 1]
draw_arrow_2d(ax, col0_C, color="tab:blue",  label=f"A2·B2·î = {np.round(col0_C, 2)}")
draw_arrow_2d(ax, col1_C, color="tab:orange", label=f"A2·B2·ĵ = {np.round(col1_C, 2)}")

# Panel 3: show those landings are the columns of C2 — also computable as dot products
ax = axes[2]
setup_ax_2d(ax, lim=4, title="Step 3: C2's columns are\nALSO row·column dot products")
draw_arrow_2d(ax, C2[:, 0], color="tab:blue",  label=f"C2[:,0] = {np.round(C2[:, 0], 2)}")
draw_arrow_2d(ax, C2[:, 1], color="tab:orange", label=f"C2[:,1] = {np.round(C2[:, 1], 2)}")
ax.text(-3.5, -3.5,
        f"C2[0,0] = dot({A2[0]}, {B2[:,0]}) = {A2[0] @ B2[:,0]:.1f}\n"
        f"C2[1,0] = dot({A2[1]}, {B2[:,0]}) = {A2[1] @ B2[:,0]:.1f}\n"
        f"C2[0,1] = dot({A2[0]}, {B2[:,1]}) = {A2[0] @ B2[:,1]:.1f}\n"
        f"C2[1,1] = dot({A2[1]}, {B2[:,1]}) = {A2[1] @ B2[:,1]:.1f}",
        fontsize=9, family="monospace")

plt.tight_layout()
plt.show()

# %% [markdown]
# Same matrix `C2`, two ways to construct it:
# - **Left + middle panel** (your view): A2 transforms B2's columns into C2's columns.
# - **Right panel**: each entry of C2 is also a row·column dot product.
#
# The arrows in the middle and right panels are IDENTICAL. They have to be —
# they're literally the same numbers.

# %% [markdown]
# ## 7. Why neural networks talk about rows (the dot product story)
#
# In `nn.Linear(in=3, out=2)`, the weight matrix has shape `(2, 3)` —
# same shape as our `A` from section 1.
#
# When you compute `y = x @ W.T` for a batch of inputs `x` of shape `(B, 3)`:
#
# ```
#   x      shape (B, 3)
#   W.T    shape (3, 2)     ← W was (2, 3), so W.T is (3, 2)
#   y      shape (B, 2)
# ```
#
# Each entry `y[b, j]` = dot product of input sample `b` with column `j` of
# `W.T` = dot product with row `j` of `W`.
#
# So the **rows of `W` ARE the neurons.** Each row is a 3-dim direction
# that the layer probes the input against. The matmul is computing
# (batch size × num neurons) = B × 2 dot products in parallel, neatly
# arranged as a matrix.

# %%
B_batch = 4
x = np.random.randn(B_batch, 3)              # 4 input samples, 3 features each
W = np.random.randn(2, 3)                    # 2 neurons, each a 3-dim weight vector
y = x @ W.T                                  # shape (4, 2)

print("x =\n", x)
print("\nW (2 neurons, each a row of 3 weights) =\n", W)
print("\ny = x @ W.T =\n", y)

print("\nManually computed neuron-by-neuron dot products:")
for b in range(B_batch):
    for j in range(2):
        dp = x[b] @ W[j]
        print(f"  y[{b}, {j}] = dot(sample {b}, neuron {j}) = {dp:.3f}")

# Same numbers as in y above. Every entry is one dot product.

# %% [markdown]
# ## Summary — the two faces of the same operation
#
# | View | Reading | Best when you're thinking about |
# |---|---|---|
# | **Transformation** (columns) | C's column j = A applied to B's column j; "where do basis vectors land" | geometry, basis change, rank, span, projection, attention's geometric intuition |
# | **Dot product** (rows × columns) | C[i, j] = row i of A · column j of B; "alignment grid" | neurons, feature detectors, similarity scores, attention's `Q @ K.T`, interpretability |
# | **Outer product sum** | C = Σₖ A[:,k] ⊗ B[k,:]; "sum of rank-1 contributions" | LoRA, low-rank approximation, SVD |
#
# ### Hey-dude one-liner
#
# > Hey dude, `A @ B` is one operation with three legal readings. Your
# > "transformation of basis vectors" instinct is correct. The "grid of
# > dot products" claim is correct. They are LITERALLY the same arithmetic
# > — `Σₖ A[i,k]·B[k,j]` — just regrouped to emphasize different things.
#
# **ML payoff:** when reading a transformer paper, switch lenses on the
# fly. `Q @ K.T` is "every query dot-producted with every key" (alignment
# grid) — but it's also "K's transformation applied to Q's rows" — and
# also "a sum of rank-1 outer products." Same numbers, different stories.
# Pick the lens that fits the question you're asking.

# %%
