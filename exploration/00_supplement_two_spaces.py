# %% [markdown]
# # SUPPLEMENT: The Two Spaces a Matrix Lives In
#
# (or: why "transformation view" and "dot product view" describe the same
#  thing — the deepest unforgettable picture of matrix multiplication)
#
# ## The one-sentence picture you need to install
#
# > **Columns of a matrix live in the OUTPUT space.**
# > **Rows of a matrix live in the INPUT space.**
# > **The same matrix entry A[i, j] does double duty in both worlds.**
#
# If that sentence doesn't sing to you yet, this file fixes that. By the
# end you'll never be confused about the link between the two readings.
#
# ## Roadmap
#
#   1. The recipe analogy — get the link without any math
#   2. Set up a concrete 2×2 example
#   3. Columns LIVE in output space (visual)
#   4. Rows LIVE in input space (visual)
#   5. The grand unified picture — both readings, side by side
#   6. The "double duty" of every entry — the climax
#   7. The transpose flip — A.T swaps which space owns what
#   8. A 3D → 2D example so the "two spaces" become undeniable
#   9. The deep reason — duality, in plain English
#  10. ML payoff and unforgettable summary

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

np.set_printoptions(precision=2, suppress=True)

# ---- helpers --------------------------------------------------------------
def setup_ax_2d(ax, lim=5.0, title="", xlabel=None, ylabel=None):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.axhline(0, color="lightgray", lw=0.5)
    ax.axvline(0, color="lightgray", lw=0.5)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=11)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)

def draw_arrow_2d(ax, vec, origin=(0, 0), color="black", label=None,
                  lw=2, label_offset=(0.15, 0.15), label_size=11):
    ax.annotate("", xy=(origin[0] + vec[0], origin[1] + vec[1]),
                xytext=origin,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label:
        tx = origin[0] + vec[0] + label_offset[0]
        ty = origin[1] + vec[1] + label_offset[1]
        ax.text(tx, ty, label, color=color, fontsize=label_size,
                fontweight="bold")

def draw_arrow_3d(ax, vec, origin=(0, 0, 0), color="black", label=None, lw=2):
    ax.quiver(origin[0], origin[1], origin[2],
              vec[0], vec[1], vec[2],
              color=color, lw=lw, arrow_length_ratio=0.1)
    if label:
        ax.text(origin[0] + vec[0] * 1.1,
                origin[1] + vec[1] * 1.1,
                origin[2] + vec[2] * 1.1,
                label, color=color, fontsize=10, fontweight="bold")

# %% [markdown]
# ## 1. The recipe analogy — install the picture before any math
#
# Imagine a juice bar that turns INGREDIENTS into DRINKS. There are
# 3 ingredients (apple, banana, carrot) and 2 drinks (Drink A, Drink B).
# The recipe matrix tells you how the conversion works:
#
# ```
#                apple   banana   carrot
#   Drink A  →    0.5     0.3      0.2       ← row of A: "how to make Drink A"
#   Drink B  →    0.1     0.4      0.5       ← row of B: "how to make Drink B"
#                 ↑       ↑        ↑
#                 columns: "what each ingredient contributes
#                           across all drinks"
# ```
#
# Now look at one cell, say `recipe[Drink A, banana] = 0.3`. That single
# number is doing two jobs at once:
#
# ```
#   ROW reading:    "Drink A uses 0.3 cups of banana"
#   COLUMN reading: "banana contributes 0.3 to Drink A"
# ```
#
# Both readings are LITERALLY THE SAME FACT. The number doesn't change.
# What changes is which question you're asking:
#
# - Reading by row → "how do I make each drink?" (output-indexed)
# - Reading by column → "what does each ingredient go into?" (input-indexed)
#
# **The ingredients live in INGREDIENT-WORLD.**
# **The drinks live in DRINK-WORLD.**
# **The recipe matrix sits between them, translating one to the other.**
#
# Same idea, mathematical form:
#
# ```
#   ingredients = "input space"      ↔  V  (3 ingredients = 3 dimensions)
#   drinks      = "output space"     ↔  W  (2 drinks      = 2 dimensions)
#   recipe      = "matrix"           ↔  A  (shape: drinks × ingredients = 2 × 3)
# ```

# %% [markdown]
# ## 2. The concrete 2×2 example we'll keep using
#
# We'll use a small matrix and trace one input through it both ways.

# %%
A = np.array([[3, 2],
              [1, 5]], dtype=float)        # shape (2, 2) — NON-symmetric on purpose
v = np.array([1, 2], dtype=float)          # input vector

Av = A @ v
print("A =\n", A)
print("\nv =", v, "   (input vector)")
print("\nA @ v =", Av, "  (output vector)")

# Verify both readings give the same Av
col_view = v[0] * A[:, 0] + v[1] * A[:, 1]
row_view = np.array([A[0] @ v, A[1] @ v])
print("\nColumn-view computation of A @ v =", col_view)
print("Row-view computation of A @ v    =", row_view)
# Both equal [7, 11]. Same map, two readings.

# %% [markdown]
# Our example (deliberately NON-symmetric so rows and columns look different):
#
# ```
#    A = [[3, 2],      v = [1, 2]      A @ v = [7, 11]
#         [1, 5]]
#
#    rows of A:    [3, 2]   and   [1, 5]
#    cols of A:    [3, 1]   and   [2, 5]
#
#    All four are different vectors — no accidental coincidences.
# ```
#
# We'll visualize this computation TWO completely different ways and watch
# them produce the same output vector `[7, 11]`.

# %% [markdown]
# ## 3. Columns LIVE in the output space
#
# When we say "column 0 of A is `[3, 1]`," that tuple is an arrow you draw
# in the **output world**. Why? Because column `j` of `A` is what happens
# to input basis vector `e_j` when `A` is applied to it:
#
# ```
#   A · î = A · [1, 0]ᵀ = column 0 of A = [3, 1]ᵀ   ← lives in output space
#   A · ĵ = A · [0, 1]ᵀ = column 1 of A = [2, 5]ᵀ   ← lives in output space
# ```
#
# So the columns of `A` are literally the **landings** of the input basis
# vectors in the output world. Their natural home is the output canvas.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: INPUT SPACE with î and ĵ
ax = axes[0]
setup_ax_2d(ax, lim=6, title="INPUT SPACE (V)\nthe domain — where input vectors live")
draw_arrow_2d(ax, [1, 0], color="tab:blue",   label="î", label_offset=(0.05, -0.4))
draw_arrow_2d(ax, [0, 1], color="tab:orange", label="ĵ", label_offset=(-0.4, 0.05))
ax.text(0, -5.2, "we start with the two input basis vectors", ha="center",
        fontsize=10, style="italic")

# Right: OUTPUT SPACE with A·î and A·ĵ — these are the columns of A
ax = axes[1]
setup_ax_2d(ax, lim=6, title="OUTPUT SPACE (W)\nthe codomain — where A sends vectors")
draw_arrow_2d(ax, A[:, 0], color="tab:blue",   label=f"A·î  =  col 0 of A  =  {A[:,0]}",
              label_offset=(0.15, -0.4))
draw_arrow_2d(ax, A[:, 1], color="tab:orange", label=f"A·ĵ  =  col 1 of A  =  {A[:,1]}",
              label_offset=(-3.5, 0.2))
ax.text(0, -5.2, "the COLUMNS of A live HERE — they're where î, ĵ landed",
        ha="center", fontsize=10, style="italic", color="darkred")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Pause and absorb:** every column of `A` is an arrow that you draw in
# the right-hand (output) canvas. The columns describe the map by saying
# "here's where the input basis vectors end up."

# %% [markdown]
# ## 4. Rows LIVE in the input space
#
# Now the dual reading. When we say "row 0 of A is `[3, 2]`," that tuple
# is an arrow you draw in the **input world**. Why?
#
# Because row `i` of `A` is a **probe**: when you dot-product it with any
# input vector, you get one coordinate of the output:
#
# ```
#   (A · v)[0] = row 0 of A  ·  v  =  dot([3, 2], v)   ← row 0 EATS an input vector v
#   (A · v)[1] = row 1 of A  ·  v  =  dot([1, 5], v)   ← row 1 EATS an input vector v
# ```
#
# Each row is a function: *"give me an input vector, I'll give you back
# one number — that's how much of YOUR output coordinate is in there."*
#
# For a row to dot-product with an input vector, it must live in the
# **same space** as the input. So rows are arrows in the INPUT canvas.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: INPUT SPACE with the row vectors of A as probes
ax = axes[0]
setup_ax_2d(ax, lim=6, title="INPUT SPACE (V)\nrows of A live HERE as probes")
draw_arrow_2d(ax, A[0, :], color="crimson",
              label=f"row 0 of A  =  {A[0]}\n(probe for out_x)",
              label_offset=(0.15, -0.5))
draw_arrow_2d(ax, A[1, :], color="navy",
              label=f"row 1 of A  =  {A[1]}\n(probe for out_y)",
              label_offset=(0.15, 0.05))
ax.text(0, -5.2, "rows are probes: dot(row_i, v) extracts output coordinate i",
        ha="center", fontsize=10, style="italic", color="darkred")

# Right: OUTPUT SPACE showing the coordinate axes (out_x, out_y)
ax = axes[1]
setup_ax_2d(ax, lim=6, title="OUTPUT SPACE (W)\nwhere the resulting output coordinates land")
ax.annotate("", xy=(5.5, 0), xytext=(-5.5, 0),
            arrowprops=dict(arrowstyle="->", color="crimson", lw=2))
ax.text(5.6, -0.4, "out_x", color="crimson", fontsize=11, fontweight="bold")
ax.annotate("", xy=(0, 5.5), xytext=(0, -5.5),
            arrowprops=dict(arrowstyle="->", color="navy", lw=2))
ax.text(0.1, 5.6, "out_y", color="navy", fontsize=11, fontweight="bold")
ax.text(0, -5.5, "out_x = dot(row 0, v)\nout_y = dot(row 1, v)",
        ha="center", fontsize=10, style="italic", color="darkred")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Mental model check:** can you draw row 0 in the same picture as the
# input vector `v`? YES — they're both in the input world, both arrows in
# input-space. That's why you can dot-product them.
#
# Can you draw row 0 in the same picture as the output vector `A·v`?
# **NO** — row 0 doesn't live there. It lives in input space.

# %% [markdown]
# ## 5. The grand unified picture
#
# Now we put it all together. We're going to compute `A @ v = [7, 11]` two
# completely different ways and see them produce the same answer.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# === LEFT panel: INPUT SPACE — row-as-probe reading ===
ax = axes[0]
setup_ax_2d(ax, lim=6, title="INPUT SPACE  —  ROW reading (probes)")

# Input vector v
draw_arrow_2d(ax, v, color="black", lw=3,
              label=f"v = {v}\n(input)",
              label_offset=(0.1, 0.2))

# Row 0 (probe for out_x)
draw_arrow_2d(ax, A[0], color="crimson",
              label=f"row 0\n= {A[0]}",
              label_offset=(0.15, -0.6))
# Row 1 (probe for out_y)
draw_arrow_2d(ax, A[1], color="navy",
              label=f"row 1\n= {A[1]}",
              label_offset=(0.15, 0.05))

ax.text(0, -5.5,
        f"dot(row 0, v) = {A[0]}·{v} = {int(A[0] @ v)}   →  out_x\n"
        f"dot(row 1, v) = {A[1]}·{v} = {int(A[1] @ v)}   →  out_y",
        ha="center", fontsize=10, family="monospace", color="darkred")

# === RIGHT panel: OUTPUT SPACE — column-as-landing reading ===
ax = axes[1]
setup_ax_2d(ax, lim=13, title="OUTPUT SPACE  —  COLUMN reading (linear combo)")

# Show v[0] * col_0 and then v[1] * col_1 stacked head-to-tail
step1 = v[0] * A[:, 0]                 # 1 * [3, 1] = [3, 1]
step2 = step1 + v[1] * A[:, 1]         # [3, 1] + 2 * [2, 5] = [7, 11]

# Draw v[0] * col 0 from origin
draw_arrow_2d(ax, v[0] * A[:, 0], origin=(0, 0), color="tab:blue", lw=2,
              label=f"{v[0]:.0f}·col 0 = {v[0] * A[:, 0]}",
              label_offset=(0.15, -0.8))
# Draw v[1] * col 1 from end of previous arrow
draw_arrow_2d(ax, v[1] * A[:, 1], origin=tuple(step1), color="tab:orange", lw=2,
              label=f"{v[1]:.0f}·col 1 = {v[1] * A[:, 1]}",
              label_offset=(0.25, 0.15))
# Draw the resulting output vector A·v
draw_arrow_2d(ax, Av, color="black", lw=3,
              label=f"A·v = {Av}",
              label_offset=(0.1, 0.3))

ax.text(0, -12.2,
        f"A·v = v[0]·col 0 + v[1]·col 1\n"
        f"    = {v[0]:.0f}·{A[:,0]} + {v[1]:.0f}·{A[:,1]}\n"
        f"    = {v[0]*A[:,0]} + {v[1]*A[:,1]} = {Av}",
        ha="center", fontsize=10, family="monospace", color="darkblue")

plt.tight_layout()
plt.show()

print(f"\nBoth panels compute the same A·v = {Av}")
print("Left panel: rows act as probes in the input space.")
print("Right panel: columns act as landings in the output space.")
print("Same final number. Different visual story. One matrix.")

# %% [markdown]
# **This is the picture you need to install.** Look at it. Cover the
# right panel and reconstruct the answer using just the left. Cover the
# left and reconstruct using just the right. Either path gets you to
# `[7, 11]`. That's the duality made visible.

# %% [markdown]
# ## 6. The "double duty" of every entry  (the climax)
#
# This is the bit that should make you go *ohhhh*. Pick **any** entry of
# `A`. Say `A[0, 1] = 2`. That single number plays two roles at once:
#
# ```
#   A = [[3, 2],         ← entry A[0, 1] = 2 is at row 0, column 1
#        [1, 5]]
# ```
#
# **Reading 1 (column-wise, in OUTPUT space):**
#
# > *"Column 1 of A is `[2, 5]`. So `ĵ` lands at `[2, 5]` in output space.
# >  In particular, the FIRST coordinate of where `ĵ` lands is `2`."*
#
# ```
#   A[0, 1]  =  2  =  the out_x component of where ĵ lands
# ```
#
# **Reading 2 (row-wise, in INPUT space):**
#
# > *"Row 0 of A is `[3, 2]`. So `out_x = 3·in_x + 2·in_y`. In particular,
# >  the SECOND coordinate of the row 0 probe is `2`."*
#
# ```
#   A[0, 1]  =  2  =  how much of in_y the out_x probe pulls in
# ```
#
# These two readings are **the same fact stated two different ways**:
#
# ```
#   "how much does input direction j contribute to output direction i?"
#                                =
#   "how much of input direction j does the output-i probe pull in?"
#                                =
#                          A[i, j]
# ```
#
# **One number. Two questions. Same answer.** That's the link.

# %%
# Let's visualize the double duty of A[0, 1] = 2 explicitly.

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

# Draw the matrix as a grid
for i in range(2):
    for j in range(2):
        color = "yellow" if (i == 0 and j == 1) else "lavender"
        ax.add_patch(Rectangle((3 + j * 0.9, 3.5 - i * 0.9), 0.9, 0.9,
                               facecolor=color, edgecolor="black", lw=1.5))
        ax.text(3.45 + j * 0.9, 3.95 - i * 0.9, f"{int(A[i, j])}",
                ha="center", va="center", fontsize=16, fontweight="bold")

ax.text(3.9, 4.7, "A", fontsize=14, fontweight="bold", ha="center")
ax.text(3.9, 5.0, "(the matrix)", fontsize=9, ha="center", style="italic")

# Annotate the highlighted cell from both readings
ax.annotate(
    "ROW READING (input space):\n"
    "  This is the 2nd coord of row 0.\n"
    "  Row 0 = the 'out_x probe' = [3, 2].\n"
    "  So out_x = 3·in_x + 2·in_y.\n"
    "  → '2 units of in_y go into out_x'",
    xy=(4.85, 3.95), xytext=(7, 4.6), fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="crimson", lw=1.5),
    color="crimson", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="mistyrose", edgecolor="crimson"))

ax.annotate(
    "COLUMN READING (output space):\n"
    "  This is the 1st coord of column 1.\n"
    "  Column 1 = where ĵ lands = [2, 5].\n"
    "  → 'ĵ ends up with an out_x of 2'",
    xy=(4.85, 3.95), xytext=(7, 1.8), fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5),
    color="tab:blue", ha="left",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="aliceblue", edgecolor="tab:blue"))

ax.annotate(
    "A[0, 1] = 2",
    xy=(4.85, 3.95), xytext=(1, 0.5), fontsize=12, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
    color="black", ha="center")

ax.text(5, 0.05,
        "ONE NUMBER. TWO QUESTIONS. SAME ANSWER. That is what 'duality' means.",
        ha="center", fontsize=11, fontweight="bold", color="darkgreen")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. The transpose flip — `A.T` swaps the two homes
#
# Now you have a tool to actually understand what transposing a matrix
# *means*. When you transpose `A`:
#
# - **Rows of A** (which lived in INPUT space) become **columns of A.T**
#   (which now live in the OUTPUT space of A.T).
# - **Columns of A** (which lived in OUTPUT space) become **rows of A.T**
#   (which now live in the INPUT space of A.T).
#
# So `A.T` literally **flips the two homes**. The arrows didn't change —
# the labels of which space owns which set of arrows did.

# %%
A_T = A.T

print("A =\n", A)
print("\nA.T =\n", A_T)
print("\nRows of A (probes in INPUT space of A):     ", A[0], " and ", A[1])
print("Columns of A.T (landings in OUTPUT space of A.T):", A_T[:, 0], " and ", A_T[:, 1])
print("\n→ Same numbers, but now they live in a different space (A.T's output).")

# Visualize the two together
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# --- TOP ROW: matrix A ---
# Top-left: input space of A with rows of A
ax = axes[0, 0]
setup_ax_2d(ax, lim=6, title="A — INPUT space\n(rows of A live here)")
draw_arrow_2d(ax, A[0], color="crimson", label=f"row 0 of A = {A[0]}",
              label_offset=(0.15, -0.5))
draw_arrow_2d(ax, A[1], color="navy", label=f"row 1 of A = {A[1]}",
              label_offset=(0.15, 0.05))

# Top-right: output space of A with columns of A
ax = axes[0, 1]
setup_ax_2d(ax, lim=6, title="A — OUTPUT space\n(columns of A live here)")
draw_arrow_2d(ax, A[:, 0], color="tab:blue", label=f"col 0 of A = {A[:,0]}",
              label_offset=(0.15, -0.5))
draw_arrow_2d(ax, A[:, 1], color="tab:orange", label=f"col 1 of A = {A[:,1]}",
              label_offset=(-3.5, 0.2))

# --- BOTTOM ROW: matrix A.T (the SAME numbers, swapped homes) ---
# Bottom-left: input space of A.T = where ROWS of A.T live = COLUMNS of A
ax = axes[1, 0]
setup_ax_2d(ax, lim=6, title="A.T — INPUT space\n(rows of A.T = columns of A live here now)")
draw_arrow_2d(ax, A_T[0], color="tab:blue", label=f"row 0 of A.T = {A_T[0]}\n(was col 0 of A)",
              label_offset=(0.15, -0.7))
draw_arrow_2d(ax, A_T[1], color="tab:orange", label=f"row 1 of A.T = {A_T[1]}\n(was col 1 of A)",
              label_offset=(-4.2, 0.2))

# Bottom-right: output space of A.T = where COLUMNS of A.T live = ROWS of A
ax = axes[1, 1]
setup_ax_2d(ax, lim=6, title="A.T — OUTPUT space\n(cols of A.T = rows of A live here now)")
draw_arrow_2d(ax, A_T[:, 0], color="crimson", label=f"col 0 of A.T = {A_T[:,0]}\n(was row 0 of A)",
              label_offset=(0.15, -0.7))
draw_arrow_2d(ax, A_T[:, 1], color="navy", label=f"col 1 of A.T = {A_T[:,1]}\n(was row 1 of A)",
              label_offset=(0.15, 0.1))

plt.suptitle("Transpose flip: the SAME arrows trade spaces between A and A.T",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Why this is profound:** transposing is the operation that turns
# "probes" (rows) into "landings" (columns) and vice versa. That's why
# attention has both a "Q matrix" and a "K matrix" — Q's rows are probes
# in the embedding space (they ASK), K's rows are also probes — and
# `Q @ K.T` is the alignment between every (probe, probe) pair.

# %% [markdown]
# ## 8. A 3D → 2D example — the two spaces become undeniable
#
# So far our examples were 2D → 2D, which means both "input space" and
# "output space" happen to be R². They're visually identical and you have
# to keep reminding yourself they're different worlds.
#
# Let's break that ambiguity with a **3D → 2D** matrix. Now:
#
# - Input space = R³ (a 3D world).
# - Output space = R² (a 2D plane).
# - **Rows live in R³** (the input world).
# - **Columns live in R²** (the output world).
#
# Drawing them in the same plot would be impossible because they
# literally live in different-dimensional spaces.

# %%
A2 = np.array([[1, 2, 3],
               [4, 5, 6]], dtype=float)        # shape (2, 3) — R³ → R²

print("A2 shape:", A2.shape, "  (maps R³ → R²)")
print("A2 =\n", A2)
print("\nRows of A2 (live in R³, the input space):")
print("  row 0 =", A2[0], "  ← a 3D arrow")
print("  row 1 =", A2[1], "  ← a 3D arrow")
print("\nColumns of A2 (live in R², the output space):")
print("  col 0 =", A2[:, 0], "  ← a 2D arrow")
print("  col 1 =", A2[:, 1], "  ← a 2D arrow")
print("  col 2 =", A2[:, 2], "  ← a 2D arrow")

# %%
fig = plt.figure(figsize=(14, 6))

# LEFT: input space (R³) with the two rows as 3D arrows
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_xlim(-1, 6); ax.set_ylim(-1, 6); ax.set_zlim(-1, 6)
ax.set_xlabel("in_x"); ax.set_ylabel("in_y"); ax.set_zlabel("in_z")
ax.set_title("INPUT SPACE (R³)\nrows of A2 live here (probes)", fontsize=11)
draw_arrow_3d(ax, A2[0], color="crimson", label=f"row 0 = {A2[0]}", lw=2)
draw_arrow_3d(ax, A2[1], color="navy",    label=f"row 1 = {A2[1]}", lw=2)

# RIGHT: output space (R²) with the three columns as 2D arrows
ax = fig.add_subplot(1, 2, 2)
setup_ax_2d(ax, lim=7, title="OUTPUT SPACE (R²)\ncolumns of A2 live here (landings)")
draw_arrow_2d(ax, A2[:, 0], color="tab:blue",
              label=f"col 0 = {A2[:,0]}  (where in_x basis lands)",
              label_offset=(0.15, -0.4))
draw_arrow_2d(ax, A2[:, 1], color="tab:orange",
              label=f"col 1 = {A2[:,1]}  (where in_y basis lands)",
              label_offset=(0.15, 0.05))
draw_arrow_2d(ax, A2[:, 2], color="tab:green",
              label=f"col 2 = {A2[:,2]}  (where in_z basis lands)",
              label_offset=(-3.2, 0.4))

plt.tight_layout()
plt.show()

# %% [markdown]
# **Look at this carefully.** The rows are arrows in a 3D world. The
# columns are arrows in a 2D world. They can't even share a plot. And yet
# both fully describe the same map `A2: R³ → R²`. Same matrix, two homes.
#
# Whenever you see a matrix shape `(m, n)`:
#
# ```
#   shape (m, n)
#          │  │
#          │  └── input dimension  (rows have n entries → rows live in Rⁿ)
#          └───── output dimension (cols have m entries → cols live in Rᵐ)
# ```

# %% [markdown]
# ## 9. The deep reason — duality (in plain English)
#
# Here's the formal-math version, kept tight:
#
# A **linear map** `L: V → W` is a single mathematical object. It has
# infinitely many equivalent descriptions, but the two most useful are:
#
# 1. **Action on basis vectors.** Specify where each basis vector of `V`
#    lands in `W`. This gives you the columns of the matrix.
# 2. **Output-coordinate functionals.** Specify, for each output
#    coordinate, a linear functional on `V` that produces that coordinate.
#    Each such functional is an element of the dual space `V*`. This
#    gives you the rows of the matrix.
#
# **Why these are equivalent:** in finite dimensions, knowing where the
# basis vectors of `V` land in `W` is the same data as knowing how each
# output coordinate is a linear combination of input coordinates. The
# matrix entry `A[i, j]` *is* both descriptions at once — it's the
# coefficient that participates in both.
#
# **Why we can draw rows in input space:** in `Rⁿ` with the standard dot
# product, the dual space `V*` is canonically isomorphic to `V` (Riesz
# representation). So every linear functional `f: Rⁿ → R` has a unique
# vector `a ∈ Rⁿ` such that `f(v) = a · v`. That vector `a` is what we
# draw in input space — it's the row of the matrix, identified with the
# functional it represents.
#
# **What the transpose actually is:** `A.T` is the matrix of the
# **adjoint map** `A* : W* → V*`. The adjoint takes a functional on the
# output and pulls it back to a functional on the input. Under the
# standard inner product, `A.T` does this by swapping rows and columns
# — turning input-space probes into output-space landings and vice versa.
#
# **You do not need this formalism for daily ML work.** But knowing the
# words helps when you read theory papers (adjoint, dual space, Riesz,
# co-vector, etc.).

# %% [markdown]
# ## 10. ML connections — where you'll meet this duality again
#
# **`nn.Linear` and "neurons":**
# - `W` has shape `(out_dim, in_dim)`.
# - **Rows of W are neurons** — they are probes living in the input
#   embedding space. Each row dot-products with the input to produce one
#   output coordinate (one neuron's activation).
# - **Columns of W are where input basis directions land.** Useful when
#   you ask "if I nudge input feature 7, what happens to all outputs?"
#
# **Attention `Q @ K.T`:**
# - Each row of `Q` is a "query" — a probe living in the embedding space.
# - Each row of `K` is a "key" — also a probe in the embedding space.
# - `Q @ K.T` computes every dot product `dot(q_i, k_j)` and arranges
#   them as a grid. Each entry says *"how aligned is query i with key j?"*
# - Reading via columns: `Q @ K.T = Q @ (K.T)` is "apply the linear map
#   `K.T` to every row of Q." Different lens, same numbers.
#
# **LoRA / low-rank decompositions:**
# - You write `ΔW ≈ B @ A`, where `B` is tall (output-space columns) and
#   `A` is wide (input-space rows).
# - The rows of `A` are "input directions worth listening to."
# - The columns of `B` are "output directions worth producing."
# - The matrix is constructed by **combining input-space probes and
#   output-space landings** — explicitly using both worlds.
#
# **SVD: `A = U Σ V.T`**
# - **`V`** = orthonormal directions in **input space** (its columns are
#   the directions the map cares about in V).
# - **`U`** = orthonormal directions in **output space** (its columns are
#   where those input directions get sent).
# - **`Σ`** = the stretching that connects them.
# - SVD is literally the dictionary that translates between the two
#   spaces, picking the "best" probes in V and the "best" landings in W.

# %% [markdown]
# ## The unforgettable summary
#
# | Aspect | Rows of A | Columns of A |
# |--------|-----------|--------------|
# | Live in | **Input** space (V) | **Output** space (W) |
# | Role | Probes — dot with input → 1 output coord | Landings — where input basis vectors go |
# | Shape | One row has `in_dim` entries | One col has `out_dim` entries |
# | Number of them | `out_dim` (one per output) | `in_dim` (one per input basis) |
# | ML metaphor | **Neurons / feature detectors** | **Image of basis directions** |
# | What transpose does | Sends them to A.T's output | Sends them to A.T's input |
#
# ### Hey-dude one-liner
#
# > Hey dude, a matrix is **one map between two spaces**. Columns live in
# > the **output** space (where input basis vectors land). Rows live in
# > the **input** space (probes that extract output coordinates). Every
# > entry `A[i, j]` is the **same number** answering both questions:
# > *"how much does input direction j contribute to output direction i?"*
# > Transpose swaps the homes. SVD finds the best bases in both worlds at
# > once.
#
# ### The picture to never forget
#
# ```
#                       ┌──────────────────┐
#                       │  INPUT  SPACE V  │
#                       │                  │
#                       │  • input vectors │
#                       │  • ROWS of A     │  ← probes
#                       │  • COLUMNS of A.T│
#                       └────────┬─────────┘
#                                │
#                                │  the matrix A maps between them
#                                ▼
#                       ┌──────────────────┐
#                       │  OUTPUT SPACE W  │
#                       │                  │
#                       │  • output vectors│
#                       │  • COLUMNS of A  │  ← landings
#                       │  • ROWS of A.T   │
#                       └──────────────────┘
# ```
#
# **ML payoff:** Every time you see a weight matrix from now on, pause and
# label its two homes. Where does each *row* live? Where does each
# *column* live? Doing this 100 times will permanently install the dual
# picture in your brain — and you'll read transformer math the way most
# people read prose.

# %%
