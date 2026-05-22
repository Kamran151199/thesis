# %% [markdown]
# # Day 0: Vectors and Matrices — The Visual Intuition
#
# Before touching any ML, get this into your bones.
# Run each cell, look at the plots, change the numbers, re-run.
#
# WATCH FIRST (or alongside):
# 3Blue1Brown "Essence of Linear Algebra" — entire playlist:
#   https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab
#
# ## The arc (built from your own notebook)
#
# To understand matrices — and essentially everything in linear algebra —
# you have to grasp the concept of VECTORS first. Each section builds on
# the previous one:
#
#   § 1.  A vector is an arrow / a specific coordinate in space.
#   § 2.  A vector is a LINEAR COMBINATION of basis vectors.
#         Its components are the SCALARS that scale the basis vectors.
#   § 3.  Those basis vectors don't HAVE to be [1, 0] and [0, 1].
#         Swap them — that swap IS a matrix.
#   § 4.  K · v  =  "use v's components as the recipe; mix K's columns."
#   § 5.  M · K · v  =  re-represent v in K's basis, then re-represent
#         the result in M's basis. Right-to-left. Sequential.
#   § 6.  Order matters: M · K  ≠  K · M  in general (swap M for a shear).
#   § 7.  Rectangular matrices: when the input and output rooms have
#         different dimensions. 3×2 lifts 2D into 3D.
#   § 8.  Wide rectangular matrices SQUASH a high-dim input into a
#         low-dim output. Information gets lost.
#   § 9.  DOT PRODUCT — one number that tells you how aligned two arrows are.
#   § 10. DETERMINANT — how much M scales areas (and whether it flips them).
#   § 11. INVERSE MATRIX — undoing a transformation (only if det ≠ 0).
#   § 12. COLUMN SPACE — the universe of outputs M can produce.
#   § 13. NULL SPACE — the directions M crushes to zero.
#   § 14. RANK — dimensionality of the column space; counts surviving directions.
#   § 15. DUALITY — every dot product is secretly a 1×n matrix.
#   § 16. EIGENVECTORS & EIGENVALUES — directions M only stretches, never rotates.
#   § 17. ABSTRACT VECTOR SPACES — vectors are anything that adds and scales.

# %% Imports
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ---
# # § 1.  A vector is an arrow
#
# Imagine the vector  A = [2, 3].
#
# That vector is a specific 2D coordinate — an arrow from the origin to the
# point (2, 3). That's it. Not abstract. Just a location with a direction.

# %% Visualize a single vector
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)

A = np.array([2, 3])
ax.annotate("", xy=tuple(A), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2.5))
ax.text(A[0] + 0.1, A[1] + 0.1, f"A = {tuple(A)}",
        color="#C44E52", fontsize=13, fontweight="bold")
ax.set_title("A vector is just an arrow from the origin\n"
             "A = [2, 3]  →  go 2 right, 3 up")
plt.show()

# %% [markdown]
# ---
# # § 2.  A vector is a LINEAR COMBINATION of basis vectors
#
# Here's the move that unlocks everything. Imagine the same vector
# A = [2, 3] is actually a **recipe** for mixing two "basis vectors"
# of 2D space:
#
#     î = [1, 0]   ← unit step along the x-axis  (this is î)
#     ĵ = [0, 1]   ← unit step along the y-axis  (this is ĵ)
#
# Then A is saying:
#
#     "Scale î by 2 (the x-component) and scale ĵ by 3 (the y-component),
#      then add them up."
#
#     A = 2·î + 3·ĵ
#       = 2·[1, 0]  +  3·[0, 1]
#       = [2, 0]    +  [0, 3]
#       = [2, 3]                                    ← BOOM!
#
# Each element of A is a **scalar** that scales the basis vector of the
# corresponding dimension. The vector itself is the SUM of those scaled
# basis vectors.

# %% Verify the arithmetic numerically
i_hat = np.array([1, 0])
j_hat = np.array([0, 1])
A     = np.array([2, 3])

step1 = A[0] * i_hat
step2 = A[1] * j_hat
result = step1 + step2

print(f"2·î = 2 · {tuple(i_hat)} = {step1}")
print(f"3·ĵ = 3 · {tuple(j_hat)} = {step2}")
print(f"2·î + 3·ĵ = {result}     ← same as A = {tuple(A)}     ✓")

# %% Visualize the recipe: scaled basis vectors stacked tip-to-tail
fig, ax = plt.subplots(1, 1, figsize=(7, 7))
ax.set_xlim(-1, 5); ax.set_ylim(-1, 5)
ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)

# faint basis vectors
ax.annotate("", xy=tuple(i_hat), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=1.5, alpha=0.4))
ax.text(1.05, -0.4, "î", color="#4C72B0", fontsize=12)
ax.annotate("", xy=tuple(j_hat), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#55A868", lw=1.5, alpha=0.4))
ax.text(-0.4, 1.05, "ĵ", color="#55A868", fontsize=12)

# 2·î
ax.annotate("", xy=tuple(step1), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=2.5))
ax.text(1.0, -0.5, "2·î", color="#4C72B0", fontsize=12, fontweight="bold")

# 3·ĵ stacked at the tip of 2·î (tip-to-tail)
ax.annotate("", xy=tuple(result), xytext=tuple(step1),
            arrowprops=dict(arrowstyle="->", color="#55A868", lw=2.5))
ax.text(step1[0] + 0.15, step1[1] + 1.2, "3·ĵ",
        color="#55A868", fontsize=12, fontweight="bold")

# the result A
ax.annotate("", xy=tuple(result), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=2.5))
ax.text(result[0] + 0.1, result[1] + 0.1, f"A = 2î + 3ĵ = {tuple(result)}",
        color="#C44E52", fontsize=12, fontweight="bold")

ax.set_title("A vector is a SUM of scaled basis vectors\n"
             "A's components ARE the scalars in that recipe")
plt.show()

# %% [markdown]
# ---
# # § 3.  Wait — why are the basis vectors [1, 0] and [0, 1]?
#
# Fair question — and actually, they DON'T have to be!
#
# You could pick any two basis vectors you like, and our learned concept
# would still hold true. A = [2, 3] would still mean "2 of the new î +
# 3 of the new ĵ." But the *final direction* of A would be totally
# different — coz you'd now be doing a linear combination of scaled basis
# vectors of a **different basis**.
#
# That "changed basis vectors" can be packed and stored — as a **MATRIX**.
#
# The matrix is saying (in your hey-dude voice):
#
#     "Hey dude, from now on:
#       - your new î is **column 0** of the matrix
#       - your new ĵ is **column 1** of the matrix"
#
# Example — let's pick the rotation matrix K (rotate 90° counterclockwise):
#
#     K = [ 0  -1 ]      ← reading columns:
#         [ 1   0 ]         new î = [0, 1]   (the old î got moved UP)
#                           new ĵ = [-1, 0]  (the old ĵ got moved LEFT)
#
# Picture: the whole 2D plane gets rotated 90° to the left.

# %% Build K and inspect its columns
K = np.array([[0, -1],
              [1,  0]])             # rotate 90° CCW

new_i = K[:, 0]                     # column 0  → where î lands
new_j = K[:, 1]                     # column 1  → where ĵ lands

print("K =\n", K)
print(f"\nnew î  =  column 0 of K  =  {tuple(new_i)}")
print(f"new ĵ  =  column 1 of K  =  {tuple(new_j)}")
print()
print("Sanity check via matrix-vector product:")
print(f"  K · î  =  K · {tuple(i_hat)}  =  {K @ i_hat}   ← same as column 0   ✓")
print(f"  K · ĵ  =  K · {tuple(j_hat)}  =  {K @ j_hat}   ← same as column 1   ✓")

# %% Shared 2D helpers — used everywhere below
def draw_arrow_2d(ax, vec, color, label=None, lw=2.5):
    ax.annotate("", xy=tuple(vec), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if label is not None:
        ax.text(vec[0] * 1.05 + 0.2, vec[1] * 1.05 + 0.2,
                label, color=color, fontsize=11, fontweight="bold")

def setup_ax_2d(ax, title, lim=4):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(title, fontsize=12)

def draw_polygon(ax, pts, color, label, alpha=0.3):
    ax.fill(pts[0], pts[1], alpha=alpha, color=color, label=label)
    ax.plot(pts[0], pts[1], color=color, lw=2)

unit_square = np.array([[0, 1, 1, 0, 0],
                        [0, 0, 1, 1, 0]])

# %% Visualize: same A, two different bases  →  different final direction
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
setup_ax_2d(ax, "Standard basis\n(î = (1,0),  ĵ = (0,1))")
draw_arrow_2d(ax, i_hat, "#4C72B0", "î")
draw_arrow_2d(ax, j_hat, "#55A868", "ĵ")
draw_arrow_2d(ax, A,     "#C44E52", f"A = 2î + 3ĵ = {tuple(A)}")

ax = axes[1]
new_A = K @ A
setup_ax_2d(ax, "K's basis\n(new î = (0,1),  new ĵ = (-1,0))")
draw_arrow_2d(ax, new_i, "#4C72B0", "new î = (0,1)")
draw_arrow_2d(ax, new_j, "#55A868", "new ĵ = (-1,0)")
draw_arrow_2d(ax, new_A, "#C44E52", f"K·A = 2·newî + 3·newĵ = {tuple(new_A)}")

plt.suptitle("Same recipe [2, 3], different ingredients  →  different result",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # § 4.  K · v  =  the recipe applied to K's columns
#
# Hey dude voice: "give me your vector v, and I'll mix MY columns using
# v's components as the recipe."
#
#     K · v  =  v[0]·col_0(K)  +  v[1]·col_1(K)
#
# Three equivalent ways to compute it — all give the same answer:
#
#   (a) numpy directly:                    K @ v
#   (b) by-hand linear combo of columns:   v[0]*K[:,0] + v[1]*K[:,1]
#   (c) walk through arithmetic:           use the matrix-product formula

# %% Three routes, one answer
print(f"K · A   (numpy)                          = {K @ A}")
print(f"A[0]·col_0(K) + A[1]·col_1(K)            = "
      f"{A[0] * K[:, 0] + A[1] * K[:, 1]}")
print("Same answer either way.")

# %% Determinant of K — how does K scale areas?  (full treatment in § 10)
print(f"\ndet(K) = {np.linalg.det(K):.2f}")
print("→ |det| = 1 means K preserves area exactly (a pure rotation).")

# %% [markdown]
# ---
# # § 5.  Multiplying matrices = applying transformations in sequence
#
# Now what happens if I multiply ONE matrix to ANOTHER?
#
# That means I'm trying to apply **several transformations in sequential order**.
# The right-most matrix goes first.
#
#     M · K · v
#
# Step-by-step, this means:
#
#     (1) Take v.
#     (2) Re-represent v via K's new basis vectors  →  call this v' = K·v.
#     (3) Take v' and re-represent IT via M's new basis vectors
#                                                   →  v'' = M·v' = M·K·v.
#
# Concretely:
#
#     K = [ 0  -1 ]   ←  K says: "from now on your basis is (0,1) and (-1,0)"
#         [ 1   0 ]      (= rotate 90° CCW)
#
#     M = [ 2   0 ]   ←  M says: "from now on your basis is (2,0) and (0,2)"
#         [ 0   2 ]      (= scale everything by 2)
#
# End result of M · K · v:
#     (1)  K rotated the vector.
#     (2)  M then doubled the rotated vector in both dimensions.

# %% Define M and reuse K from above
M = np.array([[2, 0],
              [0, 2]])      # uniform scale by 2

v = np.array([2, 3])

print("M (scale by 2) =\n", M)
print("\nK (rotate 90°) =\n", K)
print(f"\nv = {v}")

# %% Walk through the chain in your own words
v_after_K        = K @ v
v_after_M_then_K = M @ v_after_K

print(f"\nv                                       = {v}")
print(f"K · v       (re-represent v in K's basis) = {v_after_K}     ← rotated 90° CCW")
print(f"M · (K·v)   (re-represent in M's basis)   = {v_after_M_then_K}    ← then doubled")
print(f"\nDirectly via numpy:  M @ K @ v          = {M @ K @ v}    ← same answer")

# %% Visualize the chain: original → after K → after M
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
setup_ax_2d(ax, "1. Original  (standard basis)", lim=10)
draw_arrow_2d(ax, i_hat, "#4C72B0", "î", lw=2)
draw_arrow_2d(ax, j_hat, "#55A868", "ĵ", lw=2)
draw_arrow_2d(ax, v,     "#C44E52", f"v = {tuple(v)}")

ax = axes[1]
setup_ax_2d(ax, f"2. After K  (re-represented in K's basis)\nK·v = {tuple(v_after_K)}", lim=10)
draw_arrow_2d(ax, K @ i_hat, "#4C72B0", "K·î", lw=2)
draw_arrow_2d(ax, K @ j_hat, "#55A868", "K·ĵ", lw=2)
draw_arrow_2d(ax, v_after_K, "#C44E52", "K·v")

ax = axes[2]
setup_ax_2d(ax, f"3. After M  (re-represented in M's basis)\nM·K·v = {tuple(v_after_M_then_K)}", lim=10)
draw_arrow_2d(ax, M @ K @ i_hat, "#4C72B0", "M·K·î", lw=2)
draw_arrow_2d(ax, M @ K @ j_hat, "#55A868", "M·K·ĵ", lw=2)
draw_arrow_2d(ax, v_after_M_then_K, "#C44E52", "M·K·v")

plt.suptitle("M @ K @ v  —  right-to-left composition:  K first, then M",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## § 5b.  Subtlety: the columns of the *merged* matrix M·K
#
# Your sequential reading ("v → K·v → M·K·v") is the right intuition.
#
# A different (but equivalent) question is: "what if I precompute the ONE
# merged matrix that does the entire job at once? What do ITS columns look
# like?"
#
# Answer: the columns of M·K are **M applied to K's columns** — not just
# M's columns or K's columns on their own.
#
#     col 0 of M·K  =  M · (K · î)  =  M · col_0(K)
#     col 1 of M·K  =  M · (K · ĵ)  =  M · col_1(K)
#
# Same final answer; just precomputed into a single 2×2 matrix.

# %% Compare: merged columns vs sequential M-applied-to-K's-columns
MK = M @ K
print("M · K =\n", MK.round(3))
print()
print(f"col 0 of M·K   = {MK[:, 0].round(3)}     ==     M @ (K @ î) = {(M @ (K @ i_hat)).round(3)}")
print(f"col 1 of M·K   = {MK[:, 1].round(3)}     ==     M @ (K @ ĵ) = {(M @ (K @ j_hat)).round(3)}")

# %% Three equivalent ways to compute M · K · v
print(f"M @ K @ v                                = {M @ K @ v}")
print(f"(M @ K) @ v   (use merged matrix once)   = {MK @ v}")
print(f"v[0]·col_0(MK) + v[1]·col_1(MK)          = "
      f"{v[0] * MK[:, 0] + v[1] * MK[:, 1]}")

# %% Determinants — algebra mirrors geometry
print(f"det(M)    = {np.linalg.det(M):.2f}   (areas scaled ×4)")
print(f"det(K)    = {np.linalg.det(K):.2f}   (rotation preserves area)")
print(f"det(M·K)  = {np.linalg.det(MK):.2f}   = det(M) · det(K)")

# %% [markdown]
# ---
# # § 6.  Order matters!   M · K  ≠  K · M  in general
#
# "Apply K then M" is NOT the same as "apply M then K." Matrix
# multiplication is **not commutative**.
#
# In the M (scale-by-2) + K (rotation) example, swapping order actually
# gives the same answer — but only because scale-by-2 is `2·I` (a scaled
# identity), which commutes with EVERYTHING. As soon as you swap M for a
# SHEAR, the order starts mattering visibly.

# %% Swap M for a SHEAR matrix and watch composition stop commuting
S = np.array([[1, 1],
              [0, 1]])      # shear: pushes y-stuff sideways

SK = S @ K
KS = K @ S
print("S · K =\n", SK)
print()
print("K · S =\n", KS)
print()
print("Different!  Matrix multiplication is NOT commutative in general.")

# %% Visualize the order difference on a unit square
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
setup_ax_2d(ax, "Original unit square", lim=4)
draw_polygon(ax, unit_square, "#888888", "original")
ax.legend(loc="upper right")

ax = axes[1]
setup_ax_2d(ax, "S · K · square\n(rotate → shear)", lim=4)
draw_polygon(ax, unit_square,         "#888888", "original", alpha=0.15)
draw_polygon(ax, K @ unit_square,     "#DD8452", "after K",  alpha=0.25)
draw_polygon(ax, S @ K @ unit_square, "#C44E52", "after S",  alpha=0.4)
ax.legend(loc="upper right")

ax = axes[2]
setup_ax_2d(ax, "K · S · square\n(shear → rotate)", lim=4)
draw_polygon(ax, unit_square,         "#888888", "original", alpha=0.15)
draw_polygon(ax, S @ unit_square,     "#DD8452", "after S",  alpha=0.25)
draw_polygon(ax, K @ S @ unit_square, "#C44E52", "after K",  alpha=0.4)
ax.legend(loc="upper right")

plt.suptitle("Same two matrices, different order = different final shape",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # § 7.  Rectangular matrices — when the rooms change dimension
#
# So far every matrix has been SQUARE (2×2): 2 new basis vectors, each in 2D.
# Input was 2D, output was 2D — same room.
#
# But a matrix doesn't have to be square. Read its shape carefully:
#
#     M shape  =  (rows × cols)
#              =  (output_dim × input_dim)
#
#     cols  =  # of new basis vectors I'm giving you  (must match input dim)
#     rows  =  # of coordinates each new basis vector needs  (= output dim)
#
# A 3×2 matrix says:
#     • 2 columns  ⇒  giving me 2 new basis vectors  ⇒  input must be 2D
#     • 3 rows     ⇒  each new basis vector lives in 3D  ⇒  output is 3D
#
# Hey-dude voice:
#     "From now on, your new î is some 3D vector, and your new ĵ is some
#      OTHER 3D vector. So when you mix them using your 2D recipe, you
#      land somewhere in 3D space."
#
# Picture: the flat 2D plane gets picked up, tilted, and embedded inside a 3D
# room as a tilted slab.

# %% Define a 3×2 LIFT matrix
M_lift = np.array([[1, 0],
                   [0, 1],
                   [1, 2]])     # 3 rows × 2 cols   →   2D in, 3D out

v2d = np.array([3, 2])          # any 2D input

print("M_lift.shape =", M_lift.shape, "  (3 rows × 2 cols)")
print("Reads as: takes a 2D vector in, returns a 3D vector out")
print("\nM_lift =\n", M_lift)
print(f"\nv (2D) = {v2d}")

# %% Where do î, ĵ, v LAND in 3D?
i_lifted = M_lift @ i_hat
j_lifted = M_lift @ j_hat
v_lifted = M_lift @ v2d

print("\n2D î = (1, 0)    →    M_lift · î = ", i_lifted, "   ← column 0 (new î, now 3D)")
print("2D ĵ = (0, 1)    →    M_lift · ĵ = ", j_lifted, "   ← column 1 (new ĵ, now 3D)")
print(f"2D v = {tuple(v2d)}    →    M_lift · v = {tuple(v_lifted)}")

combo = v2d[0] * M_lift[:, 0] + v2d[1] * M_lift[:, 1]
print(f"\nRecipe check:  v[0]·col_0 + v[1]·col_1 = {combo}")
print("→ matches M_lift·v exactly. Recipe unchanged; ingredients now live in 3D.")

# %% Visualize: 2D input (left)   →   3D embedding (right)
fig = plt.figure(figsize=(14, 6))

ax2d = fig.add_subplot(1, 2, 1)
ax2d.set_xlim(-1, 5); ax2d.set_ylim(-1, 5)
ax2d.set_aspect("equal"); ax2d.grid(True, alpha=0.3)
ax2d.axhline(0, color="black", lw=0.5); ax2d.axvline(0, color="black", lw=0.5)
ax2d.set_title("Input: 2D plane")

draw_arrow_2d(ax2d, i_hat, "#4C72B0", "î = (1,0)", lw=2)
draw_arrow_2d(ax2d, j_hat, "#55A868", "ĵ = (0,1)", lw=2)
draw_arrow_2d(ax2d, v2d,   "#C44E52", f"v = {tuple(v2d)}")

ax3d = fig.add_subplot(1, 2, 2, projection="3d")

xx, yy = np.meshgrid(np.linspace(-1, 5, 8), np.linspace(-1, 5, 8))
ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                  alpha=0.1, color="gray", edgecolor="none")

aa, bb = np.meshgrid(np.linspace(-1, 4, 12), np.linspace(-1, 4, 12))
plane_x = aa * i_lifted[0] + bb * j_lifted[0]
plane_y = aa * i_lifted[1] + bb * j_lifted[1]
plane_z = aa * i_lifted[2] + bb * j_lifted[2]
ax3d.plot_surface(plane_x, plane_y, plane_z,
                  alpha=0.25, color="#C44E52", edgecolor="none")

def draw_arrow_3d(ax, vec, color, label, lw=2.5):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2],
              color=color, lw=lw, arrow_length_ratio=0.08)
    ax.text(vec[0] * 1.1, vec[1] * 1.1, vec[2] * 1.1,
            label, color=color, fontsize=11, fontweight="bold")

draw_arrow_3d(ax3d, i_lifted, "#4C72B0", "new î (3D)")
draw_arrow_3d(ax3d, j_lifted, "#55A868", "new ĵ (3D)")
draw_arrow_3d(ax3d, v_lifted, "#C44E52", "M·v")

ax3d.set_xlim(-1, 5); ax3d.set_ylim(-1, 5); ax3d.set_zlim(-1, 8)
ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
ax3d.view_init(elev=22, azim=-55)
ax3d.set_title("Output: 3D space\n(red sheet = image of M, a tilted plane)")

plt.suptitle("3×2 matrix:  2D in  →  3D out\n"
             "flat 2D plane gets lifted into 3D as a tilted slab",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## § 7b.  Every 2D point lands on the SAME tilted plane in 3D
#
# Take a grid of 2D points. Lift them all using M_lift. Notice: every output
# sits on the SAME tilted plane — that plane IS the "image" of M (also called
# the **column space** of M, which we'll cover formally in § 12).

# %% Grid in 2D → grid on tilted plane in 3D
xs = np.linspace(-1, 4, 6)
ys = np.linspace(-1, 4, 6)
grid_2d_x, grid_2d_y = np.meshgrid(xs, ys)
grid_2d = np.stack([grid_2d_x.ravel(), grid_2d_y.ravel()])
grid_3d = M_lift @ grid_2d

fig = plt.figure(figsize=(13, 6))

ax2d = fig.add_subplot(1, 2, 1)
ax2d.scatter(grid_2d[0], grid_2d[1], s=50, c="#888888", edgecolor="black")
ax2d.set_xlim(-2, 5); ax2d.set_ylim(-2, 5)
ax2d.set_aspect("equal"); ax2d.grid(True, alpha=0.3)
ax2d.axhline(0, color="black", lw=0.5); ax2d.axvline(0, color="black", lw=0.5)
ax2d.set_title("36 points scattered across 2D")

ax3d = fig.add_subplot(1, 2, 2, projection="3d")
ax3d.scatter(grid_3d[0], grid_3d[1], grid_3d[2],
             s=50, c="#888888", edgecolor="black")
ax3d.plot_surface(plane_x, plane_y, plane_z,
                  alpha=0.15, color="#C44E52", edgecolor="none")
ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
ax3d.set_xlim(-2, 5); ax3d.set_ylim(-2, 5); ax3d.set_zlim(-2, 10)
ax3d.view_init(elev=22, azim=-55)
ax3d.set_title("…all 36 lifted points lie on ONE tilted plane in 3D\n"
               "(the column space of M_lift)")

plt.suptitle("Every 2D point gets lifted to a 3D point on M's image plane",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% Algebraic check: every output satisfies  z = x + 2y
residuals = grid_3d[2] - (grid_3d[0] + 2 * grid_3d[1])
print(f"max |z - (x + 2y)| over all 36 lifted points: {np.max(np.abs(residuals)):.2e}")
print("→ ~0, so every output really does live on z = x + 2y.")

print(f"\nrank(M_lift) = {np.linalg.matrix_rank(M_lift)}")
print("Image is 2-dimensional even though it sits inside 3-dim ambient space.")

# %% [markdown]
# ---
# # § 8.  Wide rectangular matrices — squashing (info gets crushed)
#
# Now the reverse direction. A 2×3 matrix says:
#     • 3 columns  ⇒  giving me 3 new basis vectors  ⇒  input must be 3D
#     • 2 rows     ⇒  each new basis vector is only 2D  ⇒  output is 2D
#
# Hey-dude voice:
#     "From now on, your new î, ĵ, AND k̂ are each 2D vectors. So your 3D
#      recipe mixes them into a single 2D point. The third direction collapses
#      into the 2D answer — you can't recover where it came from."

# %% A 2×3 matrix that drops the z-coordinate (literally a shadow on the floor)
P = np.array([[1, 0, 0],
              [0, 1, 0]])      # 2 rows × 3 cols   →   3D in, 2D out

a = np.array([1, 2,  5])
b = np.array([1, 2, -3])
c = np.array([1, 2,  99])

print(f"P @ {tuple(a)} = {P @ a}")
print(f"P @ {tuple(b)} = {P @ b}")
print(f"P @ {tuple(c)} = {P @ c}")
print("\nThree very different 3D points → same 2D shadow.")
print("The z-direction got collapsed; we cannot recover it from the 2D output.")
print(f"\nrank(P) = {np.linalg.matrix_rank(P)}   (image is 2D)")

# %% [markdown]
# ---
# # § 9.  DOT PRODUCT — "how much do two arrows agree?"
#
# A dot product is ONE NUMBER (a scalar) that tells you how aligned two
# arrows are.
#
# ## The algebra
#
#     a · b  =  a[0]·b[0]  +  a[1]·b[1]  +  ...  +  a[n-1]·b[n-1]
#
# Pair components, multiply, sum. Output: a single scalar.
#
# ## The geometry
#
#     a · b  =  |a| · |b| · cos(θ)        where θ is the angle between them
#
# So:
#     θ = 0°    →  cos = +1   →  big positive   (same direction)
#     θ = 90°   →  cos =  0   →  zero            (perpendicular)
#     θ = 180°  →  cos = -1   →  big negative   (opposite)
#
# Both formulas give the same number. The bridge is the law of cosines.
#
# Hey-dude voice:
#     "Hey dude, give me two arrows. I'll give you ONE number that says how
#      much they're pointing the same way. Big positive = strongly agreeing.
#      Big negative = strongly opposing. Zero = totally unrelated."

# %% Three cases, visualized
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cases = [
    ("Same direction\n(positive)",      np.array([3, 0]),  np.array([2, 0])),
    ("Perpendicular\n(zero)",            np.array([3, 0]),  np.array([0, 2])),
    ("Opposite directions\n(negative)",  np.array([3, 0]),  np.array([-2, 0])),
]

for ax, (title, a, b) in zip(axes, cases):
    setup_ax_2d(ax, f"{title}\na · b = {int(np.dot(a, b))}", lim=4)
    draw_arrow_2d(ax, a, "#4C72B0", "a")
    draw_arrow_2d(ax, b, "#C44E52", "b")

plt.suptitle("Dot product = signed measure of directional agreement",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% The shadow picture — drop a perpendicular from a onto b's line
fig, ax = plt.subplots(1, 1, figsize=(7, 6))
setup_ax_2d(ax, "a · b  =  (shadow of a on b's line) · |b|", lim=5)

a = np.array([3, 2])
b = np.array([4, 0])
draw_arrow_2d(ax, a, "#4C72B0", "a = (3, 2)")
draw_arrow_2d(ax, b, "#C44E52", "b = (4, 0)")

# Shadow of a on b's direction (b is along x-axis here, so shadow = a's x-component)
shadow_endpoint = np.array([a[0], 0])
ax.plot([a[0], shadow_endpoint[0]], [a[1], shadow_endpoint[1]],
        "k--", lw=1)
ax.annotate("", xy=tuple(shadow_endpoint), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#888888", lw=2))
ax.text(1.5, -0.4, f"shadow = {a[0]}", color="#888888", fontsize=11,
        fontweight="bold")
ax.text(0.5, 3.5, f"a · b = shadow · |b| = {a[0]} · {int(np.linalg.norm(b))} = {int(np.dot(a, b))}",
        fontsize=11, fontweight="bold")
plt.show()

# %% Why î · ĵ = 0 — the standard basis is ORTHONORMAL
print(f"î · î = {int(np.dot(i_hat, i_hat))}    (length-1, aligned with itself)")
print(f"ĵ · ĵ = {int(np.dot(j_hat, j_hat))}    (length-1)")
print(f"î · ĵ = {int(np.dot(i_hat, j_hat))}    (perpendicular)")
print()
print("'Ortho' = orthogonal = perpendicular  →  basis-vector pairs dot to 0.")
print("'Normal' = each basis vector has length 1.")
print("That's why the simple formula  a·b = a[0]·b[0] + a[1]·b[1]  works:")
print("all the cross-terms vanish because the basis is orthonormal.")

# %% Cosine similarity — the ML workhorse
def cosine_sim(u, w):
    return np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))

# Fake embeddings (768-D, like in a real VLM)
np.random.seed(42)
king  = np.random.randn(768)
queen = king + np.random.randn(768) * 0.3     # similar
cat   = np.random.randn(768)                   # unrelated

print(f"cos(king, queen) = {cosine_sim(king, queen):.3f}   ← similar  → high")
print(f"cos(king, cat)   = {cosine_sim(king, cat):.3f}   ← different → low")
print("\nTHIS is what CLIP computes between image and text embeddings.")
print("THIS is what attention computes between tokens.")

# %% [markdown]
# ---
# # § 10.  DETERMINANT — "how much does M scale areas?"
#
# Take the unit square (area = 1) and apply M to it. You get a parallelogram
# with some new area. That new area IS det(M).
#
# More precisely, det(M) is the SIGNED area-scaling factor:
#
#     |det(M)|  =  how much M scales areas (2D) or volumes (3D)
#     sign(det) =  whether M flipped orientation (left-right mirror)
#
# Hey-dude voice:
#     "Hey dude, take the unit square. After I'm done with it, the new
#      area IS my determinant. If I flipped left and right too, my
#      determinant is negative."
#
# Cases:
#     det = +4   →  M scales areas by 4×; orientation preserved
#     det = -4   →  M scales areas by 4×; but ALSO flipped left/right
#     det =  0   →  M crushed a dimension to nothing  →  area = 0  →  not invertible
#     det =  1   →  M preserves area exactly (rotations, shears)

# %% Visualize determinants for various 2×2 matrices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

det_examples = [
    (np.array([[1, 0], [0, 1]]),      "Identity"),
    (np.array([[2, 0], [0, 2]]),      "Scale ×2"),
    (np.array([[2, 0], [0, 1]]),      "Stretch x"),
    (np.array([[1, 1], [0, 1]]),      "Shear"),
    (np.array([[0, -1], [1, 0]]),     "Rotation 90°"),
    (np.array([[1, 2], [2, 4]]),      "Rank-1 collapse"),
]

for ax, (mat, title) in zip(axes.flat, det_examples):
    det = np.linalg.det(mat)
    transformed = mat @ unit_square

    if abs(det) < 0.01:
        color = "#888888"     # crushed
    elif det > 0:
        color = "#55A868"     # green for positive
    else:
        color = "#C44E52"     # red for negative

    setup_ax_2d(ax, f"{title}\ndet = {det:.2f}", lim=4)
    draw_polygon(ax, unit_square, "#CCCCCC", "original", alpha=0.3)
    draw_polygon(ax, transformed, color, f"after M (area={abs(det):.2f})",
                 alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)

plt.suptitle("Determinant = signed area-scaling factor of the unit square",
             fontsize=14, y=1.0)
plt.tight_layout()
plt.show()

# %% Determinant of a 3×3 = signed VOLUME of the unit cube
M3 = np.array([[1, 0, 0],
               [0, 2, 0],
               [0, 0, 3]])
print(f"M3 = diag(1, 2, 3)")
print(f"det(M3) = {np.linalg.det(M3):.2f}   (unit cube becomes a 1×2×3 box = volume 6)")

# %% Why ML cares
# - In normalizing flows, det of the Jacobian tracks how a transformation
#   stretches probability density (change-of-variables formula).
# - det = 0 detects when a layer has collapsed information (degenerate).
# - In linear regression solvability, det of (X^T X) must be ≠ 0.

# %% [markdown]
# ---
# # § 11.  INVERSE MATRIX — undoing a transformation
#
# If M transforms v into M·v, can we rewind it?
#
#     M · M⁻¹  =  I       (the identity)
#     M⁻¹ · M  =  I
#
# Geometrically, M⁻¹ is the transformation that UNDOES whatever M did:
#     - M rotates 90° CCW   →  M⁻¹ rotates 90° CW
#     - M scales by 2        →  M⁻¹ scales by 1/2
#     - M shears right       →  M⁻¹ shears left
#
# **CRUCIAL**: M is invertible IF AND ONLY IF det(M) ≠ 0.
#
# Why? If det(M) = 0, M crushed a dimension. You can't un-crush it —
# many different inputs collapsed to the same output, so there's no
# way to recover which input belongs to which output.
#
# Hey-dude voice:
#     "Hey dude, if I didn't destroy information, you can rewind me.
#      If I crushed a dimension flat, that info is gone forever."

# %% A 2×2 inverse, computed
Mi = np.array([[3, 1],
               [2, 1]])
Mi_inv = np.linalg.inv(Mi)

print("M =\n", Mi)
print(f"\ndet(M) = {np.linalg.det(Mi):.2f}    (≠ 0, so invertible)")
print(f"\nM⁻¹ =\n{Mi_inv}")
print(f"\nM @ M⁻¹ =\n{(Mi @ Mi_inv).round(3)}    ← identity ✓")

# %% Visualize: apply M, then M⁻¹, get back the original
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
setup_ax_2d(ax, "Step 1: original square", lim=5)
draw_polygon(ax, unit_square, "#888888", "original")

ax = axes[1]
setup_ax_2d(ax, "Step 2: after M\n(distorted)", lim=5)
draw_polygon(ax, unit_square,      "#CCCCCC", "original", alpha=0.2)
draw_polygon(ax, Mi @ unit_square, "#DD8452", "after M",   alpha=0.5)

ax = axes[2]
setup_ax_2d(ax, "Step 3: after M⁻¹·M\n(back to original!)", lim=5)
draw_polygon(ax, unit_square,                "#CCCCCC", "original",  alpha=0.2)
draw_polygon(ax, Mi @ unit_square,           "#DD8452", "after M",   alpha=0.2)
draw_polygon(ax, Mi_inv @ Mi @ unit_square,  "#55A868", "after M⁻¹", alpha=0.6)

plt.suptitle("M⁻¹ undoes M  →  any reversible transformation can be rewound",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %% When the inverse doesn't exist
M_singular = np.array([[1, 2],
                       [2, 4]])
print(f"M_singular =\n{M_singular}")
print(f"det(M_singular) = {np.linalg.det(M_singular):.2f}")
print("Row 2 is just 2× row 1 → both columns point along the same line.")
print("M crushes everything onto that line → information is LOST.")
print("Cannot be inverted.")
try:
    np.linalg.inv(M_singular)
except np.linalg.LinAlgError as e:
    print(f"\nnumpy.linalg.inv raises:  {e}")

# %% [markdown]
# ---
# # § 12.  COLUMN SPACE — "the universe of outputs M can produce"
#
#     column_space(M)  =  span of M's columns
#                      =  { M · v   for every possible input v }
#                      =  every output M can possibly produce
#
# Hey-dude voice:
#     "Hey dude, here's the universe of outputs I can hit. ANY vector you
#      give me will land somewhere in this set. Outside this set? I cannot
#      reach it."
#
# Cases:
#     - Full-rank square M  →  column space = entire output space
#                              (M can produce anything)
#     - Rank-deficient M    →  column space is a lower-dim subspace
#                              (M can only produce outputs on this slab)
#     - 3×2 lift            →  column space = 2D plane inside 3D
#                              (= image plane we saw in § 7b)

# %% Visualize column space of a rank-1 matrix (its image is just a LINE)
M_rank1 = np.array([[1, 2],
                    [2, 4]])

# Take a grid of inputs, plot the outputs
n = 25
xs = np.linspace(-3, 3, n)
ys = np.linspace(-3, 3, n)
grid_in = np.array([(x, y) for x in xs for y in ys]).T
grid_out = M_rank1 @ grid_in

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
setup_ax_2d(ax, "Input grid (full 2D plane)", lim=4)
ax.scatter(grid_in[0], grid_in[1], s=6, c="#888888")

ax = axes[1]
setup_ax_2d(ax, "Output: collapsed onto ONE line\n(column space of M_rank1)", lim=18)
ax.scatter(grid_out[0], grid_out[1], s=6, c="#C44E52")
# draw M_rank1's two columns (both point along the same line)
draw_arrow_2d(ax, M_rank1[:, 0], "#4C72B0", "col 0", lw=2)
draw_arrow_2d(ax, M_rank1[:, 1], "#55A868", "col 1", lw=2)

plt.suptitle("Rank-1 M: image is a 1D line, not the full 2D plane.\n"
             "Both columns of M lie on the same line — that's why.",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # § 13.  NULL SPACE — "inputs that get crushed to zero"
#
#     null_space(M)  =  { v   :   M · v = 0 }
#
# Hey-dude voice:
#     "Hey dude, what inputs do I send to the origin?  THOSE inputs are
#      where I 'lose information' — their directions get crushed."
#
# Cases:
#     - Invertible M      →  null space = {0}   (only the zero vector)
#     - Rank-deficient M  →  null space is a non-trivial subspace
#                            (whole line/plane of inputs that vanish)
#
# This is the DIRECTION of information loss.

# %% Null space of M_rank1
# M_rank1·v = 0 means: v[0] + 2·v[1] = 0
# So v[0] = -2·v[1]   →  null space = span of [-2, 1]
null_direction = np.array([-2, 1])

print(f"M_rank1 @ {tuple(null_direction)}     = {M_rank1 @ null_direction}")
print(f"M_rank1 @ {tuple(2 * null_direction)}   = {M_rank1 @ (2 * null_direction)}")
print(f"M_rank1 @ {tuple(-0.5 * null_direction)}    = {M_rank1 @ (-0.5 * null_direction)}")
print("→ any vector along the direction [-2, 1] gets crushed to the origin.")

# %% Visualize: null-space line collapses to a single point (origin)
ts = np.linspace(-2, 2, 30)
null_line_in = (ts[:, None] * null_direction).T

fig, axes = plt.subplots(1, 2, figsize=(11, 5))

ax = axes[0]
setup_ax_2d(ax, "Inputs along the null-space line\n(span of [-2, 1])", lim=4)
ax.scatter(null_line_in[0], null_line_in[1], s=25, c="#C44E52")
draw_arrow_2d(ax, null_direction, "#C44E52", "null direction", lw=2)

ax = axes[1]
null_line_out = M_rank1 @ null_line_in
setup_ax_2d(ax, "All of them squashed to the origin\n(under M_rank1)", lim=4)
ax.scatter(null_line_out[0], null_line_out[1], s=120, c="#C44E52", marker="x")
ax.text(0.3, 0.3, "← all 30 inputs collapsed here",
        fontsize=10, color="#C44E52")

plt.suptitle("Null space = directions M crushes  →  the directions of info loss",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# # § 14.  RANK — "how many output dimensions survive"
#
#     rank(M)  =  dimensionality of the column space
#              =  number of linearly independent columns
#              =  number of output dimensions that DON'T collapse
#
#     Full rank        →  rank = min(rows, cols)   (nothing collapses)
#     Rank-deficient   →  rank < min(rows, cols)   (some directions crushed)
#
# ## The big idea: RANK-NULLITY THEOREM
#
#     rank(M)  +  nullity(M)  =  input_dim
#
# Reading: every input dimension gets accounted for. Some survive (rank).
# The rest get crushed (nullity). Together they MUST sum to the input dim.
#
# Hey-dude voice:
#     "Hey dude, you handed me N input dimensions. Some of them I'll keep
#      (rank). The rest I'll crush to zero (nullity). They have to add up
#      to N — I can't make dimensions disappear into thin air."

# %% Tabulate rank and nullity for several matrices
print(f"{'Name':<32}{'shape':<10}{'rank':<8}{'nullity':<10}{'check':<10}")
print("-" * 70)

rank_examples = [
    ("Identity 2×2",          np.array([[1, 0], [0, 1]])),
    ("Rank-1 collapse 2×2",   np.array([[1, 2], [2, 4]])),
    ("Rotation 2×2",          np.array([[0, -1], [1, 0]])),
    ("3×2 tall (lift)",       np.array([[1, 0], [0, 1], [1, 2]])),
    ("2×3 wide (squash)",     np.array([[1, 0, 0], [0, 1, 0]])),
    ("Zero 2×2",              np.array([[0, 0], [0, 0]])),
]

for name, Mat in rank_examples:
    r = np.linalg.matrix_rank(Mat)
    rows, cols = Mat.shape
    nullity = cols - r
    check = f"{r}+{nullity}={cols} ✓"
    print(f"{name:<32}{rows}×{cols:<8}{r:<8}{nullity:<10}{check}")

# %% Why ML cares about rank
# - Low-rank approximations: SVD truncation, PCA, LoRA (Low-Rank Adaptation)
# - LoRA in particular: instead of fine-tuning a huge W matrix, factor the
#   update as ΔW ≈ A·B where A, B are skinny rectangular matrices of
#   low rank r. Trades expressivity for memory and speed.
# - Detecting bottlenecks: if a layer's effective rank is much lower than
#   its dimensionality, you have "dead" directions.

# %% [markdown]
# ---
# # § 15.  DUALITY — "every dot product is secretly a 1×n matrix"
#
# Mind-bender alert (chapter 9 of 3Blue1Brown).
#
# Take any vector b = [b₀, b₁] in 2D. The function "dot with b" takes a
# 2D vector and gives back a single number:
#
#     f(v)  =  b · v  =  b₀·v₀ + b₁·v₁         ← a scalar
#
# This function is LINEAR (it satisfies f(αv + βw) = αf(v) + βf(w)).
# By what we've learned, every linear function from R² to R must be
# representable as a 1×2 matrix.
#
# What's its matrix?  Look at this:
#
#     [ b₀  b₁ ] · [ v₀ ]  =  b₀·v₀ + b₁·v₁          ← same as b·v ✓
#                  [ v₁ ]
#
# The 1×2 matrix is literally b's components written as a ROW.
#
# So:
#     A vector b in R²  ↔  a linear function (1×2 matrix) from R² to R
#
# These are the SAME OBJECT, just written differently. THAT'S duality.
#
# Hey-dude voice:
#     "Hey dude, every arrow b in your space is ALSO a function that asks
#      'how much do you point in my direction?'. Same thing, two outfits."

# %% A vector as a vector  vs.  as a 1×2 matrix
b = np.array([2, 1])
v = np.array([3, 4])

print(f"b = {tuple(b)}")
print(f"v = {tuple(v)}")
print()
print(f"b · v        (dot product)          = {int(np.dot(b, v))}")
print(f"[b] @ v      (1×2 matrix × vector)  = {int((b.reshape(1, 2) @ v.reshape(2, 1))[0, 0])}")
print("Same answer. Same operation. Different framing.")

# %% Visualize duality: vector b ↔ linear map "dot with b"
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
setup_ax_2d(ax, "View 1: b is a VECTOR (arrow)", lim=4)
draw_arrow_2d(ax, b, "#C44E52", f"b = {tuple(b)}")

ax = axes[1]
setup_ax_2d(ax, 'View 2: b is a "how aligned?" function\n(level sets perpendicular to b)', lim=4)
# Draw level sets of b·v = constant for several values
levels = [-4, -2, 0, 2, 4]
xs2 = np.linspace(-4, 4, 50)
for lvl in levels:
    if b[1] != 0:
        ys2 = (lvl - b[0] * xs2) / b[1]
        ax.plot(xs2, ys2, "--", color="#C44E52", alpha=0.4, lw=1)
        # label one
        if lvl == 2:
            ax.text(xs2[10], ys2[10] + 0.3, f"b·v = {lvl}",
                    color="#C44E52", fontsize=9)
draw_arrow_2d(ax, b, "#C44E52", "b")

plt.suptitle("DUALITY: same b — as an arrow, or as a function that measures alignment",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% Why ML cares about duality
# - Every linear classifier's weight vector IS a direction. The class score
#   is just "how much does the input point in MY direction?"
# - Q · Kᵀ in attention: each query is a dual vector asking each key,
#   "how aligned are you with me?"
# - Gradient vectors are dual to parameter changes: ∇L · Δθ measures
#   how much the loss responds to a parameter shift.

# %% [markdown]
# ---
# # § 16.  EIGENVECTORS AND EIGENVALUES — "directions M only stretches"
#
# Most vectors get pushed around when M is applied — they change direction.
# But SOME special vectors stay on their own line; M only SCALES them:
#
#     M · v  =  λ · v
#
#     v  =  an EIGENVECTOR of M
#     λ  =  the corresponding EIGENVALUE
#
# Hey-dude voice:
#     "Hey dude, most arrows I'd push around. But for these SPECIAL arrows,
#      I'll only stretch or shrink them — same direction, just a new length.
#      That length factor is the eigenvalue."
#
# Meaning of λ:
#     λ = 2     →  this arrow gets doubled (same direction)
#     λ = 0.5   →  this arrow gets halved
#     λ = -1    →  this arrow flips (same line, opposite way)
#     λ = 0     →  this arrow gets crushed to zero  (it's in the null space!)
#     |λ| = 1   →  this arrow's length is preserved

# %% Find eigenvectors of a 2×2 matrix
Me = np.array([[2, 1],
               [0, 3]])

eigenvalues, eigenvectors = np.linalg.eig(Me)

print(f"M =\n{Me}")
print(f"\nEigenvalues: {eigenvalues}")
print(f"Eigenvectors (each is a COLUMN):\n{eigenvectors.round(3)}")

# Verify  M @ v ≈ λ · v
for i in range(2):
    v_eig = eigenvectors[:, i]
    lam   = eigenvalues[i]
    print(f"\nEigenvector {i}:  v = {v_eig.round(3)},  λ = {lam:.2f}")
    print(f"  M @ v = {(Me @ v_eig).round(3)}")
    print(f"  λ · v = {(lam * v_eig).round(3)}    ← same ✓")

# %% Visualize: most arrows rotate; eigenvectors only stretch
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

n_arrows = 16
angles = np.linspace(0, 2 * np.pi, n_arrows, endpoint=False)
unit_arrows = np.stack([np.cos(angles), np.sin(angles)])

ax = axes[0]
setup_ax_2d(ax, "Before M\n(unit arrows + eigenvectors)", lim=3)
for k in range(n_arrows):
    draw_arrow_2d(ax, unit_arrows[:, k], "#BBBBBB", lw=1.2)
draw_arrow_2d(ax, eigenvectors[:, 0], "#C44E52", "eigenvec 0", lw=2.5)
draw_arrow_2d(ax, eigenvectors[:, 1], "#55A868", "eigenvec 1", lw=2.5)

ax = axes[1]
setup_ax_2d(ax, "After M\n(eigenvectors STAY on their own lines)", lim=6)
after = Me @ unit_arrows
for k in range(n_arrows):
    draw_arrow_2d(ax, after[:, k], "#BBBBBB", lw=1.2)
draw_arrow_2d(ax, Me @ eigenvectors[:, 0], "#C44E52",
              f"λ₀={eigenvalues[0]:.1f}·eig 0", lw=2.5)
draw_arrow_2d(ax, Me @ eigenvectors[:, 1], "#55A868",
              f"λ₁={eigenvalues[1]:.1f}·eig 1", lw=2.5)

plt.suptitle("Eigenvectors: special directions that don't rotate under M",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% Special case: rotations have NO real eigenvectors
# Because every direction rotates — none stays on its own line.
R = np.array([[0, -1], [1, 0]])         # 90° rotation
ev_R, _ = np.linalg.eig(R)
print(f"Rotation 90°  →  eigenvalues = {ev_R}")
print("→ complex numbers! Real rotations have no real eigenvectors in 2D.")
print("  (because there's no direction that stays put under a true rotation)")

# %% Why ML cares about eigenvectors
# - PCA: the top eigenvectors of the data's covariance matrix are the
#   PRINCIPAL COMPONENTS — the directions of maximum variance.
# - SVD: generalizes eigen-decomposition to non-square matrices.
#   W = U Σ Vᵀ   →   any linear layer is "rotate, scale, rotate."
# - Power method: repeatedly applying M to a random vector converges to
#   the dominant eigenvector — this is the trick behind PageRank.
# - Stability of training dynamics depends on eigenvalues of the Hessian.

# %% [markdown]
# ---
# # § 17.  ABSTRACT VECTOR SPACES — "vectors are anything that adds and scales"
#
# Surprise: a "vector" doesn't have to be an arrow.
#
# A VECTOR SPACE is any set V with two operations:
#     (1) addition           u + v          (gives another element of V)
#     (2) scalar mult         c · v          (gives another element of V)
#
# satisfying 8 axioms (boiling down to: addition and scalar multiplication
# behave "normally" — associative, commutative, distributive, zero exists,
# inverses exist).
#
# Examples — ALL of these are vector spaces:
#     - Arrows in Rⁿ                   (the obvious case)
#     - Polynomials                    (e.g., 2x² + 3x + 1)
#     - Functions                      (e.g., sin(x), x², …)
#     - 2×2 matrices                   (M + N is a matrix; 2·M is a matrix)
#     - Audio signals, images          (you can add and scale them)
#     - In ML:
#         • Token embeddings           (768-D vectors)
#         • Image patch embeddings     (768-D vectors)
#         • Hidden states              (4096-D in LLaMA)
#         • Gradients                  (one number per parameter,
#                                       sometimes trillions of dimensions)
#
# The payoff: EVERY theorem about vector spaces (linear combinations,
# bases, span, linear maps, eigenvectors, …) applies to ALL of these.
#
# Hey-dude voice:
#     "Hey dude, linear algebra is just a language for talking about
#      'things that add and scale.' Once you see that polynomials and
#      audio signals both ADD and SCALE, you can throw EVERY tool we've
#      learned at them too."

# %% Polynomials as vectors — a concrete example
# A polynomial  p(x) = a + b·x + c·x²   ↔   the vector (a, b, c).
# Operations on polynomials map exactly to operations on vectors.

p = np.array([1, 2, 3])     # represents 1 + 2x + 3x²
q = np.array([0, 1, -1])    # represents 0 + 1x - 1x²

print("Polynomials as vectors:")
print(f"  p(x) = 1 + 2x + 3x²   →   vector {tuple(p)}")
print(f"  q(x) = 0 + 1x - 1x²   →   vector {tuple(q)}")
print()
print(f"p + q   →   vector {tuple(p + q)}")
print(f"   That's the polynomial (1+0) + (2+1)x + (3-1)x² = 1 + 3x + 2x²")
print(f"   ↑ correct! Polynomials add like vectors.")
print()
print(f"2 · p   →   vector {tuple(2 * p)}")
print(f"   That's the polynomial 2 + 4x + 6x²")
print(f"   ↑ correct! Polynomials scale like vectors.")
print()
print("→ The space of degree-2 polynomials IS a 3-dimensional vector space.")
print("  Its 'basis' is {1, x, x²}. Each polynomial is a linear combination.")

# %% Functions as vectors — even more abstract
# sin and cos are 'vectors' in the function space C[ℝ→ℝ].
# Adding them gives another function: f(x) = sin(x) + cos(x).
xs_plot = np.linspace(0, 2 * np.pi, 200)

fig, ax = plt.subplots(1, 1, figsize=(9, 5))
ax.plot(xs_plot, np.sin(xs_plot),                        label="sin(x)",       color="#4C72B0", lw=2)
ax.plot(xs_plot, np.cos(xs_plot),                        label="cos(x)",       color="#55A868", lw=2)
ax.plot(xs_plot, np.sin(xs_plot) + np.cos(xs_plot),      label="sin + cos",    color="#C44E52", lw=2.5)
ax.plot(xs_plot, 2 * np.sin(xs_plot),                    label="2·sin(x)",     color="#DD8452", lw=1.5, linestyle="--")
ax.axhline(0, color="black", lw=0.5); ax.grid(True, alpha=0.3)
ax.set_title("Functions as vectors:\n"
             "you can ADD them (sin + cos) and SCALE them (2·sin)\n"
             "→ they form a vector space too!")
ax.legend()
plt.show()

# %% [markdown]
# ## The big lesson
#
# Once you see this, ML stops feeling magical:
#
#   - **Image patch ⟶ vector**: split the image into 16×16 patches, flatten
#     each one. Now you have a vector. ViTs and CLIP work on these.
#
#   - **Text token ⟶ vector**: look up its embedding row. Now it's a vector.
#     Transformers work on these.
#
#   - **Audio frame ⟶ vector**: take a 25 ms window, compute mel features.
#     Now it's a vector. Whisper works on these.
#
#   - **Gradient ⟶ vector**: one number per model parameter. SGD updates work
#     on these.
#
# All of them: things that add and scale. So all of them get the FULL
# linear-algebra treatment — basis vectors, linear transformations, dot
# products, eigenvectors, the works.
#
# **Linear algebra is the universal language of ML.** Now you speak it.

# %% [markdown]
# ---
# ## Summary
#
# | Concept                       | Visual intuition                                | "Hey dude" version                                       | ML payoff                                            |
# |-------------------------------|-------------------------------------------------|-----------------------------------------------------------|------------------------------------------------------|
# | Vector                        | Arrow from origin to a point                    | "A is a *recipe* for mixing basis vectors"               | Embeddings (token, image patch, …)                   |
# | Standard basis (î, ĵ)         | The default arrows along each axis              | "Your starting ingredients"                              | Identity layer                                       |
# | Linear combination            | Scaled basis vectors stacked tip-to-tail        | "v's components ARE the scalars"                         | Every layer output is one of these                   |
# | Matrix (square N×N)           | Re-define basis WITHIN an N-dim room            | "Hey dude, from now on, your basis is..."               | Attention Q/K/V heads, rotations, RoPE               |
# | Columns of M                  | Where each basis vector LANDS                   | "Look at col 0: that's your new î"                      | "What does this layer do to î, ĵ, …?"               |
# | M @ v                         | Recipe v applied to M's columns                 | "Take v's components, mix M's new basis"                | Computing a single layer's output                    |
# | M @ K @ v                     | Sequentially re-represent v in K, then in M     | "K rewrites your basis; M rewrites it again"            | Stacked layers in a network                          |
# | Order matters (M·K ≠ K·M)     | Different transformation order = different shape | "Re-representing in different order = different result" | Why layer order matters in a network                 |
# | Matrix (tall, M×N, M>N)       | LIFT — new basis vectors live in a bigger room  | "Your new î, ĵ are higher-dim vectors"                   | `nn.Linear(small → big)`, embedding tables           |
# | Matrix (wide, M×N, M<N)       | SQUASH — new basis vectors are smaller          | "Your 3+ new basis vectors all live in 2D"              | Projection heads, classifier outputs, bottleneck     |
# | **Dot product (§ 9)**         | Signed agreement between two arrows             | "How much do you point my way?"                          | Attention scores, CLIP similarity, retrieval         |
# | **Determinant (§ 10)**        | Signed area-scaling factor                      | "What's the new unit-square area?"                       | Jacobians (normalizing flows); detecting collapses    |
# | **Inverse (§ 11)**            | Undo a transformation                           | "Rewind me — if I didn't crush anything"                 | Numerical stability, conditioning                    |
# | **Column space (§ 12)**       | Set of all reachable outputs                    | "Universe of outputs I can hit"                          | Feature space of a layer; representational capacity  |
# | **Null space (§ 13)**         | Inputs that get crushed to zero                 | "Directions I send to the origin"                        | Information loss; what the layer can't see           |
# | **Rank (§ 14)**               | Dim of column space (= surviving directions)    | "How many directions I keep"                             | Low-rank approximations: SVD, PCA, **LoRA**          |
# | **Duality (§ 15)**            | Vector ↔ linear function to R                   | "Every arrow is also a 'how aligned?' detector"          | Linear classifiers, attention queries, gradients     |
# | **Eigenvectors (§ 16)**       | Directions that only stretch, never rotate      | "Special arrows I only scale"                            | PCA, SVD, spectral methods, training stability       |
# | **Abstract vector space (§ 17)** | Anything that adds and scales                | "It's all vectors, dude"                                 | Tokens, audio, gradients — ALL get linear-algebra'd  |
#
# None of this is axiomatic. It's all spatial and visual.
# After 3Blue1Brown plus these cells, this should feel like counting fingers.

# %%
