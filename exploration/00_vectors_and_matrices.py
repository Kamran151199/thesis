# %% [markdown]
# # Day 0: Vectors and Matrices — The Visual Intuition
#
# Before touching any ML, get this into your bones.
# Run each cell, look at the plots, change the numbers, re-run.
#
# WATCH FIRST (or alongside):
# 3Blue1Brown "Essence of Linear Algebra" — Episodes 1-4:
#   1. Vectors: https://www.youtube.com/watch?v=fNk_zzaMoSs
#   2. Linear combinations: https://www.youtube.com/watch?v=k7RM-ot2NWY
#   3. Matrix as transformation: https://www.youtube.com/watch?v=kYB8IZa5AuE
#   4. Matrix multiplication: https://www.youtube.com/watch?v=XkY2DOUCWMU

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. What IS a vector?
#
# A vector is an arrow from the origin to a point.
# v = [3, 2] means: go 3 right, 2 up.
# That's it. Not abstract — it's a location with direction.

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_xlim(-1, 5)
ax.set_ylim(-1, 5)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# A vector is just an arrow
v = np.array([3, 2])
ax.annotate('', xy=v, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(v[0] + 0.1, v[1] + 0.1, f'v = [{v[0]}, {v[1]}]', fontsize=12, color='blue')

# Another vector
w = np.array([1, 4])
ax.annotate('', xy=w, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(w[0] + 0.1, w[1] + 0.1, f'w = [{w[0]}, {w[1]}]', fontsize=12, color='red')

ax.set_title('Vectors = arrows from origin\nChange the numbers above and re-run!')
plt.show()

# %% [markdown]
# ## 2. Vector addition: "bent fingers" but with arrows
#
# v + w = put w's tail at v's tip. Where does it end up?
# Exactly like: 2 fingers + 2 fingers = 4 fingers
# But now: [3,2] + [1,4] = [4,6]

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax.set_xlim(-1, 7)
ax.set_ylim(-1, 7)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

v = np.array([3, 2])
w = np.array([1, 4])
result = v + w

# Draw v (blue)
ax.annotate('', xy=v, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(1.5, 0.5, 'v', fontsize=14, color='blue', fontweight='bold')

# Draw w starting from tip of v (red)
ax.annotate('', xy=result, xytext=v,
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(v[0] - 0.5, v[1] + 1.5, 'w', fontsize=14, color='red', fontweight='bold')

# Draw result (green)
ax.annotate('', xy=result, xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
ax.text(result[0] + 0.1, result[1] + 0.1,
        f'v+w = [{result[0]}, {result[1]}]', fontsize=12, color='green', fontweight='bold')

ax.set_title('Vector addition = chain the arrows\nSame intuition as counting fingers!')
plt.show()

# %% [markdown]
# ## 3. The dot product: "how much do two vectors agree?"
#
# dot(v, w) = v[0]*w[0] + v[1]*w[1]
#
# If both point the same way → large positive number
# If perpendicular → zero
# If opposite → large negative number
#
# THIS IS THE CORE OF ATTENTION IN TRANSFORMERS.
# "How similar is token A to token B?" = dot product of their vectors.

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

cases = [
    ("Same direction", np.array([2, 1]), np.array([3, 1.5])),
    ("Perpendicular", np.array([2, 1]), np.array([-1, 2])),
    ("Opposite", np.array([2, 1]), np.array([-2, -1])),
]

for ax, (title, a, b) in zip(axes, cases):
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    dot = np.dot(a, b)

    ax.annotate('', xy=a, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax.annotate('', xy=b, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax.set_title(f'{title}\ndot = {dot:.1f}', fontsize=12)

plt.suptitle('Dot product = "how much do they agree?"\nThis is literally what attention computes!',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. A MATRIX IS A TRANSFORMATION — Watch the grid warp
#
# This is the key insight. A matrix isn't numbers in a box.
# It's a MACHINE that takes in every point in space and moves it somewhere new.
#
# Watch what happens to the grid when you multiply by different matrices.

# %%
def plot_transformation(ax, matrix, title):
    """Draw a grid before and after matrix transformation."""
    # Create a grid of points
    t = np.linspace(-2, 2, 11)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

    # Draw original grid lines (light gray)
    for val in t:
        # Horizontal lines
        xs = np.array([t[0], t[-1]])
        ys = np.array([val, val])
        ax.plot(xs, ys, 'lightgray', linewidth=0.5)
        # Vertical lines
        ax.plot([val, val], [t[0], t[-1]], 'lightgray', linewidth=0.5)

    # Transform grid lines
    for val in t:
        # Transform horizontal line
        points = np.array([[x, val] for x in t])
        transformed = (matrix @ points.T).T
        ax.plot(transformed[:, 0], transformed[:, 1], 'blue', linewidth=0.8, alpha=0.6)

        # Transform vertical line
        points = np.array([[val, y] for y in t])
        transformed = (matrix @ points.T).T
        ax.plot(transformed[:, 0], transformed[:, 1], 'red', linewidth=0.8, alpha=0.6)

    # Show where the basis vectors land
    e1 = matrix @ np.array([1, 0])
    e2 = matrix @ np.array([0, 1])
    ax.annotate('', xy=e1, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2.5))
    ax.annotate('', xy=e2, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

    ax.plot(0, 0, 'ko', markersize=5)
    ax.set_title(title, fontsize=11)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Each matrix does something different to space
matrices = [
    (np.array([[1, 0], [0, 1]]), "Identity\n(does nothing)"),
    (np.array([[2, 0], [0, 2]]), "Scale 2x\n(zoom in)"),
    (np.array([[0, -1], [1, 0]]), "Rotate 90°\n(rotate counterclockwise)"),
    (np.array([[1, 1], [0, 1]]), "Shear\n(slant sideways)"),
    (np.array([[0.5, 0], [0, 2]]), "Stretch\n(squash x, stretch y)"),
    (np.array([[-1, 0], [0, 1]]), "Reflect\n(mirror across y-axis)"),
]

for ax, (mat, title) in zip(axes.flat, matrices):
    plot_transformation(ax, mat, f"{title}\n{mat.tolist()}")

plt.suptitle('EACH MATRIX IS A DIFFERENT TRANSFORMATION OF SPACE\n'
             'Blue lines = transformed horizontals, Red lines = transformed verticals\n'
             'Change the matrix numbers and re-run!',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Matrix multiplication = chaining transformations
#
# If A rotates and B scales, then (B @ A) rotates THEN scales.
# Matrix multiplication is just "do transformation A, then do transformation B."
#
# In ML: layer 1 transforms, then layer 2 transforms the result.
# A neural network is literally a chain of matrix multiplications
# (with non-linearities in between).

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

A = np.array([[0, -1], [1, 0]])   # rotate 90°
B = np.array([[2, 0], [0, 0.5]])  # scale x by 2, y by 0.5
C = B @ A                          # rotate THEN scale

plot_transformation(axes[0], A, f"Step 1: Rotate 90°\nA = {A.tolist()}")
plot_transformation(axes[1], B, f"Step 2: Scale\nB = {B.tolist()}")
plot_transformation(axes[2], C, f"Combined: B @ A\n= {C.tolist()}")

plt.suptitle('Matrix multiplication = chaining transformations\n'
             'Neural network layers do exactly this!',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Higher dimensions — where ML lives
#
# Everything above works in 2D (you can see it).
# In ML, vectors have 768 or 4096 dimensions — you can't visualize them directly.
# But THE SAME INTUITION HOLDS:
#
# - A vector in 768D is a "point" in 768-dimensional space
# - A matrix in 768×768 is a "transformation" of that 768D space
# - Dot product in 768D still means "how similar are these two vectors"
#
# When a transformer computes attention:
#   score = query_vector · key_vector  (dot product in 768D)
#   = "how much should this token pay attention to that token?"
#
# When a neural network layer computes:
#   output = W @ input + bias  (matrix multiplication in 768D)
#   = "transform this representation into a new one"

# %%
# Let's prove the intuition scales: dot product similarity in high dimensions
np.random.seed(42)

# Imagine these are word embeddings in a VLM
king = np.random.randn(768)
queen = king + np.random.randn(768) * 0.3    # similar to king
cat = np.random.randn(768)                     # totally different

# Dot product (well, cosine similarity — normalized dot product)
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Cosine similarity (same intuition as 2D, but in 768D):")
print(f"  king vs queen: {cosine_sim(king, queen):.3f}  ← similar vectors, high score")
print(f"  king vs cat:   {cosine_sim(king, cat):.3f}  ← different vectors, low score")
print()
print("THIS is what CLIP computes between image and text embeddings.")
print("THIS is what attention computes between tokens.")
print("Same dot product. Same intuition. Just more dimensions.")

# %% [markdown]
# ## Summary
#
# | Concept | Visual intuition | Where it shows up in ML |
# |---------|-----------------|------------------------|
# | Vector | Arrow pointing somewhere | Embeddings (word, image patch, token) |
# | Dot product | "How much do two arrows agree?" | Attention scores, similarity |
# | Matrix | Transformation of space | Neural network weight layers |
# | Matrix multiply | Chain two transformations | Stacking neural network layers |
#
# NONE of this is axiomatic. It's all spatial and visual.
# After watching 3Blue1Brown episodes 1-4, this will feel like counting fingers.
