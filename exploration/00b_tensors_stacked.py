# %% [markdown]
# # Day 0 (part B): Tensors = Stacked Cards. That's it.
#
# Physical analogy you can close your eyes and see:
#   2D matrix  →  ONE playing card
#   3D tensor  →  a DECK of cards (cards stacked)
#   4D tensor  →  a BOX of decks (several decks side by side)
#   5D tensor  →  a SHELF of boxes
#
# Run each cell. Look at the picture. Then come back and look at the shape.
# After this, shape = physical object in your head.

# %%
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1. A 2D matrix = ONE card
#
# Pick up one playing card. Write a number on every square of a grid on it.
# That's a 2D matrix.

# %%
# A single 5x5 "card"
card = np.random.rand(5, 5)
print(f"Shape: {card.shape}  ← one card, 5 rows × 5 cols")

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(card, cmap='viridis')
ax.set_title(f"ONE CARD — shape {card.shape}")
for i in range(5):
    for j in range(5):
        ax.text(j, i, f"{card[i,j]:.1f}", ha='center', va='center', color='white', fontsize=9)
plt.show()

# %% [markdown]
# ## 2. A 3D tensor = a DECK of cards (stack them vertically)
#
# Now take 4 cards. Stack them. Each card is 5x5.
# Shape: (4, 5, 5) = "4 cards, each 5 rows × 5 cols"

# %%
deck = np.random.rand(4, 5, 5)
print(f"Shape: {deck.shape}  ← a deck of 4 cards")

# Visualize as stacked cards with offset (you can SEE the depth)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

for card_idx in range(4):
    # Draw each card as a layer in 3D space
    x = np.arange(5)
    y = np.arange(5)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, card_idx, dtype=float)  # each card at a different height

    # Color the card based on its values
    colors = plt.colormaps['viridis'](deck[card_idx])
    ax.plot_surface(X, Y, Z, facecolors=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.text(-0.5, 2, card_idx, f"card {card_idx}", fontsize=10, color='black')

ax.set_xlabel('cols (5)')
ax.set_ylabel('rows (5)')
ax.set_zlabel('card #')
ax.set_title(f"DECK OF CARDS — shape {deck.shape}\n"
             f"4 cards stacked, each card is 5×5")
plt.show()

# %% [markdown]
# ## 3. The reality check: this is exactly what an RGB image is
#
# An RGB image is a 3D tensor — a deck of 3 cards:
#   card 0 = Red values for every pixel
#   card 1 = Green values for every pixel
#   card 2 = Blue values for every pixel
#
# Shape: (3, H, W) = "a deck of 3 cards, each card is H × W"

# %%
# Make a fake RGB image — 3 cards of shape (32, 32)
rgb_image = np.zeros((3, 32, 32))
# Red card: gradient left to right
rgb_image[0] = np.linspace(0, 1, 32).reshape(1, -1).repeat(32, axis=0)
# Green card: gradient top to bottom
rgb_image[1] = np.linspace(0, 1, 32).reshape(-1, 1).repeat(32, axis=1)
# Blue card: diagonal
rgb_image[2] = np.eye(32)

print(f"Shape: {rgb_image.shape}  ← 3 cards (R, G, B), each 32x32")

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

# Show each "card" separately
for i, (ax, color_name, cmap) in enumerate(zip(axes[:3], ['Red', 'Green', 'Blue'], ['Reds', 'Greens', 'Blues'])):
    ax.imshow(rgb_image[i], cmap=cmap)
    ax.set_title(f"Card {i}: {color_name} channel\nshape {rgb_image[i].shape}")
    ax.axis('off')

# Show the cards combined (what you actually see)
axes[3].imshow(rgb_image.transpose(1, 2, 0))
axes[3].set_title(f"All 3 cards stacked\n= the RGB image you see\nshape {rgb_image.shape}")
axes[3].axis('off')

plt.suptitle("A color image IS a deck of 3 cards (R, G, B) — that's all 3D means here")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. A 4D tensor = a BOX of decks
#
# Now imagine: 32 RGB images in a batch.
# Shape: (32, 3, H, W) = "a box holding 32 decks, each deck has 3 cards"

# %%
batch = np.random.rand(8, 3, 16, 16)  # small: 8 images for visualization
print(f"Shape: {batch.shape}  ← a box with 8 decks, each deck has 3 cards of 16×16")

# Visualize as 8 RGB images side by side (the box)
fig, axes = plt.subplots(1, 8, figsize=(16, 3))
for i in range(8):
    axes[i].imshow(batch[i].transpose(1, 2, 0))
    axes[i].set_title(f"deck {i}")
    axes[i].axis('off')

plt.suptitle(f"BOX OF DECKS — shape {batch.shape}\n"
             f"8 images (decks), each is 3 color channels (cards), each 16×16\n"
             f"In ML this is 'a batch of 8 RGB images'", fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. The big insight: transformations are STILL 2D
#
# When you apply a neural network layer, the weight matrix is 2D.
# The 3D/4D shape of your data just means "apply this 2D transform to ALL the cards at once."

# %%
# A batch of tokens: 32 sentences, 10 tokens each, each token is a 4-dim vector
tokens = np.random.rand(32, 10, 4)
print(f"Input shape:  {tokens.shape}  ← box of 32 decks, each deck has 10 cards, each card is 1×4")

# A 2D weight matrix — same as always, just a grid
W = np.random.rand(4, 8)
print(f"Weight shape: {W.shape}  ← ONE 2D matrix (one card)")

# Matrix multiplication: apply W to EVERY token vector
output = tokens @ W
print(f"Output shape: {output.shape}  ← same box shape, but each card's vector is now 1×8")

print()
print("What happened in physical terms:")
print(f"  You had 32 × 10 = 320 token vectors (cards)")
print(f"  You applied the SAME 2D transformation W to each one")
print(f"  Output: 320 new vectors, same shape arrangement")
print()
print("The 'higher dim' is just how many things to transform in parallel.")
print("The ACTUAL transformation is always 2D. Always. That's the punchline.")

# %% [markdown]
# ## Summary — close your eyes and see this
#
# | Shape | Physical picture |
# |-------|-----------------|
# | `(5,)` | A row of 5 coins |
# | `(5, 5)` | One playing card, 5×5 grid |
# | `(4, 5, 5)` | A deck of 4 cards |
# | `(3, 224, 224)` | A deck of 3 cards = one RGB image |
# | `(32, 3, 224, 224)` | A box of 32 decks = batch of 32 RGB images |
# | `(32, 12, 512, 512)` | A box of 32 stacks, each stack is 12 cards = attention weights for a batch |
#
# Transformations (weight matrices) are STILL 2D cards.
# Higher dims = "how many things to transform in parallel."

# %%
