# %% [markdown]
# # OPTIONAL: GANs — The Adversarial Game
#
# Not needed for your thesis. But it's a beautiful concept.
#
# Core idea in one sentence:
#   Two networks play a minimax game — one tries to FAKE real data,
#   the other tries to SPOT fakes. They both get better by playing.
#
# Analogy: a forger and a detective.
#   The forger paints fakes. The detective tries to spot them.
#   Every round:
#     - Detective trains on real + fake paintings → gets better at catching fakes
#     - Forger sees which fakes got caught → gets better at fooling the detective
#   After enough rounds, the forger makes fakes the detective can't distinguish.
#
# We'll train a GAN on 2D data you can SEE (points in a ring).

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# %% [markdown]
# ## 1. The real data: points on a ring
#
# Our "real distribution" is 2D points sampled from a ring (radius ~1, narrow band).
# The generator's job: produce points that also look ring-shaped.

# %%
def sample_real(n=512):
    """Points uniformly sampled on a unit ring with a bit of radial noise."""
    angle = torch.rand(n) * 2 * np.pi
    radius = 1.0 + torch.randn(n) * 0.05
    x = radius * torch.cos(angle)
    y = radius * torch.sin(angle)
    return torch.stack([x, y], dim=1)

real_samples = sample_real(1000)
plt.scatter(real_samples[:, 0], real_samples[:, 1], s=5, alpha=0.5, color='blue')
plt.title("REAL data — points on a ring\nOur generator must learn to produce this shape")
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 2. The two players

# %%
class Generator(nn.Module):
    """Maps random noise (2D) → a 2D point. Tries to produce ring-shaped points."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    """Takes a 2D point → outputs probability that it's REAL."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

G = Generator()
D = Discriminator()

optim_G = torch.optim.Adam(G.parameters(), lr=1e-3)
optim_D = torch.optim.Adam(D.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()

# %% [markdown]
# ## 3. The adversarial training loop
#
# Each step:
#   1. Train discriminator: show it real + fake, teach it to separate them
#   2. Train generator: produce fakes, try to fool the now-trained discriminator

# %%
snapshots = []
losses_g, losses_d = [], []

for step in range(4000):
    real = sample_real(256)
    noise = torch.randn(256, 2)
    fake = G(noise)

    # --- Train discriminator ---
    D_real = D(real)
    D_fake = D(fake.detach())  # .detach() — don't backprop into G right now
    loss_D = loss_fn(D_real, torch.ones_like(D_real)) + \
             loss_fn(D_fake, torch.zeros_like(D_fake))
    optim_D.zero_grad(); loss_D.backward(); optim_D.step()

    # --- Train generator ---
    # Generator's goal: make D think fakes are REAL
    noise = torch.randn(256, 2)
    fake = G(noise)
    D_fake = D(fake)
    loss_G = loss_fn(D_fake, torch.ones_like(D_fake))  # we WANT D to say "real"
    optim_G.zero_grad(); loss_G.backward(); optim_G.step()

    losses_g.append(loss_G.item())
    losses_d.append(loss_D.item())

    # Capture the generator's output at milestones
    if step in [0, 100, 500, 1500, 3999]:
        with torch.no_grad():
            samples = G(torch.randn(1000, 2)).numpy()
        snapshots.append((step, samples))

print(f"Done. Final D loss: {loss_D.item():.3f}, G loss: {loss_G.item():.3f}")

# %% [markdown]
# ## 4. Watch the forger learn

# %%
fig, axes = plt.subplots(1, len(snapshots), figsize=(18, 3.5))
for ax, (step, samples) in zip(axes, snapshots):
    ax.scatter(real_samples[:, 0], real_samples[:, 1], s=3, alpha=0.3, color='blue', label='real')
    ax.scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5, color='red', label='generated')
    ax.set_title(f"step {step}")
    ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
    ax.axis('equal')
    ax.legend(loc='upper right', fontsize=8)

plt.suptitle("Generator starts from random noise → learns to produce the ring", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Plot the adversarial losses
#
# In a healthy GAN, neither loss goes to zero — they oscillate.
# If D goes to 0 loss: D is too strong, G gets no signal, training collapses.
# If G goes to 0 loss: G is fooling D completely, but often the generated data
# is low-quality (mode collapse — G produces one narrow point it knows works).

# %%
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses_d, label='Discriminator loss', alpha=0.7)
ax.plot(losses_g, label='Generator loss', alpha=0.7)
ax.set_xlabel('step')
ax.set_ylabel('loss')
ax.set_title('Adversarial losses — the game between G and D')
ax.legend()
ax.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 6. Why GANs matter (historically)
#
# 2014: Goodfellow introduces GANs  → adversarial training changes everything
# 2015-2018: GANs DOMINATE image generation (StyleGAN, BigGAN)
# 2020-2022: Diffusion models start beating GANs on image quality
# 2022+: Diffusion wins for big models; GANs still used for niche tasks
#
# The BIG IDEA of GANs — "use one network as a teacher for another" — shows up everywhere:
#   - Self-supervised learning (DINO, self-distillation)
#   - RLHF (reward model guides the LLM)
#   - Adversarial attacks / robustness research

# %% [markdown]
# ## 7. Why your thesis doesn't use them
#
# Every VLM in your thesis is a DECODER-ONLY TRANSFORMER doing TEXT GENERATION.
# No image generation. No adversarial training. No discriminator.
#
# You now understand what GANs are (good for conversations and survey writing),
# but you will not write a line of GAN code for your thesis. Move on with pride.

# %% [markdown]
# ## Summary — the "bent fingers" picture
#
# Two networks, one table:
#
# ```
# ┌──────────┐                        ┌──────────────┐
# │ noise z  │ ──── Generator ─────→  │ fake image   │ ────┐
# └──────────┘                        └──────────────┘     │
#                                                           ▼
#                                               ┌───────────────────────┐
#                                               │ Discriminator         │
#                                               │ "is this real or fake?"│
#                                               └───────────────────────┘
#                                                           ▲
# ┌──────────┐                                               │
# │ real img │ ─────────────────────────────────────────────┘
# └──────────┘
# ```
#
# They train in opposite directions. The game produces realistic fakes.
# That's the whole story.
