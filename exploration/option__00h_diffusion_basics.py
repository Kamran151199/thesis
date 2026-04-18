# %% [markdown]
# # OPTIONAL: Diffusion — Learn to Denoise, Then Run Backwards
#
# Not needed for your thesis. But it's how Stable Diffusion, DALL-E 3, and
# Flux all generate images. Worth understanding.
#
# Core idea in one sentence:
#   Gradually destroy data with noise (forward process — no learning needed).
#   Train a network to REVERSE one step of that destruction.
#   At generation time: start from pure noise, reverse step-by-step → data.
#
# Analogy: imagine smudging a photo until it's static.
#   The forward process is trivial — just keep adding noise.
#   The hard part: learning to UN-smudge — to denoise.
#   If you can un-smudge noise by one small step, you can run it in a loop
#   from pure noise all the way back to a clean image.
#
# We'll do this on 2D data you can SEE — a spiral.

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# %% [markdown]
# ## 1. The target data: a 2D spiral
#
# Our "real distribution" is a spiral. If diffusion works, we can start from
# random noise and produce new points on the spiral.

# %%
def make_spiral(n=1000):
    t = torch.rand(n) * 4 * np.pi
    r = t / (4 * np.pi) * 2  # radius grows with angle
    x = r * torch.cos(t)
    y = r * torch.sin(t)
    return torch.stack([x, y], dim=1)

real_data = make_spiral(2000)
plt.scatter(real_data[:, 0], real_data[:, 1], s=3, alpha=0.5, color='blue')
plt.title("REAL data — a spiral\nOur diffusion model will learn to produce this shape")
plt.axis('equal')
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 2. The forward process — add noise gradually
#
# Formula (simplified DDPM):
#     x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
#
# Where alpha_bar_t decreases from ~1 (t=0, clean data) to ~0 (t=T, pure noise).
# No neural network here — just math. You could write this with 2 lines.

# %%
T = 100  # total diffusion steps
betas = torch.linspace(0.0001, 0.02, T)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # α_bar_t = Π_{s=1..t} α_s

def forward_diffuse(x_0, t):
    """Apply t steps of noise to x_0."""
    noise = torch.randn_like(x_0)
    a_bar = alpha_bars[t].unsqueeze(-1)  # (B, 1)
    return torch.sqrt(a_bar) * x_0 + torch.sqrt(1 - a_bar) * noise, noise

# Visualize the forward process: same spiral at different noise levels
fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
visualize_ts = [0, 10, 30, 60, 90, 99]
for ax, t in zip(axes, visualize_ts):
    t_tensor = torch.full((real_data.shape[0],), t, dtype=torch.long)
    x_t, _ = forward_diffuse(real_data, t_tensor)
    ax.scatter(x_t[:, 0], x_t[:, 1], s=2, alpha=0.4)
    ax.set_title(f"t = {t}")
    ax.axis('equal')
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)

plt.suptitle("Forward process — add noise, spiral dissolves into noise\n"
             "NO neural network needed. Just math.", fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. The network — learn to predict the noise that was added
#
# Given a noisy x_t and the timestep t, predict the noise that was added.
# If we can predict the noise, we can subtract it → one step of denoising.
#
# Input:  (x_t, t) — 2D point + timestep index
# Output: the noise (2D)
#
# NOTE: the model doesn't predict the CLEAN data directly — it predicts the noise.
# This reparameterization makes training stable. (Ho et al. 2020, DDPM.)

# %%
class DenoiseNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        # Embed the timestep so the model knows WHERE in the noise schedule we are
        self.t_embed = nn.Embedding(T, hidden)
        self.net = nn.Sequential(
            nn.Linear(2 + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x, t):
        t_emb = self.t_embed(t)
        inp = torch.cat([x, t_emb], dim=-1)
        return self.net(inp)

model = DenoiseNet()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% [markdown]
# ## 4. Training — MSE between predicted noise and actual noise

# %%
losses = []
for step in range(3000):
    batch = make_spiral(512)
    t = torch.randint(0, T, (batch.shape[0],))
    x_t, true_noise = forward_diffuse(batch, t)
    pred_noise = model(x_t, t)

    loss = ((pred_noise - true_noise) ** 2).mean()
    optim.zero_grad(); loss.backward(); optim.step()
    losses.append(loss.item())

    if step % 500 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

plt.plot(losses)
plt.xlabel('step')
plt.ylabel('MSE loss')
plt.title('Training — how well does the model predict the noise?')
plt.yscale('log')
plt.show()

# %% [markdown]
# ## 5. Reverse process — start from pure noise, denoise step by step
#
# At each step:
#   1. Predict the noise in the current x_t
#   2. Remove a SMALL portion of it → x_{t-1}
#   3. Add a bit of fresh noise (for stochasticity)
#   4. Repeat from t=T down to t=0

# %%
@torch.no_grad()
def sample(n=1000, save_every=20):
    model.eval()
    x = torch.randn(n, 2)  # start from pure noise
    trajectory = [x.clone()]

    for t in reversed(range(T)):
        t_batch = torch.full((n,), t, dtype=torch.long)
        pred_noise = model(x, t_batch)

        # DDPM reverse step (simplified)
        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        coef1 = 1 / torch.sqrt(alpha_t)
        coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
        mean = coef1 * (x - coef2 * pred_noise)

        if t > 0:
            noise = torch.randn_like(x)
            var = betas[t]
            x = mean + torch.sqrt(var) * noise
        else:
            x = mean

        if t % save_every == 0:
            trajectory.append(x.clone())

    return x, trajectory

final, trajectory = sample(n=2000)

# %% [markdown]
# ## 6. Visualize the reverse process — noise slowly becomes a spiral

# %%
fig, axes = plt.subplots(1, 6, figsize=(20, 3.5))
show_idx = [0, len(trajectory)//5, len(trajectory)//3,
            len(trajectory)//2, 2*len(trajectory)//3, len(trajectory)-1]

for ax, idx in zip(axes, show_idx):
    samples = trajectory[idx]
    ax.scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.4, color='red')
    # Show the "reverse step" — earliest idx is noise, last idx is final samples
    reverse_t = T - 1 - idx * (T // (len(trajectory) - 1))
    ax.set_title(f"reverse step, t ≈ {max(0, reverse_t)}")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    ax.axis('equal')

plt.suptitle("Reverse process — pure noise (left) → spiral (right)\n"
             "Each step removes a bit of noise. 100 steps later → clean data.",
             fontsize=12)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Side by side: real vs. generated

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(real_data[:, 0], real_data[:, 1], s=3, alpha=0.5, color='blue')
axes[0].set_title("Real data")
axes[0].set_xlim(-4, 4); axes[0].set_ylim(-4, 4); axes[0].axis('equal')

axes[1].scatter(final[:, 0], final[:, 1], s=3, alpha=0.5, color='red')
axes[1].set_title("Generated by diffusion\n(started from noise, denoised 100 steps)")
axes[1].set_xlim(-4, 4); axes[1].set_ylim(-4, 4); axes[1].axis('equal')

plt.suptitle("The diffusion model learned the spiral distribution from scratch")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. The "bent fingers" picture
#
# ```
# FORWARD PROCESS (no learning):
#   clean data ──→ + noise ──→ + noise ──→ ... ──→ pure static
#     t=0           t=1          t=2                  t=T
#
# REVERSE PROCESS (the trained network):
#   pure static ──→ - noise ──→ - noise ──→ ... ──→ clean sample
#     t=T            t=T-1        t=T-2              t=0
#
# The network learned to answer: "given x_t and t, what noise was added?"
# Subtract the predicted noise → get slightly cleaner x_{t-1}.
# Repeat 100 times → clean sample.
# ```

# %% [markdown]
# ## 9. Scaling up from here
#
# Everything you just built is the actual skeleton of Stable Diffusion:
#   - Instead of 2D points: images (projected to latent space via VAE)
#   - Instead of 4-layer MLP: U-Net with attention
#   - Instead of 2000 training steps: billions
#   - Instead of noise prediction alone: also conditioning on text embeddings
#
# The CORE mechanism is IDENTICAL. You'd recognize Stable Diffusion's training
# loop immediately after this notebook.

# %% [markdown]
# ## 10. Why your thesis doesn't use diffusion
#
# Diffusion models GENERATE images.
# Your thesis REASONS ABOUT images — doc/chart/image → text answer.
#
# No VLM in your pipeline uses diffusion. LLaVA, Qwen2-VL, InternVL —
# all decoder-only language models that attend to visual features.
# The "generative" in your thesis title refers to autoregressive text
# generation by the LLM, not image synthesis.

# %% [markdown]
# ## Summary
#
# | Stage | What happens |
# |-------|-------------|
# | Forward | Add noise step by step — NO network needed, just math |
# | Train | Network learns to predict the noise that was added |
# | Sample | Start from pure noise, iteratively subtract predicted noise |
#
# 2020 DDPM → 2022 Stable Diffusion → 2024 Flux.
# It's the dominant image generation approach today.
# But it's a DIFFERENT family of models from the VLMs you'll build.
