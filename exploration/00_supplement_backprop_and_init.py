# %% [markdown]
# # DEEP DIVE: Initialization, Backprop, and Training Diagnostics
#
# *(Built from your 5 questions after training the parity RNN. Each
#  section is its own click-in-the-brain proof.)*
#
# ---
#
# ## What you'll know by the end
#
# 1. **Why He / Xavier init exist** — and what *exactly* breaks without them.
# 2. **How backprop computes a gradient for a weight matrix** — chain rule
#    walked all the way through with worked-out numbers.
# 3. **Why gradients at `1e-6` were perfectly healthy** here — the difference
#    between *vanished* and *converged*.
# 4. **Why `mean prob = 0.5` is fine but `std prob = 0` was deadly** — and
#    what `std prob` actually measures.
# 5. **Why accuracy alone can lie** — and what to use alongside it.
#
# ---
#
# ## Roadmap (5 sections + summary)
#
# 1. The variance-preservation game (why init matters)
# 2. Backprop, step by step, with numbers
# 3. Healthy zero vs unhealthy zero — gradient diagnostics
# 4. Reading output statistics (mean vs std vs saturation)
# 5. When accuracy lies — metrics beyond it
# 6. The unified mental model

# %%
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
np.set_printoptions(precision=3, suppress=True)

# %% [markdown]
# # ─────────────────────────────────────────────────────────────────────
# # SECTION 1: Why initialization matters — the variance-preservation game
# # ─────────────────────────────────────────────────────────────────────
#
# ## The core question
#
# A deep network is a chain of matrix multiplications:
#
# ```
#   x  →  W₁  →  h₁  →  W₂  →  h₂  →  W₃  →  h₃  →  ...  →  y
# ```
#
# When you multiply a random vector `x` by a random matrix `W`, the result
# `Wx` has some new *size* (magnitude / variance). If that size grows at
# every layer → values **explode** → tanh saturates / numbers overflow.
# If it shrinks at every layer → values **vanish** → signal dies.
#
# **The goal of initialization**: pick `W`'s scale so that
# `Var(Wx) ≈ Var(x)`. Signal magnitude is *preserved* layer by layer.
# That's the whole game.
#
# ## The math (one paragraph, then code)
#
# Each output unit is a sum:
#
# ```
#   y_i  =  Σ  W[i, j] · x[j]            (j runs over the input dim, "fan_in")
#          j
# ```
#
# Assume `W` entries are i.i.d. with mean 0 and variance `σ_W²`, and `x`
# entries are i.i.d. with mean 0 and variance `σ_x²`. Then (independence + zero mean):
#
# ```
#   Var(y_i)  =  fan_in · σ_W² · σ_x²
# ```
#
# For `Var(y) ≈ Var(x)`, you need:
#
# ```
#   σ_W²  =  1 / fan_in       →  σ_W  =  sqrt(1 / fan_in)     (Xavier init)
# ```
#
# For ReLU networks (where ~half the signal is killed), you need DOUBLE
# the variance to compensate:
#
# ```
#   σ_W²  =  2 / fan_in       →  σ_W  =  sqrt(2 / fan_in)     (He init)
# ```
#
# Now let's SEE this in action.

# %%
# Set up a deep "network" that's just chained matmuls — no activations yet.
# We'll feed a random vector through it and watch its magnitude evolve.

depth = 30                              # number of layers
fan_in = 100                            # dimension per layer
num_trials = 50                         # average over multiple random inits/inputs

scales = {
    "Too small (×0.05)":       0.05,
    "Too small (×0.5/√fanin)": 0.5 / np.sqrt(fan_in),
    "Xavier (1/√fan_in)":      1.0 / np.sqrt(fan_in),
    "He (√2/√fan_in)":         np.sqrt(2) / np.sqrt(fan_in),
    "Too big  (×2.0/√fanin)":  2.0 / np.sqrt(fan_in),
    "Way too big (×1.0)":      1.0,
}

results = {name: [] for name in scales}

for name, scale in scales.items():
    for trial in range(num_trials):
        # Generate random input with unit variance
        x = torch.randn(fan_in)
        magnitudes = [x.std().item()]

        # Apply `depth` linear layers, each with the same init scale.
        # No activation here — pure linear chain so we can see the variance math cleanly.
        for layer in range(depth):
            W = torch.randn(fan_in, fan_in) * scale
            x = W @ x
            magnitudes.append(x.std().item())

        results[name].append(magnitudes)

# Plot
fig, ax = plt.subplots(figsize=(11, 6))
for name, runs in results.items():
    mean_traj = np.mean(runs, axis=0)
    ax.plot(mean_traj, label=name, lw=2)

ax.set_yscale("log")
ax.set_xlabel("Layer depth")
ax.set_ylabel("Activation magnitude (std)")
ax.set_title(
    "Signal magnitude through a 30-layer linear network at different init scales\n"
    "(log y-axis — straight lines = exponential growth or decay)"
)
ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="initial magnitude")
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## What you should see in the plot
#
# - **"Too small" curves** drop exponentially (straight line going down) →
#   by layer 30, signal magnitude is `1e-10` or less. **Dead network.**
#   In real training, gradients are even smaller — `(1e-10)²` levels.
#   Backprop has nothing to work with.
#
# - **"Way too big (×1.0)" curve** grows exponentially → by layer 30,
#   magnitude is `1e10` or more. In real training, this is where you'd
#   hit `inf`, `NaN`, or tanh-saturation at every layer.
#
# - **Xavier and He stay roughly flat near `1.0`** for the whole 30 layers.
#   Signal magnitude is preserved. **Gradients can flow back.**
#
# The slight difference between Xavier (`1/√fan_in`) and He (`√2/√fan_in`)
# matters when you add a ReLU between layers (which kills ~half the signal).
# Without ReLU, Xavier is technically the right scale; with ReLU, He compensates.

# %% [markdown]
# ## Connecting to your parity RNN
#
# Recall your three weight matrices and their `fan_in`:
#
# | Matrix | Shape | fan_in | He std `= √(2/fan_in)` |
# |---|---|---|---|
# | `W_x`    | `(32, 1)`  | 1  | `√2 ≈ 1.414` |
# | `W_hist` | `(32, 32)` | 32 | `√(2/32) ≈ 0.25` |
# | `lin.W`  | `(1, 32)`  | 32 | `√(2/32) ≈ 0.25` |
#
# Your earlier `* 0.1` was **way smaller** than `0.25` for `W_hist` and
# `lin.W` — signal shrunk at every layer, gradients arrived nearly dead at
# the weights, training crawled.
#
# He init's `√(2/fan_in)` is *exactly* the magic number that makes
# `Var(Wx) ≈ Var(x)` so signal stays alive across the 10-step recurrence.

# %%
# Verify the math empirically for your specific shapes
fan_in_examples = [1, 32, 100, 512]
print(f"{'fan_in':>8} | {'Xavier std':>11} | {'He std':>9} | "
      f"{'measured Var(Wx)':>18}")
print("-" * 70)
for fi in fan_in_examples:
    xavier_std = 1 / np.sqrt(fi)
    he_std = np.sqrt(2) / np.sqrt(fi)

    # Empirically check: if x ~ N(0, 1) and W has He init, what's Var(Wx)?
    fan_out = 32
    x_samples = torch.randn(10000, fi)
    W_he = torch.randn(fan_out, fi) * he_std
    y = x_samples @ W_he.T  # shape (10000, fan_out)
    measured_var = y.var().item()

    print(f"{fi:>8} | {xavier_std:>11.4f} | {he_std:>9.4f} | "
          f"{measured_var:>18.4f}  ← should be ~2 (He target = 2·Var(x))")

# %% [markdown]
# Notice `measured Var(Wx) ≈ 2.0` for all `fan_in` values — that's He
# successfully delivering "twice the input variance" to compensate for
# ReLU's half-killing. For tanh networks, Xavier gives `Var(Wx) ≈ 1.0`
# instead.

# %% [markdown]
# ## Hey-dude one-liner for Section 1
#
# > Hey dude, **init is variance bookkeeping.** Each matmul rescales
# > signal by a factor depending on `fan_in × Var(W)`. If that factor
# > isn't ~1, signal exponentially explodes or vanishes across depth.
# > Xavier (`σ = 1/√fan_in`) and He (`σ = √2/√fan_in`) are the SCALES
# > that make Var(output) ≈ Var(input) — keeping signal alive layer by
# > layer so gradients can flow back.

# %% [markdown]
# # ─────────────────────────────────────────────────────────────────────
# # SECTION 2: Backprop, step by step, with actual numbers
# # ─────────────────────────────────────────────────────────────────────
#
# ## What backprop actually computes
#
# Given some loss `L` and a weight matrix `W` deep inside the network,
# backprop's job is to compute `∂L / ∂W` — a matrix of the same shape as
# `W` telling you "if I nudge `W[i, j]` by a tiny amount, how much does
# `L` change?"
#
# It does this with the **chain rule**, applied to each elementary
# operation in the forward pass.
#
# ## The setup — smallest network where everything's visible
#
# Let's build a 1-hidden-layer network so tiny we can do all the math by
# hand:
#
# ```
#   forward:
#     x          shape (2,)        — input
#     z1 = W1 @ x + b1   shape (3,)   — pre-activation of hidden
#     h  = tanh(z1)       shape (3,)
#     z2 = W2 @ h + b2   shape (1,)   — pre-activation of output (logit)
#     p  = sigmoid(z2)    shape (1,)
#     L  = -[ y·log(p) + (1-y)·log(1-p) ]    (scalar BCE loss)
# ```

# %%
# Concrete numbers — we'll hand-compute gradients and verify with PyTorch.
x  = torch.tensor([1.0, 2.0])
y  = torch.tensor([1.0])

W1 = torch.tensor([[ 0.2, -0.3],
                   [ 0.5,  0.1],
                   [-0.2,  0.4]], requires_grad=True)   # shape (3, 2)
b1 = torch.tensor([0.1, -0.1, 0.05], requires_grad=True)

W2 = torch.tensor([[0.6, -0.4, 0.3]], requires_grad=True)   # shape (1, 3)
b2 = torch.tensor([0.0], requires_grad=True)

# Forward pass — step by step
z1 = W1 @ x + b1
h  = torch.tanh(z1)
z2 = W2 @ h + b2
p  = torch.sigmoid(z2)
L  = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))

print("z1 =", z1.detach().numpy())
print("h  =", h.detach().numpy())
print("z2 =", z2.detach().numpy())
print("p  =", p.detach().numpy())
print("L  =", L.detach().item())

# %% [markdown]
# ## Now the backward pass — by hand
#
# The chain rule says: to get `∂L/∂W1`, you multiply local derivatives
# along the path from `L` back to `W1`:
#
# ```
#   ∂L / ∂W1   =   ∂L/∂p  ·  ∂p/∂z2  ·  ∂z2/∂h  ·  ∂h/∂z1  ·  ∂z1/∂W1
# ```
#
# In matrix-land each ∂ is itself a matrix (or "Jacobian"). Walking
# backwards step by step:
#
# ### Step A: ∂L/∂p (scalar derivative of BCE)
#
# ```
#   L = -[ y log(p) + (1-y) log(1-p) ]
#   ∂L/∂p = -[ y/p  -  (1-y)/(1-p) ]
#         = (p - y) / [p(1-p)]
# ```
#
# ### Step B: ∂p/∂z2 (sigmoid derivative)
#
# ```
#   p = sigmoid(z2),  so  ∂p/∂z2 = p(1-p)
# ```
#
# Combining A and B (this is why people merge sigmoid + BCE):
#
# ```
#   ∂L/∂z2 = (∂L/∂p) · (∂p/∂z2) = (p - y) / [p(1-p)] · p(1-p) = (p - y)
# ```
#
# 👉 **The full BCE+sigmoid derivative is just `p - y`.** Neat, isn't it?

# %%
# Verify
dL_dz2 = (p - y).detach().numpy()
print("dL/dz2 = p - y =", dL_dz2)

# %% [markdown]
# ### Step C: ∂z2/∂W2 — the matrix-multiplication backward rule
#
# Here's the key piece. `z2 = W2 @ h + b2`. To find how `L` depends on
# `W2`, we need `∂z2/∂W2`, then chain it with `∂L/∂z2`.
#
# **The rule (memorize this — it's the heart of backprop):**
#
# > **`z = W @ h` → `∂L/∂W = (∂L/∂z) · h.T`**     (outer product!)
#
# `∂L/∂z` is a column vector, `h.T` is a row vector — their product is
# a matrix of the same shape as `W`.
#
# Why this works: each entry `W[i, j]` only multiplies `h[j]` to
# contribute to `z[i]`. So `∂z[i] / ∂W[i, j] = h[j]`, and `∂L/∂W[i, j] =
# ∂L/∂z[i] · h[j]`. The OUTER PRODUCT formula collects all these.

# %%
# Compute by hand
dL_dz2_tensor = (p - y)                          # shape (1,)
dL_dW2_manual = dL_dz2_tensor.unsqueeze(1) @ h.unsqueeze(0)  # outer product → shape (1, 3)
dL_db2_manual = dL_dz2_tensor                                # bias grad = upstream grad

print("dL/dW2 (manual) =", dL_dW2_manual.detach().numpy())
print("dL/db2 (manual) =", dL_db2_manual.detach().numpy())

# Verify against autograd
L.backward(retain_graph=True)
print("\ndL/dW2 (PyTorch) =", W2.grad.numpy())
print("dL/db2 (PyTorch) =", b2.grad.numpy())
# Should match exactly (up to floating point precision)!

# %% [markdown]
# ### Step D: backprop through to the hidden layer
#
# Now keep going further back. We need `∂L/∂h` so we can then get `∂L/∂z1`
# (through tanh), then `∂L/∂W1`.
#
# **The other half of the matmul backward rule:**
#
# > **`z = W @ h` → `∂L/∂h = W.T · (∂L/∂z)`**
#
# Why: each `h[j]` multiplies column `j` of `W` to contribute across all
# `z[i]`. So `∂z[i]/∂h[j] = W[i, j]`, and `∂L/∂h[j] = Σ_i W[i, j] · ∂L/∂z[i]`,
# which is exactly `(W.T · ∂L/∂z)[j]`.

# %%
dL_dh = W2.T @ dL_dz2_tensor                     # shape (3,)
print("dL/dh =", dL_dh.detach().numpy())

# %% [markdown]
# ### Step E: through the tanh nonlinearity
#
# `h = tanh(z1)`. The tanh derivative is `1 - tanh(z1)² = 1 - h²`.
# It's elementwise (no matrix structure), so:
#
# ```
#   ∂L/∂z1 = ∂L/∂h · (1 - h²)        (elementwise product)
# ```

# %%
dL_dz1 = dL_dh * (1 - h ** 2)
print("dL/dz1 =", dL_dz1.detach().numpy())

# %% [markdown]
# ### Step F: finally to W1 — same outer-product trick
#
# `z1 = W1 @ x + b1`, so:
#
# ```
#   ∂L/∂W1  =  ∂L/∂z1 · x.T          (outer product)
#   ∂L/∂b1  =  ∂L/∂z1
# ```

# %%
dL_dW1_manual = dL_dz1.unsqueeze(1) @ x.unsqueeze(0)
dL_db1_manual = dL_dz1

print("dL/dW1 (manual) =\n", dL_dW1_manual.detach().numpy())
print("\ndL/db1 (manual) =", dL_db1_manual.detach().numpy())

print("\n--- Compare against PyTorch ---")
print("dL/dW1 (PyTorch) =\n", W1.grad.numpy())
print("dL/db1 (PyTorch) =", b1.grad.numpy())

# %% [markdown]
# **They match.** You've just done backprop by hand and matched
# PyTorch's autograd to floating-point precision.
#
# ## The two essential rules
#
# That whole exercise reduces to TWO rules you must internalize:
#
# > **Forward**: `z = W @ h + b`
# >
# > **Backward (weight gradient)**:    `∂L/∂W = (∂L/∂z) · h.T`     ← outer product
# >
# > **Backward (input gradient)**:     `∂L/∂h = W.T · (∂L/∂z)`     ← matrix-vector
# >
# > **Backward (bias gradient)**:      `∂L/∂b = ∂L/∂z`             ← copy
#
# Every layer in every network applies these rules. PyTorch's autograd
# is a clever bookkeeping engine that does exactly this for every op in
# your forward pass, in reverse order. **There's no magic.**
#
# ## For your RNN specifically
#
# Each time step `t` did:
#
# ```
#   z_t = W_x @ x_t  +  W_hist @ h_{t-1}  +  b
#   h_t = tanh(z_t)
# ```
#
# Backprop applied:
#
# ```
#   ∂L/∂W_x      +=  (∂L/∂z_t) · x_t.T                    (summed over t)
#   ∂L/∂W_hist   +=  (∂L/∂z_t) · h_{t-1}.T                (summed over t)
#   ∂L/∂h_{t-1}   =  W_hist.T · (∂L/∂z_t)                  (gradient flows back through time)
# ```
#
# The **gradient w.r.t. `W_hist` is summed across all 10 time steps** —
# each contributes its own outer product. That's why RNN gradients have
# extra noise: 10 noisy outer products added together. And it's also why
# **vanishing-through-time** happens: `∂L/∂h_{t-1}` requires multiplying
# by `W_hist.T` at every step backward, so if `W_hist` has spectral
# radius < 1, the gradient signal shrinks exponentially with the number
# of time steps.

# %% [markdown]
# ## Hey-dude one-liner for Section 2
#
# > Hey dude, **backprop = chain rule + two matrix tricks**:
# > (1) gradient w.r.t. W = outer product of upstream gradient with input;
# > (2) gradient w.r.t. input = `W.T` times upstream gradient.
# > Apply these recursively from loss back to first layer, and you have
# > every weight's gradient. PyTorch automates the bookkeeping — but the
# > arithmetic is exactly what you just did by hand.

# %% [markdown]
# # ─────────────────────────────────────────────────────────────────────
# # SECTION 3: Healthy zero vs unhealthy zero — gradient diagnostics
# # ─────────────────────────────────────────────────────────────────────
#
# Your trained RNN ended with `|grad| ≈ 1e-6` and 100% accuracy. You
# asked "but the gradients are tiny — why did it still work?"
#
# Because tiny gradients can mean **two opposite things**:

# %% [markdown]
# ## Pattern 1: HEALTHY tiny gradients (your trained RNN)
#
# Model has converged. Loss ≈ 0. Predictions are correct AND confident.
# Then for BCE: `∂L/∂z = p - y ≈ 0` because `p` is already approximately
# equal to `y`. Of course the gradient is tiny — there's nothing to improve.
#
# **Signature:**
#
# | Metric | Value |
# |---|---|
# | Loss | low (close to 0) |
# | Accuracy | high (close to 100%) |
# | `|grad|` | tiny (1e-5 to 1e-7) |
# | `std prob` | high (~0.4 to 0.5) — confident predictions split between 0 and 1 |
# | `saturated_frac` | varies (often moderate; "switch-like" neurons are fine) |
#
# **Interpretation:** the model is *done learning*. Gradients are tiny
# *because they SHOULD be*.

# %% [markdown]
# ## Pattern 2: UNHEALTHY tiny gradients (your stuck RNN from earlier)
#
# Model can't learn. Loss = ln(2) (chance level). Predictions are ~0.5 for
# everything. Gradient w.r.t. logit `(p - y)` is ±0.5 (moderate!) — but
# this gradient has to backprop through saturated tanh units, so by the
# time it reaches `W_hist`, it's been multiplied by tiny tanh derivatives
# many times. Arrives nearly dead.
#
# **Signature:**
#
# | Metric | Value |
# |---|---|
# | Loss | stuck at chance (`ln(2) ≈ 0.69` for binary) |
# | Accuracy | stuck near chance (50% for balanced binary) |
# | `|grad|` | tiny (1e-5 to 1e-7) |
# | `std prob` | ≈ 0 — predictions are uniformly ~0.5 |
# | `saturated_frac` | high (often >0.5) if saturation is the cause |
#
# **Interpretation:** the model is *unable to learn*. Gradients are tiny
# *but SHOULDN'T be*.

# %% [markdown]
# ## How to tell them apart — the decision tree
#
# Three quick checks distinguish them at a glance:
#
# ```
#   Is loss low (≪ ln 2 for binary)?
#       │
#       ├─ YES → is accuracy high (≫ chance)?
#       │            │
#       │            ├─ YES → HEALTHY. Model converged. Done. ✓
#       │            └─ NO  → something's weird (loss says good, acc says bad — investigate).
#       │
#       └─ NO  → is std prob > 0.1?
#                    │
#                    ├─ NO  → UNHEALTHY. Model collapsed (all outputs ~the same).
#                    └─ YES → mid-training, just keep going.
# ```
#
# Tiny gradients by themselves are AMBIGUOUS. You always look at them
# alongside loss + accuracy + prediction diversity.

# %%
# Let's simulate both patterns concretely to drive this home.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Pattern 1: HEALTHY convergence
# Loss decays exponentially from 0.69 → 0; grad scales with loss
steps = np.arange(2000)
healthy_loss = 0.69 * np.exp(-steps / 200) + 1e-3 * np.random.randn(2000) ** 2
healthy_grad = 1e-2 * np.exp(-steps / 200) + 1e-7
healthy_acc  = 100 * (1 - np.exp(-steps / 150))

# Pattern 2: UNHEALTHY stuck
# Loss stays at chance, accuracy stays at 50%, grad is small from the start
unhealthy_loss = 0.69 + 0.005 * np.random.randn(2000) ** 2
unhealthy_grad = 1e-6 + 1e-7 * np.random.randn(2000) ** 2
unhealthy_acc  = 50 + 5 * np.random.randn(2000)

ax = axes[0]
ax.set_title("HEALTHY: trained RNN (your final run)")
ax.plot(steps, healthy_loss, label="loss", color="tab:blue")
ax.plot(steps, healthy_grad * 50, label="|grad| × 50", color="tab:red")
ax.plot(steps, healthy_acc / 100, label="accuracy (0-1)", color="tab:green")
ax.set_xlabel("step")
ax.set_ylim(0, 1.0)
ax.axhline(0.69, color="red", linestyle="--", alpha=0.3, label="chance loss")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.set_title("UNHEALTHY: collapsed RNN (your earlier stuck run)")
ax.plot(steps, unhealthy_loss, label="loss", color="tab:blue")
ax.plot(steps, unhealthy_grad * 1e5, label="|grad| × 1e5", color="tab:red")
ax.plot(steps, unhealthy_acc / 100, label="accuracy (0-1)", color="tab:green")
ax.set_xlabel("step")
ax.set_ylim(0, 1.0)
ax.axhline(0.69, color="red", linestyle="--", alpha=0.3, label="chance loss")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Connecting back to your run
#
# Your final plot showed:
# - Loss → 0  ✓
# - Accuracy → 100%  ✓
# - `std prob` → 0.5 (full confidence on both classes)  ✓
# - `|grad|` → 1e-6
#
# That's textbook **healthy convergence**. The tiny gradients were the
# *consequence* of being correct, not the cause of being stuck. ✓

# %% [markdown]
# ## Hey-dude one-liner for Section 3
#
# > Hey dude, **a tiny gradient is just a number** — its meaning depends
# > on context. Pair it with loss and accuracy to know what it means:
# > tiny grad + low loss + high acc = "I'm done learning, leave me alone."
# > Tiny grad + chance loss + chance acc = "I can't learn, save me."
# > Same tiny number, opposite stories.

# %% [markdown]
# # ─────────────────────────────────────────────────────────────────────
# # SECTION 4: Reading output statistics — why mean=0.5 was fine but std=0 was deadly
# # ─────────────────────────────────────────────────────────────────────
#
# Your question: *"mean prob is still 0.5 (same as failed runs). Just the
# std helped?"*
#
# Yes — **and that's an important pattern to internalize.** Let me show
# you why.
#
# ## The parity data is naturally balanced
#
# Half of all random sequences have even parity, half odd. So the AVERAGE
# label `y` over a batch is `~0.5`. Even a perfect model — outputting
# exactly the right answer for every sequence — will have `mean(prob) ≈ 0.5`
# on a balanced batch.
#
# > **`mean prob` ≈ 0.5 means almost nothing on balanced binary data.**
# > Both the perfectly-correct model AND the never-learned-anything model
# > produce that same number.

# %% [markdown]
# ## What distinguishes them: where on the [0, 1] axis predictions LIVE
#
# Two completely different models, both producing `mean prob = 0.5`:

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# A "stuck" model: predicts ~0.5 for every input
stuck_probs = 0.5 + 0.02 * np.random.randn(1000)
ax = axes[0]
ax.hist(stuck_probs, bins=50, color="tab:red", edgecolor="black", alpha=0.7)
ax.axvline(stuck_probs.mean(), color="black", linestyle="--",
           label=f"mean = {stuck_probs.mean():.3f}")
ax.set_xlim(0, 1)
ax.set_title(f"STUCK model — std = {stuck_probs.std():.3f}\n"
             "All predictions live near 0.5 → no discrimination")
ax.set_xlabel("predicted probability"); ax.set_ylabel("# samples")
ax.legend()

# A "trained" model: predicts ~0 for evens, ~1 for odds
half_zeros = 0.05 + 0.05 * np.random.randn(500).clip(0, 1)
half_ones  = 0.95 - 0.05 * np.random.randn(500).clip(0, 1)
trained_probs = np.concatenate([half_zeros, half_ones])
ax = axes[1]
ax.hist(trained_probs, bins=50, color="tab:green", edgecolor="black", alpha=0.7)
ax.axvline(trained_probs.mean(), color="black", linestyle="--",
           label=f"mean = {trained_probs.mean():.3f}")
ax.set_xlim(0, 1)
ax.set_title(f"TRAINED model — std = {trained_probs.std():.3f}\n"
             "Predictions live at extremes → confident discrimination")
ax.set_xlabel("predicted probability"); ax.set_ylabel("# samples")
ax.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# **Same mean ≈ 0.5 in both panels.** Look at the histograms:
#
# - **Left**: a single tall spike at 0.5. The model emits the same number
#   for every input. `std ≈ 0`. Loss = `−log(0.5) = 0.69` for every sample.
#   Stuck.
#
# - **Right**: a bimodal distribution — predictions hug 0 OR 1. The
#   model emits *different confident answers* for different inputs.
#   `std ≈ 0.5`. Loss → 0 because each sample is correctly classified.
#
# ## What `std prob` actually measures
#
# **`std prob` is the spread of predictions across the batch.** Concretely:
#
# - **`std ≈ 0`**: model gives the same output to every input. No
#   discrimination, no information learned.
# - **`std small but nonzero` (~0.05 to 0.15)**: model is just starting
#   to differentiate inputs but isn't confident yet. Mid-training.
# - **`std large` (~0.3 to 0.5)**: model produces confident, varied
#   predictions. For balanced binary, max possible std is ~0.5 (achieved
#   when half the predictions are at 0 and half at 1).
#
# So `std prob` is **"is the model paying attention to the input or
# ignoring it?"** — a far better stuck-detector than `mean prob`.

# %% [markdown]
# ## Related: what about saturated_frac?
#
# `saturated_frac` measures how many hidden units are pinned at ±1. It's
# a different signal — about INTERNAL activations, not outputs.
#
# - **Saturated_frac low**: hidden units in the linear-ish region.
#   Gradients flow through tanh.
# - **Saturated_frac moderate (~0.2 to 0.4)**: some units use saturation
#   as a "switch" — often healthy for classification tasks where you want
#   binary-like internal features.
# - **Saturated_frac very high (>0.7)**: gradients dying through tanh.
#   Bad sign UNLESS the model has already converged.
#
# Your final trained run had `saturated_frac ≈ 0.23` — moderate, healthy.
# The hidden units behave as "parity switches" — exactly what you'd want
# for this task.

# %% [markdown]
# ## Hey-dude one-liner for Section 4
#
# > Hey dude, **`mean prob` tells you almost nothing** on balanced binary
# > data — both garbage and perfect models give 0.5. **`std prob` tells
# > you whether the model is using its input** — std=0 means "ignoring
# > input, predicting same thing always," std≈0.5 means "confidently
# > emitting different answers for different inputs." `std prob` is the
# > stuck-detector you actually want.

# %% [markdown]
# # ─────────────────────────────────────────────────────────────────────
# # SECTION 5: When accuracy lies — and what to use alongside it
# # ─────────────────────────────────────────────────────────────────────
#
# Your question: *"is accuracy enough?"*
#
# **For balanced binary tasks like parity: yes, accuracy is a good
# headline metric.** But it has well-known failure modes you should know.

# %% [markdown]
# ## Failure mode 1: imbalanced classes
#
# Imagine a medical screening dataset: 99% of patients are healthy,
# 1% are sick. A model that **always predicts "healthy"** gets:
#
# ```
#   accuracy = 99%       ← looks great!
#   actual usefulness = ZERO   ← finds ZERO sick patients
# ```
#
# Accuracy is dominated by the majority class. You need **precision,
# recall, F1** — they measure performance per class.

# %%
# Demonstrate with a confusion matrix simulation
from sklearn.metrics import confusion_matrix  # for the print

# 1000 samples, 99% class 0, 1% class 1
y_true = np.concatenate([np.zeros(990), np.ones(10)])

# Model that always predicts 0
y_pred_dumb = np.zeros(1000)

# Compute metrics manually
def metrics(y_true, y_pred):
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)

m = metrics(y_true, y_pred_dumb)
print("DUMB model (always predicts 0) on imbalanced data:")
print(f"  accuracy  = {m['accuracy']*100:.1f}%  ← looks great")
print(f"  precision = {m['precision']*100:.1f}%   ← undefined / 0")
print(f"  recall    = {m['recall']*100:.1f}%   ← finds ZERO sick patients!")
print(f"  F1 score  = {m['f1']*100:.1f}%")
print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")

# %% [markdown]
# 99% accuracy. 0% recall. **Useless for the actual task** (finding sick
# patients) despite the great-looking number. This is why you always
# report precision and recall on imbalanced binary problems.

# %% [markdown]
# ## Failure mode 2: confidence calibration
#
# A model that says *"90% confident this is class 1"* should be right
# 90% of the time on that class of predictions. Accuracy alone doesn't
# measure this — you can be 100% accurate AND poorly calibrated (e.g.,
# always saying 99% confident when you should say 60% confident).
#
# Calibration matters for downstream decisions (e.g., if a doctor needs
# to know "should I order more tests?", a poorly-calibrated 90% is
# misleading).
#
# Common metric: **Brier score** = mean squared error between predicted
# probability and 0/1 label. Lower is better.

# %% [markdown]
# ## Failure mode 3: ranking quality matters more than threshold
#
# Sometimes you don't need correct binary predictions — you need correct
# *ranking* (which items are most likely positive?). Accuracy uses a
# 0.5 threshold, but a good model might be perfect at ranking even if
# the threshold is wrong.
#
# Metric for this: **AUC-ROC** (Area Under the Receiver Operating
# Characteristic curve). Measures the probability that a random positive
# is scored higher than a random negative. Threshold-independent.

# %% [markdown]
# ## Quick reference: which metric for which situation
#
# | Situation | Metric(s) |
# |---|---|
# | Balanced binary, simple decision (your parity) | Accuracy is fine. Add F1 for safety. |
# | Imbalanced binary | Precision, recall, F1. **Don't trust accuracy alone.** |
# | Need calibrated probabilities | Brier score, log-loss, reliability diagram |
# | Ranking matters | AUC-ROC, AUC-PR, top-k accuracy |
# | Multi-class | Per-class precision/recall, macro F1, confusion matrix |
# | Regression | MSE, MAE, R² |
# | Sequence generation | BLEU, ROUGE, perplexity, human eval |
#
# **For your parity task specifically: accuracy is perfectly adequate.**
# The data is balanced and the task is binary. You also already have
# `std prob` as a free side metric for "is the model collapsed."

# %% [markdown]
# ## Hey-dude one-liner for Section 5
#
# > Hey dude, **accuracy is fine for balanced binary tasks like parity**.
# > For imbalanced data, a dumb majority-class predictor gets sky-high
# > accuracy with zero usefulness — so you also report precision, recall,
# > F1. If you need confident probabilities (medical, finance), check
# > calibration. If you need ranking, use AUC. **The right metric depends
# > on the decision the model serves.**

# %% [markdown]
# # ─────────────────────────────────────────────────────────────────────
# # The unified mental model (5-section recap)
# # ─────────────────────────────────────────────────────────────────────
#
# ```
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                                                                 │
#   │  1. INIT IS VARIANCE BOOKKEEPING                                │
#   │     Each matmul scales signal by ≈ fan_in × Var(W).             │
#   │     Set Var(W) = 1/fan_in (Xavier) or 2/fan_in (He) to          │
#   │     PRESERVE signal magnitude across layers.                    │
#   │                                                                 │
#   │  2. BACKPROP = CHAIN RULE + TWO MATRIX TRICKS                   │
#   │     ∂L/∂W = (∂L/∂z) · h.T          (outer product)             │
#   │     ∂L/∂h = W.T · (∂L/∂z)          (matrix-vector)             │
#   │     Applied recursively from loss back to first layer.          │
#   │                                                                 │
#   │  3. TINY GRADIENT ≠ BAD                                         │
#   │     low loss + high acc + tiny grad = healthy converged ✓       │
#   │     chance loss + chance acc + tiny grad = collapsed ✗          │
#   │     Always read grad together with loss and accuracy.           │
#   │                                                                 │
#   │  4. MEAN PROB IS LIES, STD PROB IS TRUTH                        │
#   │     Mean ≈ 0.5 says nothing on balanced data.                   │
#   │     Std ≈ 0 = model collapsed. Std large = confident, varied.   │
#   │                                                                 │
#   │  5. ACCURACY IS FINE FOR BALANCED BINARY                        │
#   │     For imbalanced/calibrated/ranking tasks, use additional     │
#   │     metrics: precision, recall, F1, AUC, Brier.                 │
#   │                                                                 │
#   └─────────────────────────────────────────────────────────────────┘
# ```
#
# ## Hey-dude grand one-liner
#
# > Hey dude, training a neural network = **shaping signal so it
# > propagates cleanly forward AND backward**. Init handles forward
# > variance preservation. Backprop handles backward gradient
# > computation. Diagnostics (loss, accuracy, std prob, sat frac, grad
# > magnitude) tell you what's happening at each layer. Every "stuck"
# > model has a visible signature in these metrics — you've now seen
# > both healthy convergence AND saturation collapse from the inside,
# > and can diagnose either at a glance.

# %% [markdown]
# ## The mental checklist for any future training run
#
# 1. **Loss starts where?** For binary, `ln(2) ≈ 0.69` is chance. If it
#    starts much higher, init is too aggressive.
# 2. **Loss drops past chance?** If yes, learning began. If no, check
#    grad magnitudes and saturated_frac.
# 3. **Std prob grows?** If yes, model is differentiating inputs. If
#    stuck at 0, model is collapsed.
# 4. **Saturated_frac stays low or grows moderately?** Healthy. Spikes
#    >0.7 quickly = bad init.
# 5. **Gradient magnitudes stable or smoothly decreasing?** Healthy.
#    Sudden drops to 1e-8 mid-training = saturation cascade.
# 6. **Accuracy on balanced binary climbing toward 100%?** Done. For
#    imbalanced/multi-class, also check F1 and confusion matrix.
#
# Run that checklist on any future training and you'll catch most
# problems within the first few hundred steps.

# %%
