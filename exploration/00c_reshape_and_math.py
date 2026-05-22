# %% [markdown]
# # Day 0 (part C): Why reshape doesn't break the math
#
# The question: when you .view() a (4, 8, 65) tensor into (32, 65),
# why doesn't the math go wrong?
#
# The answer depends on WHICH OPERATION you apply after.
#   - Per-vector ops (cross-entropy, linear, relu):  RESHAPE IS SAFE
#   - Across-vector ops (attention, convolution):    RESHAPE BREAKS THINGS
#
# We'll prove both by computing the actual numbers.

# %%
import torch
import torch.nn.functional as F

torch.manual_seed(42)

# %% [markdown]
# ## 1. Setup: Karpathy's bigram scenario
#
# Shape (B=4, T=8, C=65):
#   4 sentences in a batch
#   Each sentence has 8 tokens
#   Each token has 65 logits (scores for each possible next char)

# %%
B, T, C = 4, 8, 65

logits = torch.randn(B, T, C)        # (4, 8, 65) — the model's predictions
targets = torch.randint(0, C, (B, T))  # (4, 8)    — the true next chars

print(f"logits shape:  {logits.shape}")
print(f"targets shape: {targets.shape}")
print(f"total (prediction, target) pairs: {B * T} = {B*T}")

# %% [markdown]
# ## 2. Compute cross-entropy the "manual" way — no reshape
#
# Loop over every (prediction, target) pair, compute the loss, then average.
# This is the mathematical definition: mean over all N pairs.

# %%
manual_losses = []
for b in range(B):
    for t in range(T):
        single_logit = logits[b, t]      # shape (65,)
        single_target = targets[b, t]    # a single integer
        # ce for a single pair: -log(softmax(logit)[target])
        loss_i = F.cross_entropy(single_logit.unsqueeze(0), single_target.unsqueeze(0))
        manual_losses.append(loss_i.item())

manual_mean = sum(manual_losses) / len(manual_losses)
print(f"Manual mean (32 independent computations): {manual_mean:.6f}")

# %% [markdown]
# ## 3. Compute it via reshape-then-cross_entropy (Karpathy's method)
#
# Reshape (4, 8, 65) → (32, 65). Reshape (4, 8) → (32,). Call cross_entropy once.

# %%
logits_flat = logits.view(B * T, C)        # (32, 65)
targets_flat = targets.view(B * T)         # (32,)

loss_via_reshape = F.cross_entropy(logits_flat, targets_flat)
print(f"Reshape method: {loss_via_reshape.item():.6f}")

# %% [markdown]
# ## 4. Proof: identical. To the last decimal.

# %%
print(f"Manual   : {manual_mean:.10f}")
print(f"Reshape  : {loss_via_reshape.item():.10f}")
print(f"Equal?   : {abs(manual_mean - loss_via_reshape.item()) < 1e-5}")

print()
print("Why they match:")
print("  Cross-entropy averages N independent (prediction, target) pairs.")
print("  The reshape just collapses (B, T) → (B*T) — no data moved.")
print("  Same 32 pairs. Same sum. Same mean. Identical number.")

# %% [markdown]
# ## 5. The mental model
#
# Think of cross-entropy as a function that doesn't know or care about
# "batches" or "sequences." It sees N pairs and averages the loss:
#
#     L = (1/N) · Σ  ce(logit_i, target_i)
#                i=1..N
#
# Whether the pairs were originally stored as (4, 8) or (32,) is irrelevant.
# The sum is the same.

# %% [markdown]
# ## 6. WHEN reshape IS unsafe — attention breaks
#
# Now let's break something on purpose. Attention computes softmax(Q·Kᵀ).
# Each token looks at every OTHER TOKEN IN ITS SENTENCE.
#
# If you flatten batch + seq, tokens from sentence 0 can now "see" tokens
# from sentence 3. That's wrong — different sentences shouldn't mix.

# %%
# Simulate attention scores: Q·Kᵀ  (before softmax)
# Each token in each sentence computes similarity with every OTHER token
# in that same sentence.

def attention_scores(x):
    """x shape: (B, T, C)  →  scores shape: (B, T, T)
    Each sentence has its own T×T attention matrix."""
    return x @ x.transpose(-2, -1)

correct = attention_scores(logits)
print(f"Correct attention:   {correct.shape}")
print("  → 4 sentences, each with its own 8×8 attention matrix")
print("  → tokens in sentence 0 only attend to tokens in sentence 0")

# Now FLATTEN first, then try attention  — this is WRONG
flattened = logits.view(B * T, C)
wrong = flattened @ flattened.transpose(-2, -1)
print(f"Flattened attention: {wrong.shape}")
print("  → ONE 32×32 attention matrix spanning ALL sentences")
print("  → token 3 (sentence 0) now attends to token 20 (sentence 2) — WRONG!")

# Concrete check: does token from batch 0 end up with nonzero similarity to batch 2?
token_b0_t0 = 0                # flat index of (batch=0, token=0)
token_b2_t0 = 2 * T + 0        # flat index of (batch=2, token=0)
cross_sentence_score = wrong[token_b0_t0, token_b2_t0].item()
print(f"\nCross-sentence attention score (should never exist): {cross_sentence_score:.4f}")
print("This leaks information between sentences — the model cheats.")

# %% [markdown]
# ## 7. The rule that governs everything
#
# ```
# RESHAPE IS SAFE when the next op treats each last-dim vector independently:
#   • Linear layer:    y = W·x        (one vector at a time)
#   • Activation:      y = relu(x)    (one number at a time)
#   • LayerNorm:       (one vector at a time)
#   • Cross-entropy:   (one pair at a time)
#
# RESHAPE BREAKS MATH when the op couples vectors across a dimension:
#   • Attention:       token i looks at token j in the SAME sequence
#   • Convolution:     pixel uses its spatial neighbors
#   • Pooling:         aggregates across a window
#   • Sum/mean over a specific dim
# ```
#
# Rule of thumb:
#   If the math is "do the same thing to every last-dim vector, in parallel"
#   → the grouping of leading dims doesn't matter → reshape is free.
#
#   If the math requires "this vector needs its neighbors"
#   → the grouping matters → DO NOT reshape across those boundaries.
#
# Cross-entropy falls in the first category. That's why Karpathy's
# `.view(B*T, C)` trick is mathematically identical to the per-pair loop.

# %%
