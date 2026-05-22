# %% [markdown]
# # Day 0 (part E): RNNs — Where "Time Step" Came From
#
# Before transformers, RNNs ruled sequences for a decade.
# You need to understand them because:
#   1. The word "time" (T in your tensors) comes from RNNs
#   2. The LIMITATIONS of RNNs are EXACTLY what transformers fixed
#   3. Karpathy's #2-5 videos walk through this; we're compressing it here
#
# Core idea in one sentence:
#   An RNN is a LOOP that processes one token at a time, carrying a
#   "memory" (hidden state) from step to step. Each token's processing
#   depends on the memory built from all previous tokens.

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

torch.manual_seed(42)

# %% [markdown]
# ## 1. The RNN cell — a loop body
#
# At each time step t, an RNN does:
#
#     h_t = tanh( W_x @ x_t  +  W_h @ h_{t-1}  +  b )
#
# In words:
#   h_t       = the new hidden state (memory after seeing this token)
#   x_t       = the current input token
#   h_{t-1}   = the previous hidden state (memory from before)
#   W_x, W_h  = learned weight matrices
#   b         = learned bias
#
# The hidden state h is SHARED across time — same weights at every step.
# That's the "recurrence" — same function, applied in a loop.

# %%
class MyTinyRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_x = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_sequence):
        """
        x_sequence shape: (T, input_dim)
        Returns: list of hidden states at every time step
        """
        T = x_sequence.shape[0]
        h = torch.zeros(self.hidden_dim)  # initial memory is zero
        all_hidden_states = []

        for t in range(T):
            x_t = x_sequence[t]
            # The recurrence — SAME W_x and W_h applied EVERY step
            h = torch.tanh(self.W_x(x_t) + self.W_h(h))
            all_hidden_states.append(h.detach().clone())

        return torch.stack(all_hidden_states)

rnn = MyTinyRNN(input_dim=4, hidden_dim=6)

# Feed in a sequence of 10 tokens
seq = torch.randn(10, 4)  # 10 time steps, each token is a 4-dim vector
hidden_trajectory = rnn(seq)

print(f"Input sequence shape: {seq.shape}        ← 10 time steps, 4 features each")
print(f"Hidden trajectory shape: {hidden_trajectory.shape}  ← 10 time steps, 6 hidden dims each")
print(f"\nAt each step, the RNN computed: h_t = tanh(W_x·x_t + W_h·h_{{t-1}})")
print(f"The SAME W_x and W_h were used at every step.")

# %% [markdown]
# ## 2. The "unrolled" view — see time as a chain
#
# A cool way to picture an RNN: imagine the loop UNROLLED in time.
# Same cell, repeated, with hidden state flowing left-to-right.

# %%
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(-1, 11)
ax.set_ylim(-1, 3)
ax.axis('off')

for t in range(10):
    # Draw the RNN cell as a box
    ax.add_patch(Rectangle((t, 0.5), 0.8, 1, facecolor='lightyellow', edgecolor='black'))
    ax.text(t + 0.4, 1, 'RNN', ha='center', va='center', fontsize=10, fontweight='bold')

    # Input arrow from below
    ax.annotate('', xy=(t + 0.4, 0.5), xytext=(t + 0.4, -0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(t + 0.4, -0.6, f'x_{t}', ha='center', color='blue', fontsize=10)

    # Output arrow up
    ax.annotate('', xy=(t + 0.4, 2.3), xytext=(t + 0.4, 1.5),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.text(t + 0.4, 2.5, f'h_{t}', ha='center', color='green', fontsize=10)

    # Hidden state flow (rightward)
    if t < 9:
        ax.annotate('', xy=(t + 1, 1), xytext=(t + 0.8, 1),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax.text(5, 2.9, "Hidden state flows LEFT TO RIGHT through time",
        ha='center', fontsize=12, color='red', fontweight='bold')
ax.text(5, -1, "Same RNN cell (same weights) applied at every time step",
        ha='center', fontsize=11, style='italic')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Visualize the memory evolving through time
#
# Watch the hidden state change as the RNN reads more of the sequence.
# Each "row" in the plot is the memory after seeing that many tokens.

# %%
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(hidden_trajectory.numpy(), cmap='RdBu', aspect='auto')
ax.set_xlabel('Hidden dimension (6 units of memory)')
ax.set_ylabel('Time step')
ax.set_title("Hidden state over time\n"
             "Each row = memory after reading tokens 0 through t\n"
             "You can see memory EVOLVING as more tokens arrive")
ax.set_yticks(range(10))
ax.set_yticklabels([f"after reading t={t}" for t in range(10)])
plt.colorbar(im, ax=ax, label='activation value')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. PyTorch's built-in RNN (this is what people actually use)

# %%
torch_rnn = nn.RNN(input_size=4, hidden_size=6, batch_first=True)

# Input: (batch, time, features)
x = torch.randn(1, 10, 4)  # one sequence of 10 tokens, each 4-dim
output, final_h = torch_rnn(x)

print(f"Input:         {x.shape}         ← (B=1, T=10, features=4)")
print(f"Output:        {output.shape}    ← hidden state at EVERY time step")
print(f"Final hidden:  {final_h.shape}   ← just the LAST hidden state")

print()
print("THE 'T' YOU SEE IN YOUR TRANSFORMER TENSORS — (B, T, C) —")
print("LITERALLY CAME FROM THIS. Each t is a time step the RNN would process.")

# %% [markdown]
# ## 5. Why RNNs died — and why transformers won
#
# RNNs have TWO critical limitations that transformers completely solve:

# %% [markdown]
# ### Limitation 1: SEQUENTIAL processing
#
#     h_0 → h_1 → h_2 → h_3 → ... → h_99
#
# h_1 needs h_0. h_2 needs h_1. h_3 needs h_2. You CANNOT parallelize.
# If your sequence has 1000 tokens, you do 1000 steps serially.
# GPUs want to do 1000 things at once. RNNs say "no, one at a time."
#
# Transformers: attention at every position in parallel. 1000 tokens? Done in one GPU shot.

# %% [markdown]
# ### Limitation 2: VANISHING MEMORY
#
# By the time the RNN has processed 100 tokens, the effect of token 0
# has been multiplied and squashed 100 times. Almost zero influence left.
# The model "forgets" what happened early.
#
# LSTMs and GRUs tried to fix this with gating mechanisms.
# They helped — but the fundamental problem stays: information must flow
# through every time step.
#
# Transformers: every position can DIRECTLY attend to every other position.
# Token 100 can look right back at token 0 in ONE attention operation.

# %%
# Let's demonstrate the parallelism advantage
import time

seq_len = 5000
input_dim = 128
hidden_dim = 256

# RNN — sequential
rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
x = torch.randn(8, seq_len, input_dim)

t0 = time.time()
for _ in range(5):
    _ = rnn(x)
rnn_time = (time.time() - t0) / 5

# Transformer encoder layer — parallel
encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8, batch_first=True)

t0 = time.time()
for _ in range(5):
    _ = encoder_layer(x)
tfm_time = (time.time() - t0) / 5

print(f"RNN (seq_len={seq_len}):         {rnn_time*1000:.1f} ms")
print(f"Transformer (seq_len={seq_len}): {tfm_time*1000:.1f} ms")
print(f"\nOn short sequences the gap is small, but RNN scales LINEARLY with T,")
print(f"and Transformer does all T positions IN PARALLEL — huge win on GPUs.")

# %% [markdown]
# ## 6. Why this matters for your thesis
#
# Every VLM you'll study (CLIP, BLIP-2, LLaVA, Qwen2-VL) is built on
# transformers, NOT RNNs. Understanding WHY is important:
#
# - Documents can have 2000+ tokens → RNN would need 2000 sequential steps
# - Charts have long-range dependencies (label ↔ data point) → RNN forgets
# - Vision-language fusion needs parallelism → transformers scale, RNNs don't
#
# When you see the phrase "autoregressive generation" in papers, it uses
# transformers but in a LOOP (like RNN-style) ONLY at inference time,
# because you genuinely have to produce one token before you can produce the next.
# During training, transformers process ALL tokens in parallel using causal masking.
# This "parallel training + sequential inference" trick is a big deal.

# %% [markdown]
# ## Summary
#
# | Concept | Picture |
# |---------|---------|
# | **RNN cell** | A loop body that reads one token and updates memory |
# | **Hidden state** | The memory it carries forward |
# | **Time step (T)** | One loop iteration = one token processed |
# | **Why it died** | Sequential = slow; long-range memory = weak |
# | **What replaced it** | Transformers (parallel, direct attention) |
#
# When you see `(B, T, C)` in transformer code, `T` is a historical name —
# it comes from this loop. There's no actual "time" happening in a transformer;
# all T positions are processed at once.

# %%
