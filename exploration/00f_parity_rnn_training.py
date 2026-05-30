# %% [markdown]
# # The Parity Task — Training a `torch.nn.RNN` for Real
#
# ## What we're building
#
# Take a sequence of 0s and 1s of length `T`. The label is **1 if the
# count of 1s is odd, 0 if even** — i.e., parity.
#
# Example:
# ```
#   sequence = [1, 0, 1, 1, 0]    →   three 1s    →   label = 1 (odd)
#   sequence = [0, 1, 0, 1, 0]    →   two 1s      →   label = 0 (even)
# ```
#
# ## Why this task
#
# - **The RNN MUST use its memory.** You can't tell parity from just the
#   last token — you have to carry running info across the whole sequence.
# - **Tiny.** Input dim = 1, output dim = 1 (binary). Trains in seconds.
# - **Clear pass/fail.** If the model isn't using memory, accuracy stays
#   at chance (50%). If it is, you'll see it climb past 95%.
#
# ## Your job
#
# Build this end-to-end. Use `torch.nn.RNN` (NOT your `NaiveRNN`) so we
# can focus on the training mechanics. The file is structured as a
# guided checklist — every cell is a `TODO` with hints, no code. Fill
# them in as you go.

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 1: Imports and hyperparameters
# ─────────────────────────────────────────────────────────────────────
#
# %%
from math import e

from attr import dataclass
from numpy import diag
from torch import nn
import torch as t


t.manual_seed(42)

T = 10
hidden_dim = 32
batch_size = 64
learning_rate = 1
num_train_steps = 20_000

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 2: A function to generate a batch of (sequence, parity_label)
# ─────────────────────────────────────────────────────────────────────
#
# TODO: write a function `generate_batch(batch_size, seq_len)` that
# returns two tensors:
#
#     x of shape (batch_size, seq_len, 1)   ← each token is a scalar 0 or 1
#                                              (the trailing "1" is the
#                                              input_dim — even a scalar
#                                              input needs an explicit dim)
#     y of shape (batch_size,)              ← binary label, dtype float32
#                                              (parity = sum of x % 2)
#
# HINTS:
#   - torch.randint(0, 2, (batch_size, seq_len, 1)).float()  gives random 0/1.
#   - y is computed FROM x: sum each sequence along the time axis, mod 2.
#   - Make sure y is the right dtype for whatever loss you'll use later
#     (probably float for BCEWithLogitsLoss).
#
# GOTCHA: PyTorch's RNN expects shape (B, T, input_dim) when batch_first=True.
# Don't forget the trailing "1" on x.
def generate_batch(batch_size, seq_len):
    x = t.randint(0, 2, (batch_size, seq_len, 1)).float()  # shape: (B, T, 1)
    y = (x.sum(dim=1) % 2).squeeze(-1)  # shape: (B,) after squeezing
    return x, y

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 3: Define the model — nn.RNN + output head
# ─────────────────────────────────────────────────────────────────────
#
# TODO: build a small `nn.Module` (call it ParityRNN) that contains:
#         (a) an nn.RNN(input_size=1, hidden_size=hidden_dim, batch_first=True)
#         (b) an nn.Linear(hidden_dim, 1)         ← the output head
#
# In forward(self, x):
#   - run x through the RNN → get (output, h_n)
#   - extract the FINAL hidden state (because parity is a sequence-level
#     property — we only need one prediction per sequence, not one per token)
#   - run it through the Linear head → get a logit
#   - return the logit, shape (batch_size,) or (batch_size, 1) — pick one
#     and squeeze if needed
#
# HINTS:
#   - `output` from nn.RNN has shape (B, T, hidden_dim). The LAST time
#     step is output[:, -1, :].
#   - `h_n` from nn.RNN has shape (num_layers, B, hidden_dim). For
#     single-layer, h_n[0] gives shape (B, hidden_dim).
#   - Either works — pick whichever feels cleaner.
#   - The output head turns hidden_dim → 1 (single logit per sequence).
#   - Don't apply sigmoid here — let the LOSS function do that (more
#     numerically stable, see step 4).

from dataclasses import dataclass

@dataclass
class DiagnosticVal:
    epoch: int
    loss: float
    accuracy: float
    
    rnn_W_x_grad_mean: float
    rnn_W_x_b_grad_mean: float
    rnn_W_hist_grad_mean: float

    rnn_W_x_mean: float
    rnn_W_x_b_mean: float
    rnn_W_hist_mean: float

    lh_w_x_grad_mean: float
    lh_b_grad_mean: float

    lh_w_x_mean: float
    lh_b_mean: float

    prob_mean: float
    prob_std: float
    saturated_frac: float


class ParityRNN:
    def __init__(self, channel_dim: int, hidden_space_dim: int, batch_first: bool = True) -> None:
        self.W_x = (t.randn((hidden_dim, channel_dim))   * (2/channel_dim)**0.5).requires_grad_(True)  # shape: (hidden_dim, input_size)
        self.W_x_b = t.zeros((hidden_space_dim,), requires_grad=True)

        self.W_hist = (t.randn((hidden_dim, hidden_dim))    * (2/hidden_dim)**0.5).requires_grad_(True)
        self.activation = t.tanh
        self.batch_first = batch_first

    def forward(self, x):
        # x shape: (B, T, 1) if the batch_first=True, else (T, B, 1)
        if not self.batch_first:
            x = x.transpose(0, 1)  # convert to (B, T, 1) for consistency
        B, T, _ = x.shape

        # h is the memory state at current time_step for each item in the batch (B, hidden_dim)
        h = t.zeros((B, self.W_hist.shape[0]))  # initial memory is zero
        h_for_each_seq = []
        for seq_idx in range(T):
            out = x[:, seq_idx, :] @ self.W_x.T + self.W_x_b + h @ self.W_hist.T  # shape: (B, hidden_dim)
            h = self.activation(out)  # update memory
            h_for_each_seq.append(h.clone().detach())  # store hidden state for this time step (detach to avoid backprop through time)
        
        return h_for_each_seq, h

class ParityLinear:
    def __init__(self, in_dim: int, out_dim: int) -> None:
        self.W = (t.randn((out_dim, in_dim)) * (2/in_dim)**0.5).requires_grad_(True)  # shape: (out_dim, in_dim)
        self.b = t.zeros((out_dim,), requires_grad=True)
    
    def forward(self, x):
        return x @ self.W.T + self.b  # shape: (B, out_dim)


rnn = ParityRNN(channel_dim=1, hidden_space_dim=hidden_dim)
linear_head = ParityLinear(in_dim=hidden_dim, out_dim=1)
diagnostic_vals = []

for epoch in range(num_train_steps):
    x, y = generate_batch(batch_size, T)

    h_for_each_seq, final_hidden = rnn.forward(x)
    logit = linear_head.forward(final_hidden)  # shape: (B, 1)
    prob = t.sigmoid(logit)  # shape: (B, 1)
    loss = (-(y.unsqueeze(1) * t.log(prob) + (1 - y.unsqueeze(1)) * t.log(1 - prob))).mean()  # BCE loss
    
    # backpropagation
    loss.backward()

    if epoch and epoch % 100 == 0:
        predicted_class = (prob > 0.5).float().squeeze(1)  # shape: (B,)
        accuracy = (predicted_class == y).float().mean()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%")
        diagnostic_vals.append(DiagnosticVal(
            epoch=epoch,
            loss=loss.item(),
            accuracy=accuracy.item() * 100,
            rnn_W_x_grad_mean=rnn.W_x.grad.abs().mean().item(),
            rnn_W_x_b_grad_mean=rnn.W_x_b.grad.abs().mean().item(),
            rnn_W_hist_grad_mean=rnn.W_hist.grad.abs().mean().item(),
            rnn_W_x_mean=rnn.W_x.abs().mean().item(),
            rnn_W_x_b_mean=rnn.W_x_b.abs().mean().item(),
            rnn_W_hist_mean=rnn.W_hist.abs().mean().item(),
            lh_w_x_grad_mean=linear_head.W.grad.abs().mean().item(),
            lh_b_grad_mean=linear_head.b.grad.abs().mean().item(),
            lh_w_x_mean=linear_head.W.abs().mean().item(),
            lh_b_mean=linear_head.b.abs().mean().item(),
            prob_mean=prob.mean().item(),
            prob_std=prob.std().item(),
            saturated_frac=((final_hidden.abs() > 0.99)).float().mean().item()
        ))

    # update weights (simple SGD step)
    with t.no_grad():
        for param in [rnn.W_x, rnn.W_x_b, rnn.W_hist, linear_head.W, linear_head.b]:
            param -= learning_rate * param.grad
            param.grad.zero_()  # clear gradients for the next step


# %% [markdown]
# ## Diagnostic charts

# %%
from matplotlib import pyplot as plt

epochs = [d.epoch for d in diagnostic_vals]
losses = [d.loss for d in diagnostic_vals]
accuracies = [d.accuracy for d in diagnostic_vals]

grad_W_x = [d.rnn_W_x_grad_mean for d in diagnostic_vals]
grad_W_hist = [d.rnn_W_hist_grad_mean for d in diagnostic_vals]
grad_lin_W = [d.lh_w_x_grad_mean for d in diagnostic_vals]

w_x = [d.rnn_W_x_mean for d in diagnostic_vals]
w_x_b = [d.rnn_W_x_b_mean for d in diagnostic_vals]
w_hist = [d.rnn_W_hist_mean for d in diagnostic_vals]

lin_w_x = [d.lh_w_x_mean for d in diagnostic_vals]
lin_b = [d.lh_b_mean for d in diagnostic_vals]


prob_mean = [d.prob_mean for d in diagnostic_vals]
prob_std = [d.prob_std for d in diagnostic_vals]
saturated_frac = [d.saturated_frac for d in diagnostic_vals]


fig, axes = plt.subplots(3, 2, figsize=(12, 9))

axes[0, 0].plot(epochs, losses)
axes[0, 0].set_title("Loss")
axes[0, 0].axhline(0.69, color="red", linestyle="--", label="ln(2) chance")
axes[0, 0].legend()


axes[0, 1].plot(epochs, accuracies)
axes[0, 1].set_title("Accuracy (%)")


axes[1, 0].plot(epochs, grad_W_x,    label="W_x")
axes[1, 0].plot(epochs, grad_W_hist, label="W_hist")
axes[1, 0].plot(epochs, grad_lin_W,  label="lin.W")
axes[1, 0].set_yscale("log")
axes[1, 0].set_title("|grad| (log scale)")
axes[1, 0].legend()


axes[1, 1].plot(epochs, w_x, label="W_x")
axes[1, 1].plot(epochs, w_x_b, label="W_x_b")
axes[1, 1].plot(epochs, w_hist, label="W_hist")
axes[1, 1].set_title("Mean abs weight value")
axes[1, 1].legend()


axes[2, 0].plot(epochs, lin_w_x, label="lin.W")
axes[2, 0].plot(epochs, lin_b, label="lin.b")
axes[2, 0].set_title("Mean abs weight value (linear head)")
axes[2, 0].legend()


axes[2, 1].plot(epochs, prob_mean, label="mean prob")
axes[2, 1].plot(epochs, prob_std, label="std prob")
axes[2, 1].plot(epochs, saturated_frac, label="saturated hidden frac")
axes[2, 1].set_title("Output prob stats")
axes[2, 1].legend()


plt.tight_layout()
plt.show()

# %% [markdown]
# ## Final evaluation on fresh data
# %%
test_batch = generate_batch(batch_size, T)
test_x, test_y = test_batch

with t.no_grad():
    h_for_each_seq, final_hidden = rnn.forward(test_x)
    logit = linear_head.forward(final_hidden)
    prob = t.sigmoid(logit)
    predicted_class = (prob > 0.5).float().squeeze(1)
    accuracy = (predicted_class == test_y).float().mean()
    print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")




















# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 4: Loss function
# ─────────────────────────────────────────────────────────────────────
#
# TODO: define `criterion = nn.BCEWithLogitsLoss()`
#
# WHY this one specifically (think about it before reading):
#   - We want a binary prediction (parity is 0 or 1).
#   - The model outputs a single logit per sequence (not a probability).
#   - BCEWithLogitsLoss expects raw logits AND applies sigmoid internally,
#     more numerically stable than doing sigmoid → BCELoss yourself.
#
# If you used 2 output dims instead, you'd use nn.CrossEntropyLoss.
# Either choice works; BCEWithLogitsLoss is simpler for binary tasks.
#
# GOTCHA: the loss expects model output shape and target shape to MATCH.
# If your model outputs (B, 1) but y is (B,), reshape one of them
# (e.g., `logits.squeeze(-1)` to turn (B, 1) into (B,)).

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 5: Optimizer
# ─────────────────────────────────────────────────────────────────────
#
# TODO: create an optimizer using torch.optim.Adam, passing
# `model.parameters()` and the learning rate from step 1.
#
# WHY Adam (vs SGD):
#   - Adam adapts the per-parameter learning rate. For a simple RNN on
#     a simple task, it converges much faster than vanilla SGD.
#   - SGD with momentum would also work but needs more tuning.
#
# HINT: model.parameters() works because the model is an nn.Module
# wrapping nn.RNN and nn.Linear — both register their weights as
# nn.Parameter automatically. No manual list-building needed.

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 6: The training loop
# ─────────────────────────────────────────────────────────────────────
#
# TODO: write a loop that runs for `num_train_steps` iterations.
#
# Each iteration:
#   1. Generate a fresh batch with generate_batch(batch_size, T)
#   2. Forward pass: logits = model(x)
#   3. Compute loss: criterion(logits, y)
#   4. optimizer.zero_grad()       ← clear old grads BEFORE backward
#   5. loss.backward()             ← compute gradients
#   6. optimizer.step()            ← update weights
#   7. Every N steps, print the loss and an accuracy check
#
# HINTS:
#   - Shape match: see the gotcha in step 4.
#   - For accuracy: predicted_class = (sigmoid(logits) > 0.5).float()
#     then compare with y, take .mean() to get accuracy. Or skip sigmoid
#     and check (logits > 0).float() — same boundary at 0.5 probability.
#   - Expected behavior: loss starts ~0.69 (= ln(2), random binary classifier),
#     drops over a few hundred steps, accuracy climbs from 50% to >95%.

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 7: Evaluation on fresh data
# ─────────────────────────────────────────────────────────────────────
#
# TODO: after training, generate a NEW batch (call it test_x, test_y)
# the model has never seen, run forward, and report final accuracy.
#
# Wrap the forward pass in `with torch.no_grad():` — disables gradient
# tracking (a bit faster, no memory wasted on the autograd graph
# during inference).
#
# If accuracy ≥ 95%, the RNN learned to maintain parity through time. 🎉
# If accuracy ≈ 50%, something's wrong:
#   - Check shape mismatches (y vs logits)
#   - Check that you're using the FINAL hidden state, not the first
#   - Check learning rate (too high → diverges; too low → won't learn in time)
#   - Check data: print a few x[0] and y[0] pairs to confirm labels are right

# %%
# ─────────────────────────────────────────────────────────────────────
# STEP 8 (BONUS): Watch the RNN struggle with longer sequences
# ─────────────────────────────────────────────────────────────────────
#
# TODO: re-train with sequence length T = 30, then T = 50, then T = 100.
# Same model size, same training budget. Watch accuracy drop.
#
# WHY this happens:
#   - Vanilla RNN suffers from "vanishing gradients" — when you do
#     backprop through T time steps, gradient signal multiplies through
#     T tanh derivatives and one shared W_h matrix. The product
#     shrinks (or explodes) and the model can't learn long-range
#     dependencies.
#   - LSTMs and GRUs solve this with gating. Try nn.LSTM or nn.GRU as
#     a drop-in replacement and notice they handle longer sequences
#     much better.
#
# This is exactly why transformers won — they bypass the sequential
# bottleneck entirely.

# %% [markdown]
# ## Checklist (run through this at the end)
#
# - [ ] Imports done, seed set, hyperparameters defined.
# - [ ] `generate_batch` returns (x, y) with the right shapes and dtypes.
# - [ ] Model is an `nn.Module` containing `nn.RNN` + `nn.Linear`.
# - [ ] Forward returns a logit per sequence (not per token).
# - [ ] Loss is BCEWithLogitsLoss with shape-matched logits and labels.
# - [ ] Optimizer is Adam over `model.parameters()`.
# - [ ] Training loop: zero_grad → forward → loss → backward → step.
# - [ ] Loss prints every N steps; accuracy is reported.
# - [ ] After training, fresh-data evaluation reaches ≥95% accuracy.
# - [ ] (Bonus) tried longer sequences and saw degradation.
#
# When you're done — or stuck — ping me and I'll check your code.
