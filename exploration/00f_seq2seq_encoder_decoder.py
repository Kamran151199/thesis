# %% [markdown]
# # Day 0 (part F): Seq2Seq — The Pre-Transformer Encoder-Decoder
#
# Before transformers, people stacked TWO RNNs together to do translation:
#   one RNN READS the input (encoder)
#   one RNN WRITES the output (decoder)
#
# This is where "encoder" and "decoder" as architectural terms come from.
# The transformer kept these names and just replaced the RNN internals.
#
# We'll build a tiny translator that maps numbers to their reversed form:
#   input:  [1, 2, 3, 4]  →  output: [4, 3, 2, 1]
#
# Trivially solvable by hand, but it exercises every part of seq2seq.

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

torch.manual_seed(42)

# %% [markdown]
# ## 1. The encoder: reads the input, compresses into ONE vector
#
# The original 2014 seq2seq idea was dead simple:
#   1. Feed input sequence into an RNN, ONE TOKEN AT A TIME
#   2. Take the FINAL hidden state — this is the "context vector"
#   3. Throw away the per-step outputs, keep only the final h
#
# The context vector is supposed to hold ALL the information about the input.
# (Spoiler: this turns out to be a terrible bottleneck for long sequences.)

# %%
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        """
        x shape: (B, T_src)       source sequence of token indices
        returns: (B, hidden_dim)  FINAL hidden state — the "context vector"
                 (B, T_src, hidden_dim)  all hidden states (used by attention)
        """
        embedded = self.embedding(x)           # (B, T_src, embed_dim)
        all_hidden, last_hidden = self.rnn(embedded)
        # last_hidden shape: (1, B, hidden_dim) — squeeze the first dim
        return all_hidden, last_hidden.squeeze(0)

# %% [markdown]
# ## 2. The decoder: writes the output, one token at a time
#
# The decoder is ANOTHER RNN. It runs in a loop:
#   - Start with a special <SOS> token and the encoder's context vector
#   - Produce a token
#   - Feed that token back in as the next input
#   - Repeat until <EOS>
#
# In the naive 2014 version, the decoder ONLY has access to the final
# context vector from the encoder. It never re-reads the input.

# %%
class NaiveDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, init_hidden):
        """
        tgt shape: (B, T_tgt)           — ground-truth output for teacher forcing
        init_hidden: (B, hidden_dim)    — encoder's final hidden state (CONTEXT)
        """
        embedded = self.embedding(tgt)
        # init_hidden needs an extra dim for GRU: (1, B, hidden_dim)
        h = init_hidden.unsqueeze(0)
        outputs, _ = self.rnn(embedded, h)
        logits = self.output(outputs)   # (B, T_tgt, vocab_size)
        return logits

# %% [markdown]
# ## 3. Put them together — the original 2014 seq2seq
#
# Task: reverse a sequence of digits.
# Vocabulary: 0-9 plus SOS (10), EOS (11), PAD (12).

# %%
VOCAB = 13
SOS, EOS, PAD = 10, 11, 12

def make_batch(batch_size=32, length=5):
    """Generate random digit sequences and their reversals."""
    src = torch.randint(0, 10, (batch_size, length))
    tgt = torch.flip(src, dims=[1])
    # Prepend SOS to decoder input, append EOS to target
    decoder_input = torch.cat([torch.full((batch_size, 1), SOS), tgt], dim=1)
    decoder_target = torch.cat([tgt, torch.full((batch_size, 1), EOS)], dim=1)
    return src, decoder_input, decoder_target

encoder = Encoder(vocab_size=VOCAB, embed_dim=16, hidden_dim=32)
decoder = NaiveDecoder(vocab_size=VOCAB, embed_dim=16, hidden_dim=32)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()

losses = []
for step in range(2000):
    src, dec_in, dec_tgt = make_batch(batch_size=64, length=5)
    encoder_outputs, context = encoder(src)
    logits = decoder(dec_in, context)  # (B, T, vocab)
    loss = loss_fn(logits.reshape(-1, VOCAB), dec_tgt.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if step % 500 == 0:
        print(f"step {step}: loss = {loss.item():.4f}")

plt.plot(losses)
plt.xlabel('step')
plt.ylabel('loss')
plt.title('Seq2seq training — reverse-the-sequence task')
plt.show()

# %% [markdown]
# ## 4. Test the trained model

# %%
def translate(src_seq):
    """Greedy decoding — pick the max-prob token at each step."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        x = torch.tensor([src_seq])
        _, context = encoder(x)
        h = context.unsqueeze(0)

        # Start with SOS
        token = torch.tensor([[SOS]])
        output_seq = []

        for _ in range(10):  # max output length
            emb = decoder.embedding(token)
            out, h = decoder.rnn(emb, h)
            logits = decoder.output(out[:, -1])
            token = logits.argmax(dim=-1, keepdim=True)
            if token.item() == EOS:
                break
            output_seq.append(token.item())

        return output_seq

print("\nTest — can it reverse sequences?")
for test_input in [[1, 2, 3, 4, 5], [9, 0, 3, 7, 2], [5, 5, 5, 5, 5]]:
    result = translate(test_input)
    print(f"  input:  {test_input}")
    print(f"  output: {result}")
    print(f"  truth:  {test_input[::-1]}")
    print()

# %% [markdown]
# ## 5. Visualize the architecture

# %%
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Encoder side
ax.text(3, 5.5, "ENCODER RNN", fontsize=14, fontweight='bold', ha='center', color='darkblue')
for i in range(4):
    ax.add_patch(Rectangle((0.5 + i*1.3, 3), 1, 1, facecolor='lightblue', edgecolor='black'))
    ax.text(1 + i*1.3, 3.5, f"h_{i}", ha='center', va='center', fontsize=10)
    ax.text(1 + i*1.3, 2.5, f"x_{i}", ha='center', color='blue', fontsize=10)
    ax.annotate('', xy=(1 + i*1.3, 3), xytext=(1 + i*1.3, 2.7),
                arrowprops=dict(arrowstyle='->', color='blue'))
    if i < 3:
        ax.annotate('', xy=(0.5 + (i+1)*1.3, 3.5), xytext=(1.5 + i*1.3, 3.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# The context vector — the bottleneck
ax.annotate('', xy=(7.5, 3.5), xytext=(5.4, 3.5),
            arrowprops=dict(arrowstyle='->', color='darkred', lw=3))
ax.text(6.5, 4, "CONTEXT VECTOR\n(the bottleneck!)", ha='center', color='darkred',
        fontsize=10, fontweight='bold')

# Decoder side
ax.text(11, 5.5, "DECODER RNN", fontsize=14, fontweight='bold', ha='center', color='darkgreen')
for i in range(4):
    ax.add_patch(Rectangle((8 + i*1.3, 3), 1, 1, facecolor='lightgreen', edgecolor='black'))
    ax.text(8.5 + i*1.3, 3.5, f"s_{i}", ha='center', va='center', fontsize=10)
    ax.text(8.5 + i*1.3, 5, f"y_{i}", ha='center', color='darkgreen', fontsize=10)
    ax.annotate('', xy=(8.5 + i*1.3, 4.8), xytext=(8.5 + i*1.3, 4),
                arrowprops=dict(arrowstyle='->', color='darkgreen'))
    if i < 3:
        ax.annotate('', xy=(8 + (i+1)*1.3, 3.5), xytext=(9 + i*1.3, 3.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax.text(7, 1, "The decoder generates y_0, y_1, y_2... ONLY using the final h_3.\n"
              "The source tokens are SQUEEZED into one vector — this is the bottleneck.",
        ha='center', fontsize=11, style='italic')

plt.show()

# %% [markdown]
# ## 6. The fatal flaw: the context vector bottleneck
#
# The encoder reads the entire input and compresses it into a SINGLE fixed-size vector.
# For a 5-word sentence: probably fine.
# For a 50-word sentence: the context vector has to memorize everything. It can't.
# For a 500-word document: totally hopeless.
#
# Translation quality DROPPED DRAMATICALLY with longer inputs. This was THE problem
# that motivated the next evolution.

# %% [markdown]
# ## 7. The fix: attention (Bahdanau 2015)
#
# Instead of compressing everything into h_final, keep ALL the encoder hidden states:
#   h_0, h_1, h_2, ..., h_{T_src}
#
# At each DECODER step, let the decoder LOOK BACK at all encoder states and
# decide which ones are relevant RIGHT NOW.
#
# This is cross-attention — the same mechanism used in the original Transformer
# and in every modern VLM (Q-Former, LLaVA's MLP projection is a degenerate form).

# %%
class AttentionDecoder(nn.Module):
    """Bahdanau-style additive attention. Simplified for clarity."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, encoder_outputs, init_hidden):
        """
        tgt:              (B, T_tgt)        ground-truth decoder input
        encoder_outputs:  (B, T_src, H)     ALL encoder hidden states
        init_hidden:      (B, H)            encoder's final hidden state
        """
        B, T_tgt = tgt.shape
        H = encoder_outputs.shape[-1]
        h = init_hidden.unsqueeze(0)
        logits_all = []

        for t in range(T_tgt):
            y_t = self.embedding(tgt[:, t:t+1])        # (B, 1, E)
            # Attention: decoder looks at all encoder positions
            h_broadcast = h[-1].unsqueeze(1).expand(-1, encoder_outputs.shape[1], -1)
            combined = torch.cat([encoder_outputs, h_broadcast], dim=-1)
            scores = self.attention(torch.tanh(combined)).squeeze(-1)  # (B, T_src)
            weights = torch.softmax(scores, dim=-1).unsqueeze(1)        # (B, 1, T_src)
            context_t = weights @ encoder_outputs                        # (B, 1, H)

            # Feed [input_token, context] into RNN
            rnn_input = torch.cat([y_t, context_t], dim=-1)
            out, h = self.rnn(rnn_input, h)
            logits_all.append(self.output(out))

        return torch.cat(logits_all, dim=1)

print("The AttentionDecoder above is Bahdanau's 2015 innovation.")
print("Every decoder step computes its OWN context by looking back at ALL encoder states.")
print()
print("This attention mechanism is what the Transformer (2017) generalized:")
print("  - Self-attention: tokens attend to OTHER tokens in the SAME sequence")
print("  - Cross-attention: decoder attends to encoder — SAME as Bahdanau, just cleaner math")

# %% [markdown]
# ## 8. The lineage that will show up in your thesis
#
# ```
# 2014  seq2seq (plain RNN + RNN)            → one context vector bottleneck
#    ↓
# 2015  Bahdanau attention                    → decoder looks back at all encoder states
#    ↓
# 2017  Transformer                           → replace RNNs with attention EVERYWHERE
#                                               (self-attention within encoder,
#                                                self-attention within decoder,
#                                                cross-attention between them)
#    ↓
# 2018  BERT (encoder only) + GPT (decoder only)
#    ↓
# 2022  BLIP-2 (encoder + Q-Former with cross-attention to frozen LLM)
#    ↓
# 2023+ LLaVA, Qwen2-VL (ViT encoder → projection → decoder-only LLM)
# ```
#
# **Why this matters for your thesis:**
#
# - **Donut** (document understanding) uses encoder-decoder transformer
#   → direct descendant of what you just built
# - **Pix2Struct** (chart understanding) is encoder-decoder
# - **BLIP-2's Q-Former** uses cross-attention — the EXACT mechanism
#   you implemented in AttentionDecoder above, just with transformer layers
# - **LLaVA** is decoder-only — it skips having a "decoder reading encoder"
#   because the image tokens are just prefix tokens in the decoder's input
#
# The seq2seq lineage you just touched is 50% of your thesis landscape.

# %% [markdown]
# ## Summary
#
# | Year | Model | What's new |
# |------|-------|-----------|
# | 2014 | Seq2seq (Sutskever) | Encoder-decoder CONCEPT with RNNs |
# | 2015 | Bahdanau attention | Decoder peeks at all encoder states |
# | 2017 | Transformer | Attention replaces recurrence entirely |
# | 2018 | BERT / GPT | Encoder-only / Decoder-only split |
# | 2022 | BLIP-2 | Q-Former cross-attends to frozen vision encoder |
# | 2023 | LLaVA | Decoder-only LLM with visual prefix tokens |
#
# The "encoder" and "decoder" concepts you hear about daily in ML were invented
# for RNN translation models. The transformer inherited the concepts and upgraded
# the internals. Modern VLMs pick and choose between encoder, decoder, or both.
