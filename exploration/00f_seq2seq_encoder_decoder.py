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

from IPython import embed
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
class NaiveRNN:
    def __init__(self, channel_dim: int, hidden_dim: int, batch_first: bool = True) -> None:
        self.batch_first = batch_first
        # weights that translate the input token (channel_dim input space) into the hidden state (output space)
        self.W_x = (torch.randn(hidden_dim, channel_dim) * 1 / channel_dim ** 0.5).requires_grad_(True)
        self.W_x_b = torch.zeros((hidden_dim,), requires_grad=True)

        self.W_hist = (torch.randn((hidden_dim, hidden_dim)) * 1 / hidden_dim ** 0.5).requires_grad_(True)        
        self.activation = torch.tanh


    def forward(
            self,
            sequence: torch.Tensor,
            state: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sequence shape: (B, T, channel_dim) - (100, 20, 10)
        returns: list of hidden states at every time step
        """

        if not self.batch_first:
            sequence = sequence.transpose(1, 0)

        T = sequence.shape[1]  # 10
        B = sequence.shape[0]  # 100

        if state is None:
            h = torch.zeros((B, self.W_hist.shape[1]))  # initial memory is zero (100, 10) zeros
        else:
            h = state

        all_hidden_states = []

        for t in range(T):
            x_t = sequence[:, t, :]  # (100, single-token-at-time, 10)

            h = self.activation(x_t @ self.W_x.T + self.W_x_b + h @ self.W_hist.T)  # (B, HiddenSpaceDim)
            all_hidden_states.append(h)

        return torch.stack(all_hidden_states, 1), h

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [
            self.W_x,
            self.W_x_b,
            self.W_hist
        ]
    
    @property
    def parameters_dict(self) -> dict[str, torch.Tensor]:
        return dict(
            W_x=self.W_x,
            W_x_b=self.W_x_b,
            W_h=self.W_hist
        )
    

# %%
class NaiveEncoder:
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int,
            hidden_dim: int,
            batch_first: bool
        ):
        """Initializer for the encoder

        Attributes:
            vocab_size: The dimensionality of a single token (kinda like the representation of the token in the input space)
            embed_dim: The dimensionality of a single input token in our "derived/learned" space.
            hidden_dim: dimensionality of the information storage in RNN.
            batch_first: If True, the input shape is (B, T, C) else (T, B, C) 
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        
        self.embedding = (torch.randn((vocab_size, embed_dim)) * 1/vocab_size ** 0.5).requires_grad_(True)
        self.rnn = NaiveRNN(embed_dim, hidden_dim, batch_first=True)
    
    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            sequence = sequence.transpose(1, 0)  # if input is (T, B, C) make it (B, T, C)
        B = sequence.shape[0]
        T = sequence.shape[1]

        # if the sequence was one-hot-encoded or multi-dim by design, then we would use this:
        # embedded = sequence @ self.embedding.T

        # since our sequence is just a list of integers, we gonna use the following:
        embedded = self.embedding[sequence]

        # remember that these are plural in terms of the "Batch" - aka B dim.
        whole_seq_storages, last_token_storages = self.rnn.forward(embedded, None)

        assert whole_seq_storages.shape == (B, T, self.hidden_dim)
        assert last_token_storages.shape == (B, self.hidden_dim)  # only the channel/output representation of the last token in each batch
        return whole_seq_storages, last_token_storages

    @property
    def parameters_dict(self) -> dict[str, dict | torch.Tensor]:
        return dict(
            embedding=self.embedding,
            rnn=self.rnn.parameters_dict
        )
    
    @property
    def parameters(self) -> list[torch.Tensor]:
        return [
            self.embedding,
            *self.rnn.parameters
        ]


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
class NaiveDecoder:
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, batch_first: bool = True):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first

        self.embedding = (torch.randn(size=(vocab_size, embed_dim)) * 1 / vocab_size ** 0.5).requires_grad_(True)
        self.rnn = NaiveRNN(embed_dim, hidden_dim, batch_first)

        # this is the function that will calculate the logits based on the hidden state of the RNN
        self.linear = (torch.randn(size=(vocab_size, hidden_dim)) * 1 / hidden_dim ** 0.5).requires_grad_(True)
        self.linear_b = torch.zeros(size=(vocab_size, )).requires_grad_(True)

    def forward(self, tgt: torch.Tensor, init_hidden: torch.Tensor) -> torch.Tensor:
        """
        tgt shape: (B, T_tgt)           — ground-truth output for teacher forcing
        init_hidden: (B, hidden_dim)    — encoder's final hidden state (CONTEXT)
        """

        if not self.batch_first:
            tgt = tgt.transpose(1, 0)  # turn the sequence to (B, T, C)

        B = tgt.shape[0]
        T = tgt.shape[1]
        
        embedded_seq = self.embedding[tgt]

        batched_states, last_state = self.rnn.forward(embedded_seq, init_hidden)

        assert batched_states.shape == (B, T, self.hidden_dim)
        assert last_state.shape == (B, self.hidden_dim)
        # Each row of self.linear is a LEARNED PROTOTYPE for one vocab token (in hidden space).
        # This matmul dot-products the hidden state with each token's prototype,
        # producing one alignment score per vocab token = vocab_size logits per step
        logits = batched_states @ self.linear.T + self.linear_b  
        return logits
    
    @property
    def parameters_dict(self) -> dict[str, dict | torch.Tensor]:
        return dict(
            embedding=self.embedding,
            rnn=self.rnn.parameters_dict,
            linear_x=self.linear,
            linear_b=self.linear_b
        )
    
    @property
    def parameters(self) -> list[torch.Tensor]:
        return [
            self.embedding,
            *self.rnn.parameters,
            self.linear,
            self.linear_b
        ]


def make_batch(batch_size: int = batch_size, length: int = sequence_len) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random digit sequences and their reversals."""
    src = torch.randint(0, 10, (batch_size, length))  # gives us the (B, T) of random numbers from 0 to 10
    tgt = torch.flip(src, dims=[1])  # group truth (T, ) - flipped version of the src

    # Prepend SOS to decoder input, append EOS to target
    decoder_input = torch.cat([torch.full((batch_size, 1), SOS), tgt], dim=1)  # create the inputs for the decoder. Decoder should have some kinda "start" token, hence, we have [SOS] + [TargetGroundTruth] 
    decoder_target = torch.cat([tgt, torch.full((batch_size, 1), EOS)], dim=1)  # create the outputs/labels for the decoder. Decoder should end with EOS token, hence we have [TargetGroundTruth] + [EOS]
    return src, decoder_input, decoder_target

# %% [markdown]
# ## 3. Put them together — the original 2014 seq2seq
#
# Task: reverse a sequence of digits.
# Vocabulary: 0-9 plus SOS (10), EOS (11), PAD (12).

# %%
VOCAB = 13
SOS, EOS, PAD = 10, 11, 12

epochs = 20_000
lr = 0.1
embedding_dim = 20
hidden_dim = 80
batch_size = 32
sequence_len = 20




encoder = NaiveEncoder(VOCAB, embedding_dim,  hidden_dim, True)
decoder = NaiveDecoder(VOCAB, embedding_dim, hidden_dim, True)
metrics = []

for epoch in range(epochs):
    batch = make_batch()

    batch_src, batch_decoder_in, batch_decoder_target = batch

    enc_whole_seq_states, enc_last_states = encoder.forward(batch_src)
    logits = decoder.forward(batch_decoder_in, enc_last_states)

    # this is the softmax in torch - just for reference
    # batch_probs = torch.softmax(logits, -1)
    batch_probs = torch.exp(logits) / torch.exp(logits).sum(dim=-1, keepdim=True)

    # now the batch_probs are still Batch of Sequences where each token is a VOCAB_SIZE vector (probability for each char)
    assert batch_probs.shape == (batch_size, sequence_len + 1, VOCAB)

    # but we need to have the single probability per token.
    # to achieve that, we need to get all the probability vectors (regardless of the sequence and batch - at this stage of logic)
    all_prob_vectors = batch_probs.reshape(batch_size * (sequence_len + 1), VOCAB)
    all_targets = batch_decoder_target.reshape(batch_size * (sequence_len + 1), )

    probs = all_prob_vectors[torch.arange(batch_size * (sequence_len + 1)), all_targets]
    loss = -torch.log(probs).mean()

    # Backprop
    for param in [encoder.parameters, decoder.parameters]:
        for p in param:
            p.grad = None

    loss.backward()


    # log
    if epoch % 100 == 0:
        acc = (probs > 0.5).float().mean().item()  # very rough accuracy estimate
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    
        # collect metrics
        metrics.append({
            "epoch": epoch,
            "loss": loss.item(),

            # encoder
            "encoder__rnn__W_x__grad_mean": encoder.parameters_dict['rnn']['W_x'].grad.abs().mean().item(),
            "encoder__rnn__W_h__grad_mean": encoder.parameters_dict['rnn']['W_h'].grad.abs().mean().item(),
            "encoder__rnn__W_x__weights_mean": encoder.parameters_dict['rnn']['W_x'].abs().mean().item(),
            "encoder__rnn__W_h__weights_mean": encoder.parameters_dict['rnn']['W_h'].abs().mean().item(),
            

            # decoder
            "decoder__rnn__W_x__grad_mean": decoder.parameters_dict['rnn']['W_x'].grad.abs().mean().item(),
            "decoder__rnn__W_h__grad_mean": decoder.parameters_dict['rnn']['W_h'].grad.abs().mean().item(),
            "decoder__rnn__W_x__weights_mean": decoder.parameters_dict['rnn']['W_x'].abs().mean().item(),
            "decoder__rnn__W_h__weights_mean": decoder.parameters_dict['rnn']['W_h'].abs().mean().item(),

            "decoder__linear__W_x__grad_mean": decoder.parameters_dict['linear_x'].grad.abs().mean().item(),
            "decoder__linear__W_x__weights_mean": decoder.parameters_dict['linear_x'].abs().mean().item(),

            "prob_mean": probs.mean().item(),
            "prob_std": probs.std().item(),
        })

    # Gradient descent step
    for param in [encoder.parameters, decoder.parameters]:
        for p in param:
            p.data -= lr * p.grad


# %% [markdown]
# ## 3.5. Metrics and training dynamics

# %%

from matplotlib import pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(12, 8))

# plot loss
ax[0, 0].plot([m['epoch'] for m in metrics], [m['loss'] for m in metrics])
ax[0, 0].set_title("Loss")

# plot encoder W_x gradients
ax[0, 1].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_x__grad_mean'] for m in metrics], label='encoder W_x grad')
ax[0, 1].set_title("Encoder W_x Gradients")
ax[0, 1].set_yscale('log')

# plot encoder W_x weights
ax[0, 2].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_x__weights_mean'] for m in metrics], label='encoder W_x weights')
ax[0, 2].set_title("Encoder W_x Weights")
ax[0, 2].set_yscale('log')

# plot encoder W_h gradients
ax[1, 0].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_h__grad_mean'] for m in metrics], label='encoder W_h grad')
ax[1, 0].set_title("Encoder W_h Gradients")
ax[1, 0].set_yscale('log')

# plot encoder W_h weights
ax[1, 1].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_h__weights_mean'] for m in metrics], label='encoder W_h weights')
ax[1, 1].set_title("Encoder W_h Weights")
ax[1, 1].set_yscale('log')


# plot decoder W_x gradients
ax[1, 2].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_x__grad_mean'] for m in metrics], label='decoder W_x grad')
ax[1, 2].set_title("Decoder W_x Gradients")
ax[1, 2].set_yscale('log')

# plot decoder W_x weights
ax[2, 0].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_x__weights_mean'] for m in metrics], label='decoder W_x weights')
ax[2, 0].set_title("Decoder W_x Weights")
ax[2, 0].set_yscale('log')

# plot decoder W_h gradients
ax[2, 1].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_h__grad_mean'] for m in metrics], label='decoder W_h grad')
ax[2, 1].set_title("Decoder W_h Gradients")
ax[2, 1].set_yscale('log')

# plot decoder W_h weights
ax[2, 2].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_h__weights_mean'] for m in metrics], label='decoder W_h weights')
ax[2, 2].set_title("Decoder W_h Weights")
ax[2, 2].set_yscale('log')

plt.tight_layout()



def show_eigenvalues(W_h, name="W_h"):
    eigenvalues = torch.linalg.eigvals(W_h.detach())   # complex tensor, shape (hidden_dim,)
    real = eigenvalues.real.numpy()
    imag = eigenvalues.imag.numpy()
    magnitudes = eigenvalues.abs().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter on complex plane vs unit circle
    theta = torch.linspace(0, 2*torch.pi, 200)
    axes[0].plot(torch.cos(theta), torch.sin(theta), 'k--', label='unit circle (|λ|=1)')
    axes[0].scatter(real, imag, c=magnitudes, cmap='RdYlGn_r', s=40)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].axvline(0, color='gray', lw=0.5)
    axes[0].set_xlabel('Re(λ)')
    axes[0].set_ylabel('Im(λ)')
    axes[0].set_title(f'{name} eigenvalues — complex plane')
    axes[0].set_aspect('equal')
    axes[0].legend()

    # Right: histogram of magnitudes
    axes[1].hist(magnitudes, bins=20)
    axes[1].axvline(1.0, color='red', linestyle='--', label='|λ|=1 (preserves)')
    axes[1].set_xlabel('|λ|')
    axes[1].set_ylabel('count')
    axes[1].set_title(f'{name} — distribution of |λ|')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"max |λ|:  {magnitudes.max():.3f}")
    print(f"min |λ|:  {magnitudes.min():.3f}")
    print(f"mean |λ|: {magnitudes.mean():.3f}")


show_eigenvalues(encoder.parameters_dict['rnn']['W_h'], name="Encoder W_h")
show_eigenvalues(decoder.parameters_dict['rnn']['W_h'], name="Decoder W_h")
# %% [markdown]
# ## 4. Test the trained model

# %%
def greedy_decode(encoder, decoder, src, max_len=10):
    enc_whole_seq_states, enc_last_states = encoder.forward(src)
    batch_size = src.shape[0]
    decoder_input = torch.full((batch_size, 1), SOS, dtype=torch.long)  # start with SOS token
    hidden_state = enc_last_states

    generated_tokens = []

    for _ in range(max_len):
        logits = decoder.forward(decoder_input, hidden_state)
        next_token = logits.argmax(dim=-1)[:, -1]  # get the last token's logits and pick the highest one
        generated_tokens.append(next_token)

        if (next_token == EOS).all():
            break

        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)  # append the predicted token to the input for the next step

    return torch.stack(generated_tokens, dim=1)


sample_src = torch.tensor([[8, 5, 6, 7, 8], [4, 7, 1, 7, 3]])
generated = greedy_decode(encoder, decoder, sample_src)
print("Input:\n", sample_src)
print("Generated output:\n", generated)

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
class AttentionDecoder:
    """Bahdanau-style additive attention. Simplified for clarity."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        self.embedding = (torch.randn(vocab_size, embed_dim) * 1 / vocab_size ** 0.5).requires_grad_(True)
        self.rnn = NaiveRNN(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = (torch.randn(1, hidden_dim * 2) * 1 / (hidden_dim * 2) ** 0.5).requires_grad_(True)
        self.output = (torch.randn(vocab_size, hidden_dim) * 1 / hidden_dim ** 0.5).requires_grad_(True)

    def forward(self, tgt, encoder_outputs, init_hidden):
        """
        tgt:              (B, T_tgt)        ground-truth decoder input
        encoder_outputs:  (B, T_src, H)     ALL encoder hidden states
        init_hidden:      (B, H)            encoder's final hidden state and decoder's initial hidden state
        """
        B, T_tgt = tgt.shape
        H = encoder_outputs.shape[-1]
        T = encoder_outputs.shape[-2]
        h = init_hidden # (B, H)
        logits_all = []

        for t in range(T_tgt):
            y_t = self.embedding[tgt[:, t:t+1]]         # (B, 1, E)
            # Attention: decoder looks at all encoder positions
            # h[-1] is the current hidden state of the decoder (starts as init_hidden, then updates each step)
            h_broadcast = h.unsqueeze(1).expand(-1, encoder_outputs.shape[1], -1)  # (B, T_src, H) - we need to compare the current decoder state with each encoder state, so we broadcast it across the T_src dimension

            # "combined" is the collection of hidden states for each token 
            combined = torch.cat([encoder_outputs, h_broadcast], dim=-1)  # (B, T_src, 2H)

            # attention is (1, 2H) <- takes hidden states and tells how important that state is!
            activated = torch.tanh(combined)  # (B, T_src, 2H)
            scores = (activated @ self.attention.T ).squeeze(-1)  # (B, T_src)

            # normalized importance score for each token
            weights = torch.softmax(scores, dim=-1).unsqueeze(1)        # (B, 1, T_src)

            # each token has some hidden state, and when we multiply the normalized score of each token with its hidden state vector
            # we are essentially deriving a single "mixed" hidden state that contains the info of all the hidden states but weighted by importance.
            context_t = weights @ encoder_outputs                        # (B, 1, H)

            # Feed [input_token, context] into RNN
            rnn_input = torch.cat([y_t, context_t], dim=-1)
            out, h = self.rnn.forward(rnn_input, h)

            assert out.shape == (B, 1, H)
            assert h.shape == (B, H)

            logits = out @ self.output.T

            assert logits.shape == (B, 1, VOCAB)
            
            logits_all.append(logits)

        return torch.cat(logits_all, dim=1)  # (B, T, VOCAB)

    @property
    def parameters(self):
        return [
            self.embedding,
            *self.rnn.parameters,
            self.attention,
            self.output
        ]
    
    @property
    def parameters_dict(self):
        return dict(
            embedding=self.embedding,
            rnn=self.rnn.parameters_dict,
            attention=self.attention,
            output=self.output
        )

print("The AttentionDecoder above is Bahdanau's 2015 innovation.")
print("Every decoder step computes its OWN context by looking back at ALL encoder states.")
print()
print("This attention mechanism is what the Transformer (2017) generalized:")
print("  - Self-attention: tokens attend to OTHER tokens in the SAME sequence")
print("  - Cross-attention: decoder attends to encoder — SAME as Bahdanau, just cleaner math")


# %%
VOCAB = 13
SOS, EOS, PAD = 10, 11, 12

epochs = 20_000
lr = 0.05
embedding_dim = 20
hidden_dim = 80
batch_size = 20
sequence_len = 20
metrics = []

attention_decoder = AttentionDecoder(
    vocab_size=VOCAB,
    embed_dim=embedding_dim,
    hidden_dim=hidden_dim
)
encoder = NaiveEncoder(VOCAB, embedding_dim, hidden_dim, True)

for epoch in range(epochs):
    # make batch inputs
    batch_src, batch_decoder_inp, batch_decoder_tar = make_batch(batch_size, sequence_len)
    
    # run encoder
    enc_out_full, enc_out_last = encoder.forward(batch_src)
    
    # run attention decoder
    decoder_out = attention_decoder.forward(batch_decoder_inp, enc_out_full, enc_out_last)


    # calculate prob
    batch_probs = torch.softmax(decoder_out, dim=-1)  # (B, T, VOCAB)
    probs_tokenwise = batch_probs.reshape(batch_size * (sequence_len + 1), VOCAB)  # (B*T, VOCAB)
    actual_tokenwise_vals = batch_decoder_tar.reshape(batch_size * (sequence_len + 1)) # (B*T, )

    probs_for_actual_vals = probs_tokenwise[torch.arange(batch_size * (sequence_len + 1)), actual_tokenwise_vals]

    loss = -torch.log(probs_for_actual_vals).mean()

    # Backprop
    for param in [encoder.parameters, attention_decoder.parameters]:
        for p in param:
            p.grad = None

    loss.backward()


    # log
    if epoch % 100 == 0:
        acc = (probs_for_actual_vals > 0.5).float().mean().item()  # very rough accuracy estimate
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}")

    
        # collect metrics
        metrics.append({
            "epoch": epoch,
            "loss": loss.item(),

            # encoder
            "encoder__rnn__W_x__grad_mean": encoder.parameters_dict['rnn']['W_x'].grad.abs().mean().item(),
            "encoder__rnn__W_h__grad_mean": encoder.parameters_dict['rnn']['W_h'].grad.abs().mean().item(),
            "encoder__rnn__W_x__weights_mean": encoder.parameters_dict['rnn']['W_x'].abs().mean().item(),
            "encoder__rnn__W_h__weights_mean": encoder.parameters_dict['rnn']['W_h'].abs().mean().item(),
            

            # decoder
            "decoder__rnn__W_x__grad_mean": attention_decoder.parameters_dict['rnn']['W_x'].grad.abs().mean().item(),
            "decoder__rnn__W_h__grad_mean": attention_decoder.parameters_dict['rnn']['W_h'].grad.abs().mean().item(),
            "decoder__rnn__W_x__weights_mean": attention_decoder.parameters_dict['rnn']['W_x'].abs().mean().item(),
            "decoder__rnn__W_h__weights_mean": attention_decoder.parameters_dict['rnn']['W_h'].abs().mean().item(),

            "decoder__linear__W_x__grad_mean": attention_decoder.parameters_dict['attention'].grad.abs().mean().item(),

            "prob_mean": probs_for_actual_vals.mean().item(),
            "prob_std": probs_for_actual_vals.std().item(),
        })

    # Gradient descent step
    for param in [encoder.parameters, attention_decoder.parameters]:
        for p in param:
            p.data -= lr * p.grad


# %%
from matplotlib import pyplot as plt

fig, ax = plt.subplots(3, 3, figsize=(12, 8))

# plot loss
ax[0, 0].plot([m['epoch'] for m in metrics], [m['loss'] for m in metrics])
ax[0, 0].set_title("Loss")

# plot encoder W_x gradients
ax[0, 1].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_x__grad_mean'] for m in metrics], label='encoder W_x grad')
ax[0, 1].set_title("Encoder W_x Gradients")
ax[0, 1].set_yscale('log')

# plot encoder W_x weights
ax[0, 2].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_x__weights_mean'] for m in metrics], label='encoder W_x weights')
ax[0, 2].set_title("Encoder W_x Weights")
ax[0, 2].set_yscale('log')

# plot encoder W_h gradients
ax[1, 0].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_h__grad_mean'] for m in metrics], label='encoder W_h grad')
ax[1, 0].set_title("Encoder W_h Gradients")
ax[1, 0].set_yscale('log')

# plot encoder W_h weights
ax[1, 1].plot([m['epoch'] for m in metrics], [m['encoder__rnn__W_h__weights_mean'] for m in metrics], label='encoder W_h weights')
ax[1, 1].set_title("Encoder W_h Weights")
ax[1, 1].set_yscale('log')


# plot decoder W_x gradients
ax[1, 2].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_x__grad_mean'] for m in metrics], label='decoder W_x grad')
ax[1, 2].set_title("Decoder W_x Gradients")
ax[1, 2].set_yscale('log')

# plot decoder W_x weights
ax[2, 0].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_x__weights_mean'] for m in metrics], label='decoder W_x weights')
ax[2, 0].set_title("Decoder W_x Weights")
ax[2, 0].set_yscale('log')

# plot decoder W_h gradients
ax[2, 1].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_h__grad_mean'] for m in metrics], label='decoder W_h grad')
ax[2, 1].set_title("Decoder W_h Gradients")
ax[2, 1].set_yscale('log')

# plot decoder W_h weights
ax[2, 2].plot([m['epoch'] for m in metrics], [m['decoder__rnn__W_h__weights_mean'] for m in metrics], label='decoder W_h weights')
ax[2, 2].set_title("Decoder W_h Weights")
ax[2, 2].set_yscale('log')

plt.tight_layout()



def show_eigenvalues(W_h, name="W_h"):
    eigenvalues = torch.linalg.eigvals(W_h.detach())   # complex tensor, shape (hidden_dim,)
    real = eigenvalues.real.numpy()
    imag = eigenvalues.imag.numpy()
    magnitudes = eigenvalues.abs().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scatter on complex plane vs unit circle
    theta = torch.linspace(0, 2*torch.pi, 200)
    axes[0].plot(torch.cos(theta), torch.sin(theta), 'k--', label='unit circle (|λ|=1)')
    axes[0].scatter(real, imag, c=magnitudes, cmap='RdYlGn_r', s=40)
    axes[0].axhline(0, color='gray', lw=0.5)
    axes[0].axvline(0, color='gray', lw=0.5)
    axes[0].set_xlabel('Re(λ)')
    axes[0].set_ylabel('Im(λ)')
    axes[0].set_title(f'{name} eigenvalues — complex plane')
    axes[0].set_aspect('equal')
    axes[0].legend()

    # Right: histogram of magnitudes
    axes[1].hist(magnitudes, bins=20)
    axes[1].axvline(1.0, color='red', linestyle='--', label='|λ|=1 (preserves)')
    axes[1].set_xlabel('|λ|')
    axes[1].set_ylabel('count')
    axes[1].set_title(f'{name} — distribution of |λ|')
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    print(f"max |λ|:  {magnitudes.max():.3f}")
    print(f"min |λ|:  {magnitudes.min():.3f}")
    print(f"mean |λ|: {magnitudes.mean():.3f}")


show_eigenvalues(encoder.parameters_dict['rnn']['W_h'], name="Encoder W_h")
show_eigenvalues(attention_decoder.parameters_dict['rnn']['W_h'], name="Decoder W_h")


# %% 
def greedy_decode(encoder, decoder, src, max_len=20):
    enc_whole_seq_states, enc_last_states = encoder.forward(src)
    batch_size = src.shape[0]
    decoder_input = torch.full((batch_size, 1), SOS, dtype=torch.long)  # start with SOS token
    hidden_state = enc_last_states

    generated_tokens = []

    for _ in range(max_len):
        logits = decoder.forward(decoder_input, enc_whole_seq_states, hidden_state)
        next_token = logits.argmax(dim=-1)[:, -1]  # get the last token's logits and pick the highest one
        generated_tokens.append(next_token)

        if (next_token == EOS).all():
            break

        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(1)], dim=1)  # append the predicted token to the input for the next step

    return torch.stack(generated_tokens, dim=1)


sample_src = torch.randint(0, 10, (2, 20))  # two random sequences of length 5
generated = greedy_decode(encoder, attention_decoder, sample_src)
print("Input:\n", sample_src)
print("Generated output:\n", generated)


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
