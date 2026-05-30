# %% [markdown]
# # 01: Self-Attention From Scratch
# Roadmap: Days 3-4
# Prerequisite: 00_vectors_and_matrices.py
# Watch alongside: Karpathy "Let's build GPT" (transcript in notes/videos/transcripts/)

# %%
# TODO: Build after completing Day 0 (vectors/matrices)
# This file will implement:
# 1. Single-head self-attention from scratch
# 2. Multi-head attention
# 3. Visualize attention weights
# 4. Understand Q, K, V projections

# %% [markdown]
# ## Imports
# %%
import torch
from typing import TypedDict
import matplotlib.pyplot as plt


# %% [markdown]
# ## Build a Self-Attention 

# %%

class WeightMetrics(TypedDict):
    value: torch.Tensor
    grad: torch.Tensor
    update_to_value_ratio: torch.Tensor

class AttentionParams(TypedDict):
    W_key: WeightMetrics
    W_query: WeightMetrics
    W_value: WeightMetrics

class SelfAttention:
    def __init__(
        self,
        input_token_dim: int,
        attention_dim: int,
        value_dim: int
    ):
        self.input_token_dim = input_token_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim

        self.W_key = (torch.randn(attention_dim, input_token_dim) * 1 / input_token_dim ** 0.5).requires_grad_(True)  # tells "hey, my meaning is A"
        self.W_query = (torch.randn(attention_dim, input_token_dim) * 1 / input_token_dim ** 0.5).requires_grad_(True)  # tell "hey, I'm looking for B"
        self.W_value = (torch.randn(value_dim, input_token_dim) * 1 / input_token_dim ** 0.5).requires_grad_(True) # tell "hey, this is what I'll GIVE you if you pick me"

    def forward(self, batched_sequences: torch.Tensor):
        """Forward pass of the module.

        Attributes:
            batched_sequences: Batch of sequences/sentences (B, T, InpDim)
        """
        
        B, T, _ = batched_sequences.shape

        keys: torch.Tensor = batched_sequences @ self.W_key.T
        assert keys.shape == (B, T, self.attention_dim)
        
        queries: torch.Tensor = batched_sequences @ self.W_query.T
        assert queries.shape == (B, T, self.attention_dim)

        # this is the value of the token itself - it didn't take into consideration other tokens YET! 
        values = batched_sequences @ self.W_value.T
        assert values.shape == (B, T, self.value_dim)

        # here, we are essentially doing the dot-product between the key vector and query vector (to know how much they match).
        # hence, we needed, to transpose the query matrix to be (Batch, AttentionDim, Sequence)
        # so that when we matmul with keys (Batch, Sequence, Attention), we kinda do Key vector (Sequence, Attention) "dot_prod" (Sequence, Attention) vector of Query  
        attention_matrix = keys @ queries.transpose(2, 1) / (self.attention_dim ** 0.5)
        assert attention_matrix.shape == (B, T, T)

        # now, we need to make sure that the attention_matrix is normalized
        # we need to normalize across SequenceLength dim, so that we get the importance per Token in a sequence
        normalised_attention_matrix = torch.softmax(attention_matrix, dim=-1) 


        # now attention contains (Batch, Sequence, ImportanceOfEachSequenceToken)
        context_aware_value = normalised_attention_matrix @ values
        # when we multiply attention_matrix with the values matrix, we are kinda infusing the 
        # context into the value_dim (using the weighted values of each token in the sequence) 
        assert context_aware_value.shape == (B, T, self.value_dim)

        return context_aware_value, normalised_attention_matrix


# %% [markdown]
# ## Sanity Check

# %%
sa = SelfAttention(input_token_dim=8, attention_dim=8, value_dim=8)
x = torch.randn(2, 5, 8)
out = sa.forward(x)
print(out[0].shape)   # should be (2, 5, 8)

# %% [markdown]
# ## Visualise Attention

# %%
# sample vocab/tokenizer

vocab = ["the", "cat", "sat", "on", "mat", "<pad>"]

# sample input
tokens = ["the", "cat", "sat", "on", "mat"]

vocab_size = len(vocab)
input_token_dim = 8
embedding = torch.randn(vocab_size, input_token_dim) * 0.1

token_ids = [vocab.index(t) for t in tokens]   # [0, 1, 2, 3, 4]
token_ids_tensor = torch.tensor(token_ids).unsqueeze(0)
embedded = embedding[token_ids_tensor]

attention_head = SelfAttention(input_token_dim, 10, 10)

out, attn = attention_head.forward(embedded)
attn[0]   # (T, T)

plt.imshow(attn[0].detach())
plt.xticks(range(len(tokens)), tokens)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("attended TO")
plt.ylabel("attended FROM")
plt.colorbar()
plt.title("Attention weights (random init — meaningless!)")
plt.show()



# %% [markdown]
# ## My thoughts on what's been implemented so far:

# Till this point, we have implemented a single-head attention. 
# This means that each token in the sequence is trying to learn a single "hey, this is what I am looking for" (query), "this is what I am" (key), and "this is what I will give you if you pick me" (value) vector.
# This is a very limited way of looking at the world, because in reality, each token
# can have multiple "facets" or "aspects" to it. For example, the word "bank" can refer to a financial institution or the side of a river.
# In a single-head attention, we are forcing the model to learn a single representation for "bank", which is not ideal. 
# In multi-head attention, we allow the model to learn multiple representations for each token, which is more powerful and allows the model to capture more complex relationships between tokens.


# %% [markdown]
# ## Multi-Head Attention
# %%
class MultiHeadAttention:
    def __init__(
            self,
            input_vocab_dim: int,
            attention_dim: int,
            value_dim: int,
            num_heads: int,
            enable_causal_mask: bool = True 
        ):
        """Multi-head (many Q, K, V) attention module.

        Attributes:
            input_vocab_dim: The dimensionality of the input token (embedding dim)
            attention_dim: Per-HEAD attention dimensionality (unlike the traditional way of providing combined-head attention dim)
            value_dim: Per-HEAD "value" dimensionality.
            num_heads: The number of heads.
            enable_casual_mask: Whether to mask out the future-tokens. 
                The token in position `t`, must not attend to token at position `t + 1`.  
        """

        self.input_vocab_dim = input_vocab_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.enable_causal_mask = enable_causal_mask

        self.queries: torch.Tensor = (torch.randn(num_heads, attention_dim, input_vocab_dim) * 1 / input_vocab_dim ** 0.5).requires_grad_(True)
        self.keys: torch.Tensor = (torch.randn(num_heads, attention_dim, input_vocab_dim) * 1 / input_vocab_dim ** 0.5).requires_grad_(True)
        self.values: torch.Tensor = (torch.randn(num_heads, value_dim, input_vocab_dim) * 1 / input_vocab_dim ** 0.5).requires_grad_(True)
        self.head_mixer: torch.Tensor = (torch.randn(input_vocab_dim, num_heads * value_dim) * 1 / (num_heads * value_dim) ** 0.5).requires_grad_(True)


    def forward(self, sequence: torch.Tensor) ->tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = sequence.shape
        H = self.num_heads
        V = self.value_dim
        A = self.attention_dim

        # (B, T, InputVocabDim) @ (B, H, AttentionDim, InputVocabDim) -> won't work
        # hence, we need sth like (B, 1, T, InputVocabDim) @ (B, H, InputVocabDim, AttentionDim) -> (B, H, T, AttentionDim)
        queries = sequence.unsqueeze(1) @ self.queries.transpose(2, 1)
        assert queries.shape == (B, H, T, A)

        keys = sequence.unsqueeze(1) @ self.keys.transpose(2, 1)
        assert keys.shape == (B, H, T, A)

        # (B, 1, T, InputVocabDim) @ (B, H, InputVocabDim, ValueDim) -> (B, H, T, ValueDim)
        values = sequence.unsqueeze(1) @ self.values.transpose(2, 1)
        assert values.shape == (B, H, T, V)

        # keys = (B, H, T, Attention)  @ (B, H, Attention, T) -> (B, H, T, T)
        attention_matrix = (queries @ keys.transpose(-1, -2)) * (1 / A ** 0.5)
        assert attention_matrix.shape == (B, H, T, T)
        
        if self.enable_causal_mask:
            attention_matrix +=  torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1)

        normalized_attentions = torch.softmax(attention_matrix, dim=-1)
        assert normalized_attentions.shape == (B, H, T, T)

        # (B, H, T, T) @ (B, H, T, V) -> (B, H, T, V)
        context_aware_values = normalized_attentions @ values

        # We need sth like (B, T, H * V) to make sure that we provide the "representation" of a single
        # token from all the heads. But just concatting them is not gonna be optimal, so we need some kinda weighting
        # hence, we have self.head_mixer of shape (InputTokenDim, H * V)

        # reshape the context_aware_values from (B, H, T, V) to (B, T, H * V), then multiply with self.head_mixer
        # (B, T, H * V) @ (H*V, InputTokenDim) -> (B, T, InputTokenDim) 
        out = context_aware_values.transpose(-3, -2).contiguous().reshape(B, T, H * V) @ self.head_mixer.T
        assert out.shape == (B, T, self.input_vocab_dim)

        # remember the last attention map so we can visualize it later (a "hook")
        self.last_attention = normalized_attentions   # (B, H, T, T)

        return out, normalized_attentions

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.queries, self.keys, self.values, self.head_mixer]
    

# %% [markdown]
#### Test it out


# %%
ma = MultiHeadAttention(
    input_vocab_dim=8,
    attention_dim=8,
    value_dim=8,
    num_heads=3
)
x = torch.randn(2, 5, 8)
out, attention_maps = ma.forward(x)
print(out.shape)   # should be (2, 5, 8)
print(attention_maps.shape) # should be (2, 3, 5, 5)
plt.imshow(attention_maps[0, 0].detach())





# %% [markdown]
# ## Summary of what I've learned from the MultiHeadAttention.
# Basically, the MHA is allowing each token to do attention mechanism (to attend each other)
# based on different leaned aspects. The SingleHeadAttention could also learn multiple-aspects,
# but a single vector could not carry much info of different aspects, right ? 
# Hence, we have MHA which does the SHA per token, and then merges those values from SHA using 
# some kinda leaned weighted summation.


# %% [markdown]
# ## Feed Forward Network


# %%
class FeedForwardNetwork:
    def __init__(
        self,
        input_token_dim: int,
        expansion: int = 4
    ):
        self.input_token_dim = input_token_dim
        self.expansion = expansion

        # Expand the input token's representation
        self.W_x1 = (torch.randn(input_token_dim * expansion, input_token_dim) * 1 / input_token_dim ** 0.5).requires_grad_(True)
        self.W_b1 = torch.zeros(input_token_dim * expansion, ).requires_grad_(True)

        # Shrink to the token's internal representation to the original value 
        self.W_x2 = (torch.randn(input_token_dim, input_token_dim * expansion) * 1 / (expansion * input_token_dim) ** 0.5).requires_grad_(True)
        self.W_b2 = torch.zeros(input_token_dim, ).requires_grad_(True)

        self.activation = torch.nn.GELU()


    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        B, T, C = sequence.shape
        assert C == self.input_token_dim
        H = self.input_token_dim * self.expansion

        # (B, T, C) @ (C, H) -> (B, T, H)
        expanded = sequence @ self.W_x1.T + self.W_b1
        assert expanded.shape == (B, T, H)

        # apply GELU non-linearity
        activated = self.activation(expanded)
        assert activated.shape == (B, T, H)
        
        # (B, T, H) * (H, C) -> (B, T, C)
        restored = activated @ self.W_x2.T + self.W_b2
        assert restored.shape == (B, T, C)

        return restored

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.W_x1, self.W_b1, self.W_x2, self.W_b2]


# %% [markdown]
# ### Test it out

# %%

ffn = FeedForwardNetwork(8, 4)
random = torch.randn(2, 4, 8)

result = ffn.forward(random)
print(result.shape) # must be (2, 4, 8)

# %% [markdown]
# ### LayerNorm

# Each token embedding starts as something like [2, 5, 3, 1, 6]
# 
# As it passes through layers, multiplications cause MAGNITUDE DRIFT:
#   - Token might end up [100, 400, 300, 100, 600]  → explosion
#   - Or [0.001, 0.03, 0.003, 0.001, 0.006]         → vanishing
# 
# Either way, gradients become unstable through deep networks.
#  
# LayerNorm fixes this by normalizing EACH TOKEN'S VECTOR independently
# (across its channels) to mean=0, std=1:
#   [100, 400, 300, 100, 600]  →  [-1.05, 0.53, 0.0, -1.05, 1.58]
# 
# Each token gets reset to a clean magnitude — independent of other tokens
# in the batch or the sequence.
#  
# Then `gamma` (per-channel scale) and `beta` (per-channel shift) let the
# network LEARN the magnitudes it actually wants — restoring expressiveness
# while keeping training stable.


# %% 
class LayerNorm:
    def __init__(
        self,
        input_vocab_dim: int,
    ):
        self.input_vocab_dim = input_vocab_dim

        self.gamma = torch.ones(input_vocab_dim, requires_grad=True)
        self.beta = torch.zeros(input_vocab_dim, requires_grad=True)
    

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        B, T, C = sequence.shape
        assert C == self.input_vocab_dim

        standardised = (sequence - sequence.mean(dim=-1, keepdim=True)) / (sequence.std(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        
        # (B, T, C) * (C,) -> "(C, )" gets broadcasted to (1, 1, C) and then elementwise multiplied -> (B, T, C)
        normalized = standardised * self.gamma + self.beta
        assert normalized.shape == (B, T, C)

        return normalized

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]
    

# %% [markdown]
# ### Test it out


# %%

ln = LayerNorm(8)
rand_inp = torch.randn(1, 4, 8) * 10

normalized = ln.forward(rand_inp)


print(f"Original value of first token: {rand_inp[0][0]}")
print(f"Normalized value of the first token: {normalized[0][0]}")
print(f"Stats:\n Mean: {normalized[0][0].mean(dim=-1)}\t Std: {normalized[0][0].std(dim=-1)}")

# %% [markdown]
# ## Transformer Block

class TransformerBlock:
    def __init__(
        self,
        input_vocab_dim: int,
        attention_dim: int,
        value_dim: int,
        num_heads: int,
        expansion: int = 4
    ):
        self.input_vocab_dim = input_vocab_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.expansion = expansion

        # for input normalization
        self.ln1 = LayerNorm(input_vocab_dim=input_vocab_dim)

        # for each token to attend to each other from different perspectives
        self.mh_attention = MultiHeadAttention(
            input_vocab_dim=input_vocab_dim,
            attention_dim=attention_dim,
            value_dim=value_dim,
            num_heads=num_heads
        )

        # store the info each token carries 
        self.ffn = FeedForwardNetwork(
            input_token_dim=input_vocab_dim,
            expansion=expansion,
        )

        # normalize the outputs of the ffn
        self.ln2 = LayerNorm(input_vocab_dim=input_vocab_dim)



    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        B, T, C = sequence.shape

        normalized_sequences = self.ln1.forward(sequence)
        assert normalized_sequences.shape == (B, T, C)

        context_aware_tokens_sequences, _ = self.mh_attention.forward(normalized_sequences)
        assert context_aware_tokens_sequences.shape == (B, T, C)

        sequence = sequence + context_aware_tokens_sequences

        normalized_after_mha = self.ln2.forward(sequence)
        assert normalized_after_mha.shape == (B, T, C)

        sequence = sequence + self.ffn.forward(normalized_after_mha)
        assert sequence.shape == (B, T, C)

        return sequence

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [
            *self.ln1.parameters,
            *self.mh_attention.parameters,
            *self.ffn.parameters,
            *self.ln2.parameters,
        ]


# %% [markdown]
# ### Test the TransformerBlock

# %%
block = TransformerBlock(input_vocab_dim=8, attention_dim=8, value_dim=8, num_heads=4)
x = torch.randn(2, 5, 8)
before = x.clone()
out = block.forward(x)
print("out shape:", out.shape)            # (2, 5, 8)
print("input unchanged:", torch.equal(x, before))  # True — no in-place mutation


# %% [markdown]
# ## Data + Tokenizer (char-level)
#
# nanoGPT needs text turned into integer token ids. We use the simplest possible
# tokenizer: one integer per CHARACTER. No BPE, no subwords. vocab ~= 65.
#
# Pipeline:
#   raw text  ->  list of char ids  ->  tensor  ->  sample windows for training

# %%
from pathlib import Path

text = Path("/Users/komronvalijonov/work/personal/thesis/exploration/data/tinyshakespeare.txt").read_text()
print(f"dataset length (chars): {len(text):,}")
print("first 200 chars:\n", text[:200])

# Build the vocabulary: every unique character in the corpus
chars = sorted(set(text))
vocab_size = len(chars)
print(f"\nvocab_size: {vocab_size}")
print("vocab:", "".join(chars))

# The tokenizer IS just two lookup dicts (char<->int)
stoi = {ch: i for i, ch in enumerate(chars)}   # string-to-int
itos = {i: ch for i, ch in enumerate(chars)}   # int-to-string

def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]

def decode(ids: list[int]) -> str:
    return "".join(itos[i] for i in ids)

# Sanity: round-trip
print("\nencode('hello'):", encode("hello"))
print("decode(encode('hello')):", decode(encode("hello")))


# %% [markdown]
# ### Encode the whole dataset + train/val split

# %%
data = torch.tensor(encode(text), dtype=torch.long)   # (N,) — every char as an int id
print("data shape:", data.shape, "dtype:", data.dtype)

# 90/10 split — val set lets us check we're not just memorizing
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"train: {len(train_data):,}  val: {len(val_data):,}")


# %% [markdown]
# ### Batching — sample random windows
#
# A transformer trains on fixed-length windows (block_size tokens).
# For each window, the TARGET is the same window shifted right by one:
#   x = "To be or not to b"
#   y = "o be or not to be"   (each position predicts the NEXT char)
#
# This is the next-token-prediction task — every position is a training example.

# %%
block_size = 32   # context length: how many chars the model sees at once
batch_size = 16   # how many independent windows per batch

def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = train_data if split == "train" else val_data
    # pick batch_size random start positions
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i : i + block_size] for i in ix])           # (B, T)
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])   # (B, T) shifted +1
    return x, y

xb, yb = get_batch("train")
print("xb shape:", xb.shape, "yb shape:", yb.shape)   # both (16, 32)
print("\nfirst window x:", decode(xb[0].tolist()))
print("first window y:", decode(yb[0].tolist()))      # same text, shifted by 1

# %% [markdown]
# ## Let's build NanoGPT!

# %%
class NanoGPT:
    def __init__(
        self,
        vocab_size: int,
        input_vocab_dim: int,
        attention_dim: int,
        value_dim: int,
        num_heads: int,
        num_blocks: int,
        ffn_expansion: int = 4,
        block_size: int = 32,
    ):
        self.block_size = block_size  # this is the "context size"
        self.vocab_size = vocab_size
        self.input_vocab_dim = input_vocab_dim  # this is "d_model"
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_expansion = ffn_expansion


        # token embedding: (vocab_size, E) — lookup table, row k = vector for token id k
        self.token_embedding = (torch.randn(vocab_size, input_vocab_dim) * 0.1).requires_grad_(True)
        # position embedding: (block_size, E) — lookup table, row p = vector for position p
        self.positional_embedding = (torch.randn(block_size, input_vocab_dim) * 0.1).requires_grad_(True)


        self.transformer_blocks = [TransformerBlock(
            input_vocab_dim=input_vocab_dim,
            attention_dim=attention_dim,
            value_dim=value_dim,
            expansion=ffn_expansion,
            num_heads=num_heads
        ) for _ in range(num_blocks)]

        self.ln = LayerNorm(input_vocab_dim=input_vocab_dim)
        self.lm_head = (torch.randn(vocab_size, input_vocab_dim) * 1 / input_vocab_dim ** 0.5).requires_grad_(True)
    
    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # idx: (B, T) integer token ids — straight from get_batch
        B, T = idx.shape
        E = self.input_vocab_dim

        assert T <= self.block_size   # can't exceed context (no pos emb beyond block_size)

        # token lookup: each id -> its E-dim vector
        tokens_embedded = self.token_embedding[idx]            # (B, T, E)
        assert tokens_embedded.shape == (B, T, E)

        # position lookup: positions 0..T-1, INDEPENDENT of token content
        positions_embedded = self.positional_embedding[:T]     # (T, E)
        assert positions_embedded.shape == (T, E)

        # add — (T, E) broadcasts across the batch dim
        position_aware_emb = tokens_embedded + positions_embedded   # (B, T, E)
        assert position_aware_emb.shape == (B, T, E)

        out = position_aware_emb
        for tb in self.transformer_blocks:
            out = tb.forward(out)

        assert out.shape == (B, T, E)

        normalized = self.ln.forward(out)
        assert normalized.shape == (B, T, E)

        logits = normalized @ self.lm_head.T
        assert logits.shape == (B, T, self.vocab_size)

        return logits

    @property
    def parameters(self) -> list[torch.Tensor]:
        # own tensors + every block's parameters (which recurse into attn/ffn/ln)
        params = [
            self.token_embedding,
            self.positional_embedding,
            self.lm_head,
            *self.ln.parameters,
        ]
        for block in self.transformer_blocks:
            params += block.parameters
        return params

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressive sampling. idx: (B, T) starting context."""
        for _ in range(max_new_tokens):
            # take the last "block_size" number of chars/tokens for each batch
            idx_cond = idx[:, -self.block_size:]      # crop to context window

            # run the model to get the logit for the next tokens
            logits = self.forward(idx_cond)            # (B, T, vocab)

            # get the logit for the latest T (that's the last word/token in the block/context-window)
            logits = logits[:, -1, :]                  # last position -> (B, vocab)
            
            # apply softmax to normilize it - probability disstribution 
            probs = torch.softmax(logits, dim=-1)

            # get one from the distribution
            next_id = torch.multinomial(probs, num_samples=1)   # sample (B, 1)
            idx = torch.cat([idx, next_id], dim=1)     # append
        return idx


# %% [markdown]
# ## Train NanoGPT on tinyshakespeare

# %%
import torch.nn.functional as F

# hyperparameters
d_model = 64  # token embedding dimension ("internal representation" of each token)
n_heads = 4  # how many different "aspects" each token attends with
n_blocks = 3  # how many times the tokens can "re-attend" to each other (and re-process through FFN)
learning_rate = 1e-3  # starting learning rate == 0.001
num_steps = 5000  # epochs
eval_interval = 500  # in how many steps to run the validation evaluation

torch.manual_seed(1337)
model = NanoGPT(
    vocab_size=vocab_size,
    input_vocab_dim=d_model,
    attention_dim=d_model,
    value_dim=d_model,
    num_heads=n_heads,
    num_blocks=n_blocks,
    block_size=block_size,
)

params = model.parameters
print(f"parameter tensors: {len(params)}")
print(f"scalar params: {sum(p.numel() for p in params):,}")

# --- Adam state (plain SGD trains transformers VERY slowly; Adam is near-essential) ---
# We implement Adam by hand to stay in the bare-tensor spirit.
m = [torch.zeros_like(p) for p in params]   # 1st moment (mean of grads)
v = [torch.zeros_like(p) for p in params]   # 2nd moment (mean of grad^2)
beta1, beta2, eps = 0.9, 0.999, 1e-8


@torch.no_grad()
def estimate_loss(split: str, iters: int = 20) -> float:
    losses = []
    for _ in range(iters):
        xb, yb = get_batch(split)

        # (B, T, C)
        logits = model.forward(xb)

        # logits.reshape(-1, vocab_size) == (B, T*C)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)

# %%

for step in range(1, num_steps + 1):
    xb, yb = get_batch("train")
    logits = model.forward(xb)                                        # (B, T, vocab)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), yb.reshape(-1))

    loss.backward()

    # manual Adam update
    with torch.no_grad():
        for i, p in enumerate(params):
            g = p.grad
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * g * g
            m_hat = m[i] / (1 - beta1 ** step)
            v_hat = v[i] / (1 - beta2 ** step)
            p -= learning_rate * m_hat / (v_hat.sqrt() + eps)
            p.grad = None

    if step % eval_interval == 0 or step == 1:
        tr = estimate_loss("train")
        va = estimate_loss("val")
        print(f"step {step:>5}: train {tr:.4f}  val {va:.4f}")

# %% [markdown]
# ## Generate text from the trained model

# %%
context = torch.zeros((1, 1), dtype=torch.long)   # start from token 0
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))


# %% [markdown]
# ## Visualize the LEARNED attention patterns
#
# Now that the model is trained, the attention maps are MEANINGFUL (not random).
# We feed a sample sequence, run a forward pass (which stores each block's
# attention via the `last_attention` hook), then plot every block x head.
#
# Look for:
#   - LOWER-TRIANGULAR structure (causal mask — can't attend to the future)
#   - bright spots showing which past chars each char attends to

# %%
sample = "To be, or not to be, that is the"
sample_ids = torch.tensor([encode(sample)])        # (1, T)
print(f"sample ({sample_ids.shape[1]} chars): {sample!r}")

# forward pass — this populates each block's `last_attention`
_ = model.forward(sample_ids)

chars_list = list(sample)
n_blocks = len(model.transformer_blocks)
n_heads = model.transformer_blocks[0].mh_attention.num_heads

fig, axes = plt.subplots(n_blocks, n_heads, figsize=(6 * n_heads, 6 * n_blocks))
if n_blocks == 1:
    axes = axes.reshape(1, -1)

for b, block in enumerate(model.transformer_blocks):
    attn = block.mh_attention.last_attention[0].detach()   # (H, T, T) — drop batch
    for h in range(n_heads):
        ax = axes[b, h]
        ax.imshow(attn[h], cmap="viridis")
        ax.set_title(f"block {b}, head {h}", fontsize=9)
        ax.set_xticks(range(len(chars_list)))
        ax.set_yticks(range(len(chars_list)))
        ax.set_xticklabels(chars_list, fontsize=6)
        ax.set_yticklabels(chars_list, fontsize=6)

fig.suptitle("Learned attention: row = query char, col = attended-to char", fontsize=12)
plt.tight_layout()
plt.show()
# %%
