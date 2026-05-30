# %% [markdown]
# # 05 — Mini LLaVA from scratch
#
# **Goal:** wire your ViT (file 03) + your nanoGPT (file 01) together with a tiny
# Projector, train end-to-end on CIFAR captions, watch it generate text.
#
# ## What LLaVA actually is, in one diagram
#
# ```
# image  →  vision encoder  →  patch tokens  →  PROJECTOR  →  "image tokens" in LLM space
#                                                    ↓
# text  →  tokenizer + embed →  text tokens   →  [concat with image tokens]
#                                                    ↓
#                                               causal LLM  →  next-token logits
#                                                    ↓
#                                          cross-entropy on the text-token positions
# ```
#
# **The whole trick:** project image patches into the LLM's token space, prepend them
# to text, run the LLM as usual. The LLM doesn't know image tokens aren't real words —
# they're just vectors in the embedding space, and the LLM was trained to predict
# the next vector from a sequence.
#
# ## What we're going to build
#
# - **ViT** (copied from file 03) — turns CIFAR images into (B, 65, 64) patch+CLS tokens.
# - **nanoGPT** (copied from file 01, slightly trimmed) — turns text into (B, T, 64) tokens,
#   does causal self-attention, emits logits.
# - **Projector** (NEW, 1 line) — `Linear(vit_dim, gpt_dim)` translates image tokens into LLM space.
# - **MiniLLaVA** (NEW) — glues all three, handles concat + loss-masking.
# - **Char tokenizer** for captions like `"a photo of a cat"`.
# - **Training loop** — manual Adam, ~2000 steps.
# - **generate()** — autoregressive sampling: image in, caption out.
#
# Everything here is < 600 lines, all from-scratch (no `nn.Module`, just `requires_grad_`).

# %% [markdown]
# ### Imports

# %%
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision


# %% [markdown]
# ## Building blocks (copied from files 01 and 03)
#
# These are exactly what you wrote in `01_attention_from_scratch.py` and
# `03_vision_transformer.py`. No new concepts here — skip-read if you trust them.

# %%
class MultiHeadAttention:
    def __init__(self, input_vocab_dim, attention_dim, value_dim, num_heads, enable_causal_mask=True):
        self.input_vocab_dim = input_vocab_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.enable_causal_mask = enable_causal_mask

        s = 1 / input_vocab_dim ** 0.5
        self.queries = (torch.randn(num_heads, attention_dim, input_vocab_dim) * s).requires_grad_(True)
        self.keys    = (torch.randn(num_heads, attention_dim, input_vocab_dim) * s).requires_grad_(True)
        self.values  = (torch.randn(num_heads, value_dim, input_vocab_dim) * s).requires_grad_(True)
        self.head_mixer = (torch.randn(input_vocab_dim, num_heads * value_dim)
                           * (1 / (num_heads * value_dim) ** 0.5)).requires_grad_(True)

    def forward(self, sequence):
        B, T, _ = sequence.shape
        H, A, V = self.num_heads, self.attention_dim, self.value_dim

        q = sequence.unsqueeze(1) @ self.queries.transpose(2, 1)   # (B, H, T, A)
        k = sequence.unsqueeze(1) @ self.keys.transpose(2, 1)      # (B, H, T, A)
        v = sequence.unsqueeze(1) @ self.values.transpose(2, 1)    # (B, H, T, V)

        attn = (q @ k.transpose(-1, -2)) * (1 / A ** 0.5)          # (B, H, T, T)
        if self.enable_causal_mask:
            attn = attn + torch.triu(torch.ones(T, T) * float("-inf"), diagonal=1)
        attn = torch.softmax(attn, dim=-1)
        ctx = attn @ v                                               # (B, H, T, V)

        out = ctx.transpose(-3, -2).contiguous().reshape(B, T, H * V) @ self.head_mixer.T
        return out

    @property
    def parameters(self):
        return [self.queries, self.keys, self.values, self.head_mixer]


class FeedForwardNetwork:
    def __init__(self, input_token_dim, expansion=4):
        self.input_token_dim = input_token_dim
        self.expansion = expansion
        self.W_x1 = (torch.randn(input_token_dim * expansion, input_token_dim)
                     * (1 / input_token_dim ** 0.5)).requires_grad_(True)
        self.W_b1 = torch.zeros(input_token_dim * expansion).requires_grad_(True)
        self.W_x2 = (torch.randn(input_token_dim, input_token_dim * expansion)
                     * (1 / (expansion * input_token_dim) ** 0.5)).requires_grad_(True)
        self.W_b2 = torch.zeros(input_token_dim).requires_grad_(True)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        return self.activation(x @ self.W_x1.T + self.W_b1) @ self.W_x2.T + self.W_b2

    @property
    def parameters(self):
        return [self.W_x1, self.W_b1, self.W_x2, self.W_b2]


class LayerNorm:
    def __init__(self, input_vocab_dim):
        self.gamma = torch.ones(input_vocab_dim, requires_grad=True)
        self.beta = torch.zeros(input_vocab_dim, requires_grad=True)

    def forward(self, x):
        m = x.mean(dim=-1, keepdim=True)
        s = x.std(dim=-1, keepdim=True, unbiased=False)
        return (x - m) / (s + 1e-5) * self.gamma + self.beta

    @property
    def parameters(self):
        return [self.gamma, self.beta]


class TransformerBlock:
    def __init__(self, dim, attention_dim, value_dim, num_heads, expansion=4, causal_mask=True):
        self.ln1 = LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, attention_dim, value_dim, num_heads, enable_causal_mask=causal_mask)
        self.ln2 = LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim, expansion=expansion)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.ffn.forward(self.ln2.forward(x))
        return x

    @property
    def parameters(self):
        return [*self.ln1.parameters, *self.attn.parameters, *self.ln2.parameters, *self.ffn.parameters]


class PatchEmbedding:
    def __init__(self, channels=3, img_h=32, img_w=32, patch_h=4, patch_w=4, embed_dim=64):
        assert img_h % patch_h == 0 and img_w % patch_w == 0
        self.channels = channels
        self.img_h, self.img_w = img_h, img_w
        self.patch_h, self.patch_w = patch_h, patch_w
        self.embed_dim = embed_dim
        self.num_patches = (img_h // patch_h) * (img_w // patch_w)
        self.patch_dim = channels * patch_h * patch_w
        self.embedding = (torch.randn(embed_dim, self.patch_dim) * (1 / self.patch_dim ** 0.5)).requires_grad_(True)

    def forward(self, imgs):
        B = imgs.shape[0]
        # unfold-based patchify (same math as your double-loop, just vectorized)
        patches = (imgs
                   .unfold(2, self.patch_h, self.patch_h)
                   .unfold(3, self.patch_w, self.patch_w)         # (B, C, Hp, Wp, ph, pw)
                   .permute(0, 2, 3, 1, 4, 5)
                   .contiguous()
                   .reshape(B, self.num_patches, self.patch_dim))  # (B, N, C*ph*pw)
        return patches @ self.embedding.T                          # (B, N, embed_dim)

    @property
    def parameters(self):
        return [self.embedding]


class MiniViT:
    """Image → (B, num_patches+1, embed_dim).  Returns CLS + patches as token sequence."""
    def __init__(self, embed_dim=64, num_blocks=3, num_heads=4, attention_dim=16, value_dim=16):
        self.embed_dim = embed_dim
        self.patcher = PatchEmbedding(embed_dim=embed_dim)
        self.cls = (torch.randn(1, embed_dim) * (1 / embed_dim ** 0.5)).requires_grad_(True)
        self.pos = (torch.randn(self.patcher.num_patches + 1, embed_dim) * (1 / embed_dim ** 0.5)).requires_grad_(True)
        self.blocks = [TransformerBlock(embed_dim, attention_dim, value_dim, num_heads, causal_mask=False)
                       for _ in range(num_blocks)]
        self.ln = LayerNorm(embed_dim)

    def forward(self, imgs):
        B = imgs.shape[0]
        patches = self.patcher.forward(imgs)                           # (B, N, E)
        seq = torch.cat([self.cls.expand(B, -1, -1), patches], dim=1)  # (B, N+1, E)
        seq = seq + self.pos
        for blk in self.blocks:
            seq = blk.forward(seq)
        return self.ln.forward(seq)                                    # (B, N+1, E)

    @property
    def parameters(self):
        params = [self.cls, self.pos, *self.patcher.parameters, *self.ln.parameters]
        for b in self.blocks:
            params += b.parameters
        return params


class MiniGPT:
    """Token ids → next-token logits.  Same as your nanoGPT, exposed in pieces so MiniLLaVA can use them."""
    def __init__(self, vocab_size, embed_dim=64, num_blocks=3, num_heads=4,
                 attention_dim=16, value_dim=16, block_size=64):
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.token_emb = (torch.randn(vocab_size, embed_dim) * 0.1).requires_grad_(True)
        self.pos_emb = (torch.randn(block_size, embed_dim) * 0.1).requires_grad_(True)
        self.blocks = [TransformerBlock(embed_dim, attention_dim, value_dim, num_heads, causal_mask=True)
                       for _ in range(num_blocks)]
        self.ln = LayerNorm(embed_dim)
        self.lm_head = (torch.randn(vocab_size, embed_dim) * (1 / embed_dim ** 0.5)).requires_grad_(True)

    def embed_tokens(self, ids):
        """ids (B, T) → (B, T, embed_dim).  Token + position embedding lookup."""
        T = ids.shape[1]
        return self.token_emb[ids] + self.pos_emb[:T]

    def run_blocks(self, seq):
        """Apply causal transformer blocks + final LN."""
        for blk in self.blocks:
            seq = blk.forward(seq)
        return self.ln.forward(seq)

    def lm_logits(self, seq):
        """(B, T, embed_dim) → (B, T, vocab_size).  The final lm_head projection."""
        return seq @ self.lm_head.T

    @property
    def parameters(self):
        params = [self.token_emb, self.pos_emb, self.lm_head, *self.ln.parameters]
        for b in self.blocks:
            params += b.parameters
        return params


# %% [markdown]
# ## NEW Part 1 — The Projector (YOU build this)
#
# This is the **entire new architectural idea** of LLaVA. Tiny module.
# Maps vision-encoder output dim → LLM token dim.
#
# In real LLaVA-1.5 it's a **2-layer MLP** (Linear → GELU → Linear). Start there.
# Original LLaVA was just a single Linear — too shallow, gets dropped quickly.
#
# ### What you're building
#
# ```
# input:   vit_tokens   shape (B, N_img_tokens, vit_dim)
# output:  proj_tokens  shape (B, N_img_tokens, gpt_dim)   ← same N, new dim
# ```
#
# ### Hints
# - Same pattern as your FFN class — two weight matrices, two biases, an activation in the middle.
# - Use `torch.nn.GELU()` for the activation (same as LLaVA-1.5).
# - Init scale: same as FFN — `1 / sqrt(fan_in)` for each weight.
# - Don't forget a `.parameters` property so MiniLLaVA can collect grads.
#
# ### Ask yourself
# - What's the shape of W1 if the input dim is `vit_dim` and the hidden dim is `hidden_dim`?
# - For matmul `x @ W.T`, where does `vit_dim` need to be in W's shape?

# %%
class Projector:
    """vit_dim → gpt_dim translator. 2-layer MLP with GELU (LLaVA-1.5 style)."""
    def __init__(self, vit_dim, gpt_dim, hidden_dim=None):
        hidden_dim = hidden_dim or gpt_dim
        self.hidden_dim = hidden_dim
        self.vit_dim = vit_dim
        self.llm_dim = gpt_dim
        
        
        # components
        self.activation = torch.nn.GELU()
        self.hidden = (torch.randn(self.hidden_dim, self.vit_dim) * 1 / self.vit_dim ** 0.5).requires_grad_(True)
        self.hidden_b = torch.zeros(self.hidden_dim,).requires_grad_(True)

        self.llm_projector = (torch.randn(self.llm_dim, self.hidden_dim) * 1 / self.hidden_dim ** 0.5).requires_grad_(True) 
        self.llm_projector_b = torch.zeros(self.llm_dim,).requires_grad_(True)


    def forward(self, vit_tokens: torch.Tensor) -> torch.Tensor:
        B, NP, E = vit_tokens.shape

        hidden = vit_tokens @ self.hidden.T + self.hidden_b
        assert hidden.shape == (B, NP, self.hidden_dim)

        activated = self.activation(hidden)

        projected = activated @ self.llm_projector.T + self.llm_projector_b
        assert projected.shape == (B, NP, self.llm_dim)

        return projected

    @property
    def parameters(self):
        return [
            self.hidden,
            self.hidden_b,
            self.llm_projector,
            self.llm_projector_b
        ]


# %% [markdown]
# ## NEW Part 2 — MiniLLaVA: the glue (YOU build this)
#
# Tie the three pieces together. The forward pass is the **actual BOOM of LLaVA**:
# image tokens get prepended to text tokens, the LLM runs once over the full
# concatenated sequence.
#
# ### Six-step forward (in order)
#
# ```
# 1. images          →  vit.forward             →  (B, num_img_tokens, vit_dim)
# 2. vit_out         →  projector.forward        →  (B, num_img_tokens, gpt_dim)
# 3. text_ids        →  gpt.embed_tokens         →  (B, T, gpt_dim)
# 4. concat image + text along dim=1            →  (B, N_img + T, gpt_dim)
# 5. seq             →  gpt.run_blocks           →  (B, N_img + T, gpt_dim)
# 6. seq             →  gpt.lm_logits            →  (B, N_img + T, vocab_size)
# ```
#
# ### Hints
# - `torch.cat([a, b], dim=1)` is the operation that "prepends image to text."
#   `a` and `b` must agree on EVERY dim except dim=1.
# - `parameters` should expose ViT + Projector + GPT params all combined (use `*expansion`).
# - `num_img_tokens` is just a helper — useful in the training loop for slicing.
#
# ### Ask yourself
# - Does ViT's output already include the CLS token? If yes, what's `num_img_tokens` for a 32x32 image with 4x4 patches?
# - Do you add GPT's positional embedding to the image tokens? (Hint: where does the ViT's own positional info come from?)

# %%
class MiniLLaVA:
    def __init__(self, vit: MiniViT, gpt: MiniGPT, projector: Projector):
        self.vit = vit
        self.gpt = gpt
        self.proj = projector

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        """
        images:   (B, 3, 32, 32)
        text_ids: (B, T) integer token ids — the CAPTION

        Returns: logits (B, num_img_tokens + T, vocab_size)
        """
        vit_tokens = self.vit.forward(images)
        projected_tokens = self.proj.forward(vit_tokens)

        text_tokens = self.gpt.embed_tokens(text_ids)

        assert text_tokens.shape[0] == projected_tokens.shape[0], "Batches must be of equal size"
        assert text_tokens.shape[-1] == projected_tokens.shape[-1], "Image and text tokens must be of same dimensionality"

        llm_input_tokens = torch.concat((projected_tokens, text_tokens), dim=1)

        # shape must become (B, NP + T, LLM_DIM)
        assert llm_input_tokens.shape == (text_tokens.shape[0], projected_tokens.shape[1] + text_tokens.shape[1], text_tokens.shape[-1])

        # make each token to talk to each other
        attention_aware_tokens_seq = self.gpt.run_blocks(llm_input_tokens)
        
        # get the logits -> from GPT's internal embedding space to vocab/input space
        logits = self.gpt.lm_logits(attention_aware_tokens_seq)
        return logits


    @property
    def parameters(self):
        return [
            *self.vit.parameters,
            *self.gpt.parameters,
            *self.proj.parameters
        ]

    @property
    def num_img_tokens(self):
        return self.vit.pos.shape[0]


# %% [markdown]
# ## Char-level tokenizer for captions
#
# Same trick as nanoGPT. Reserve id 0 for PAD. We also need BOS and EOS so the
# model knows when to start and stop generating.

# %%
# Pretend vocab — built from the chars we'll ever see in captions
CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']
CAPTION_TEMPLATE = "a photo of a {}"  # → "a photo of a cat", etc.

# Reserved special tokens
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
SPECIAL = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

# Build vocab from all chars + specials
all_chars = sorted(set("".join(CAPTION_TEMPLATE.format(c) for c in CIFAR_CLASSES)))
char_to_id = {ch: i + 3 for i, ch in enumerate(all_chars)}  # leave 0,1,2 for specials
id_to_char = {i + 3: ch for i, ch in enumerate(all_chars)}
id_to_char.update(SPECIAL)
VOCAB_SIZE = len(char_to_id) + 3

# Pad to the longest caption + 2 (for BOS, EOS)
MAX_CAPTION_LEN = max(len(CAPTION_TEMPLATE.format(c)) for c in CIFAR_CLASSES) + 2
print(f"VOCAB_SIZE = {VOCAB_SIZE}  (incl. PAD/BOS/EOS)")
print(f"MAX_CAPTION_LEN = {MAX_CAPTION_LEN}")


def encode_caption(text):
    """text → (MAX_CAPTION_LEN,) tensor of ids:  [BOS] + chars + [EOS] + [PAD]*"""
    ids = [BOS_ID] + [char_to_id[c] for c in text] + [EOS_ID]
    ids = ids[:MAX_CAPTION_LEN]
    ids += [PAD_ID] * (MAX_CAPTION_LEN - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def decode_ids(ids):
    """ids tensor → string. Stops at first EOS, skips PAD/BOS."""
    out = []
    for i in ids.tolist():
        if i == EOS_ID:
            break
        if i in (PAD_ID, BOS_ID):
            continue
        out.append(id_to_char.get(i, "?"))
    return "".join(out)


# Pre-tokenize the 10 class captions
CAPTION_IDS = torch.stack([encode_caption(CAPTION_TEMPLATE.format(c)) for c in CIFAR_CLASSES])
print(f"CAPTION_IDS shape: {tuple(CAPTION_IDS.shape)}")
print(f"caption[0] decoded: {decode_ids(CAPTION_IDS[0])!r}")


# %% [markdown]
# ## CIFAR data + batch sampler
#
# Same convention as before. One image per class per batch (avoids duplicate captions).

# %%
_train = torchvision.datasets.CIFAR10(root="exploration/data", train=True, download=True)
_test = torchvision.datasets.CIFAR10(root="exploration/data", train=False, download=True)

X_train = torch.tensor(_train.data).permute(0, 3, 1, 2).float() / 255.0
X_test = torch.tensor(_test.data).permute(0, 3, 1, 2).float() / 255.0
y_train = torch.tensor(_train.targets)
y_test = torch.tensor(_test.targets)

_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
X_train = (X_train - _mean) / _std
X_test = (X_test - _mean) / _std

_train_by_class = [torch.where(y_train == k)[0] for k in range(10)]
_test_by_class = [torch.where(y_test == k)[0] for k in range(10)]


def get_batch(split="train"):
    """(images (10, 3, 32, 32), captions (10, MAX_CAPTION_LEN))."""
    by_class = _train_by_class if split == "train" else _test_by_class
    X = X_train if split == "train" else X_test
    chosen = torch.tensor([
        int(by_class[k][int(torch.randint(0, len(by_class[k]), (1,)).item())].item())
        for k in range(10)
    ])
    return X[chosen], CAPTION_IDS  # captions already in class order


# %% [markdown]
# ## Build the model
#
# Small dims so the laptop can train it in a couple minutes.

# %%
torch.manual_seed(42)

VIT_DIM = 64
GPT_DIM = 64
BLOCK_SIZE = 128  # must hold 65 image tokens + caption length

vit = MiniViT(embed_dim=VIT_DIM, num_blocks=3, num_heads=4, attention_dim=16, value_dim=16)
gpt = MiniGPT(vocab_size=VOCAB_SIZE, embed_dim=GPT_DIM, num_blocks=3, num_heads=4,
              attention_dim=16, value_dim=16, block_size=BLOCK_SIZE)
proj = Projector(vit_dim=VIT_DIM, gpt_dim=GPT_DIM)
llava = MiniLLaVA(vit, gpt, proj)

print(f"total params: {sum(p.numel() for p in llava.parameters):,}")
print(f"  ViT:       {sum(p.numel() for p in vit.parameters):,}")
print(f"  Projector: {sum(p.numel() for p in proj.parameters):,}")
print(f"  GPT:       {sum(p.numel() for p in gpt.parameters):,}")
print(f"image tokens: {llava.num_img_tokens}  (64 patches + 1 CLS)")


# %% [markdown]
# ## Training loop — the loss-masking trick
#
# This is the subtle part. The model emits logits over the ENTIRE concatenated
# sequence (image tokens + text tokens). But we only want to **supervise** the text
# predictions — we don't care what the model predicts at image-token positions.
#
# The standard LLaVA recipe:
#
# ```
# input  to LLM:   [img_0 ... img_64,  BOS, "a", " ", "p", ..., EOS]
#                  └──── 65 ────┘     └────── caption ─────┘
#
# target predict:  [ -,      ...    -,  "a", " ", "p", ..., EOS, -]
#                   ↑ ignore loss here   ↑ supervise here          ↑ last pos has no target
# ```
#
# In code: compute CE only over positions `[num_img_tokens, num_img_tokens + T - 1]`
# (these are the positions whose **next** token is a caption token we care about).
#
# Concretely, at position `num_img_tokens` the model sees the last image token and
# should predict `BOS`. At position `num_img_tokens + 1` it sees `BOS` and should
# predict the first caption char. Etc.

# %%
def llava_loss(llava: MiniLLaVA, images: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on caption positions only. Image positions are ignored.

    YOU build this.

    The shift trick — concretely:
      The model sees (img_tokens + caption_ids) as INPUT.
      Position N_img - 1 (the LAST image token) should predict caption_ids[0].
      Position N_img     (BOS) should predict caption_ids[1].
      Position N_img + T - 2 (second-to-last caption token) should predict caption_ids[T-1].

    So you want predictions at positions [N_img - 1 : N_img - 1 + T] vs targets caption_ids.

    Hints:
      - Use F.cross_entropy with reduction='none' so you can mask PAD positions.
      - Build a boolean mask: targets != PAD_ID.
      - Loss = (per-position-loss * mask).sum() / mask.sum().clamp(min=1)

    Ask yourself:
      - If you don't mask PAD, what happens? (Why would padded captions hurt training?)
      - Why do we slice logits at [N_img - 1 : N_img - 1 + T] and not [N_img : N_img + T]?
    """
    logits = llava.forward(images=images, text_ids=caption_ids)
    unmasked_loss = F.cross_entropy(logits[:, llava.num_img_tokens - 1:llava.num_img_tokens - 1 + caption_ids.shape[1]].flatten(0, 1), caption_ids.flatten(0), reduction='none')
    mask = torch.ones((caption_ids.flatten(0).shape[0], )).masked_fill(caption_ids.flatten(0) == PAD_ID, 0)
    loss = (unmasked_loss * mask).sum() / mask.sum().clamp(min=1)
    return loss


# %% [markdown]
# ## Manual Adam training loop
#
# Same template as your ViT / nanoGPT / CLIP runs. Watch the loss drop from
# ~log(VOCAB_SIZE) ≈ log(30) ≈ 3.4 down toward 0.5-ish.

# %%
lr = 3e-4
beta1, beta2, eps = 0.9, 0.999, 1e-8
m_state = [torch.zeros_like(p) for p in llava.parameters]
v_state = [torch.zeros_like(p) for p in llava.parameters]
step = 0
losses = []

NUM_STEPS = 4000
EVAL_EVERY = 200
t0 = time.time()

for it in range(NUM_STEPS):
    images, captions = get_batch("train")
    loss = llava_loss(llava, images, captions)

    for p in llava.parameters:
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    step += 1
    for i, p in enumerate(llava.parameters):
        if p.grad is None:
            continue
        m_state[i] = beta1 * m_state[i] + (1 - beta1) * p.grad
        v_state[i] = beta2 * v_state[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_state[i] / (1 - beta1 ** step)
        v_hat = v_state[i] / (1 - beta2 ** step)
        with torch.no_grad():
            p -= lr * m_hat / (v_hat.sqrt() + eps)

    losses.append(loss.item())

    if (it + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            xi, ti = get_batch("test")
            test_loss = llava_loss(llava, xi, ti).item()
        print(f"step {it+1:4d} | train {loss.item():.4f} | test {test_loss:.4f} | "
              f"{time.time() - t0:.1f}s")

print(f"\ntraining done in {time.time() - t0:.1f}s")


# %% [markdown]
# ## Loss curve

# %%
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(losses, alpha=0.3, label="raw")
window = 50
smoothed = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window) for i in range(len(losses))]
ax.plot(smoothed, label=f"smoothed (w={window})", color="orange", linewidth=2)
ax.axhline(y=float(torch.log(torch.tensor(float(VOCAB_SIZE)))), color="red", linestyle="--",
           label=f"log(vocab) = {np.log(VOCAB_SIZE):.2f} (random baseline)")
ax.set_xlabel("step")
ax.set_ylabel("loss")
ax.set_title("Mini LLaVA training loss — caption supervision")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## NEW Part 3 — generate(): autoregressive sampling
#
# Inference for LLaVA = "show me an image, give me a caption."
#
# Pipeline:
#   1. Encode image once with ViT + Projector → image_tokens (1, 65, GPT_DIM)
#   2. Start text with `BOS`. Embed it. Concat. Run GPT → logits at last position.
#   3. Sample next token from logits[-1].
#   4. Append to text, repeat until EOS or max_len.
#
# We do this with a clean forward call rather than caching activations (slower, but simpler).

# %%
@torch.no_grad()
def generate(
    llava: MiniLLaVA,
    image: torch.Tensor,
    max_new_tokens=MAX_CAPTION_LEN,
    temperature=1.0,
    greedy=True,
):
    """Generate a caption for one image. image: (3, 32, 32). Returns string.

    YOU build this.

    Pipeline:
      1. image.unsqueeze(0)  → (1, 3, 32, 32)        (model expects a batch)
      2. Run ViT + Projector ONCE to get image_tokens (1, 65, gpt_dim).
         Cache them — they don't change during the loop.
      3. Start text_ids with [[BOS_ID]] (shape (1, 1)).
      4. Loop until max_new_tokens (or until EOS sampled):
           a. Embed current text_ids via gpt.embed_tokens
           b. Concat image_tokens + text_emb
           c. Run gpt.run_blocks + gpt.lm_logits
           d. Take the LAST position's logits (shape (1, vocab))
           e. greedy: argmax; otherwise: softmax → multinomial sample
           f. Append the new token to text_ids
           g. If sampled EOS_ID, break
      5. Decode text_ids[0] back to string.

    Hints:
      - Use @torch.no_grad() at the top — you don't need gradients for inference. (Done.)
      - `logits[:, -1, :]` selects the last position. `.argmax(dim=-1, keepdim=True)` keeps the (1, 1) shape.
      - `torch.multinomial(probs, num_samples=1)` samples ONE token from a probability distribution.

    Ask yourself:
      - Why do we keep concat'ing the FULL text_ids every iteration instead of just appending?
        (Hint: it's because we don't have a KV-cache. Real implementations would cache for speed.)
      - Why does greedy mode tend to produce repetitive outputs?
    """
    image_tokens = llava.vit.forward(image.unsqueeze(0))
    llmized = llava.proj.forward(image_tokens)
    bos_embedding = llava.gpt.embed_tokens(torch.tensor([BOS_ID]).unsqueeze(0))

    assert llmized.shape == (1, llava.num_img_tokens, llava.proj.llm_dim)

    llm_input = torch.concat((llmized, bos_embedding), dim=1)

    generated_text = []
    chosen = torch.tensor([[BOS_ID]])  # start with BOS
    generated_ids = torch.tensor([[BOS_ID]])  # keep track of generated ids for decoding
    
    while (chosen and chosen[0, 0] != EOS_ID) and len(generated_text) < max_new_tokens:
        attention_aware_seq = llava.gpt.run_blocks(llm_input)

        # (B, VOCAB_SIZE)  
        batch_logits = llava.gpt.lm_logits(attention_aware_seq)
        logits = batch_logits[:, -1, :] / temperature

        if greedy:
            # (B, 1)
            chosen = logits.argmax(dim=-1, keepdim=True)
        else:
            # (B, 1)
            chosen = torch.multinomial(torch.softmax(logits, dim=-1), 1)

        # append generated token back to next input (autoregression)
        generated_ids = torch.concat((generated_ids, chosen), dim=1)
        logits_embedded = llava.gpt.embed_tokens(generated_ids)

        llm_input = torch.concat((llmized, logits_embedded), dim=1)

        # decode the text
        text = decode_ids(chosen[0])

        # append to the text
        generated_text.append(text)
    return "".join(generated_text)


# %% [markdown]
# ## Demo — caption some unseen test images
#
# Pick one image from each class, ask the model to caption it, print result.

# %%
print("MiniLLaVA captions on unseen test images:\n")
print(f"{'class':<12} | {'predicted caption':<35} | {'true caption':<25}")
print("-" * 80)

correct_class_in_caption = 0
for k in range(10):
    test_imgs, _ = get_batch("test")
    test_img = test_imgs[k]   # one image of class k

    pred = generate(llava, test_img, greedy=True)
    true = CAPTION_TEMPLATE.format(CIFAR_CLASSES[k])

    # Loose match: does the predicted caption contain the true class name?
    contains_class = CIFAR_CLASSES[k] in pred
    if contains_class:
        correct_class_in_caption += 1

    mark = "✓" if contains_class else "✗"
    print(f"{CIFAR_CLASSES[k]:<12} | {pred!r:<35} | {true!r:<25} {mark}")

print(f"\nClass-name-in-caption accuracy: {correct_class_in_caption}/10")


# %% [markdown]
# ## What you should see
#
# - **Loss** drops from ~3.4 (random over 30-symbol vocab) to ~0.3-0.7 over 2000 steps.
# - **Generations** are grammatical (the model learned the template "a photo of a X").
# - **Class word** in the prediction matches the actual image class for *some* classes,
#   not all. CIFAR is tiny + low-res; a 3-block ViT can't perceive much.
#   Expect 4-7 out of 10 correct after 2000 steps.
#
# **The thing to notice:** even with a tiny ViT that gets ~55% accuracy on
# CIFAR-10 *with a classification head*, the same ViT + a projector + a fresh GPT
# can learn to **generate the correct class word**, autoregressively, from the
# image alone. That's the entire LLaVA recipe, in 600 lines.
#
# ### What's missing from "real" LLaVA
#
# 1. **Pretrained components**: real LLaVA uses CLIP-ViT-L/14 (frozen) and LLaMA-7B (frozen-then-tuned).
#    Ours train both from scratch on tiny data — that's why generation is mushy.
# 2. **Two-stage training**: real LLaVA pretrains the projector alone first, THEN fine-tunes the LLM.
#    We jam-train everything together.
# 3. **Instruction tuning**: real LLaVA is trained on (instruction, image, answer) triplets, not
#    plain captions. That's what makes it answer arbitrary questions vs just describe.
# 4. **Scale**: ~7B vs our ~30K params.
#
# Everything else — the architecture, the loss-masking trick, the autoregressive
# generation, the image-tokens-in-LLM-space idea — is **identical**. You've now
# written the same forward pass that runs inside `llava-hf/llava-1.5-7b-hf`,
# just smaller.

# %%
