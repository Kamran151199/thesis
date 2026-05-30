# %% [markdown]
# # 03: Vision Transformer (ViT) — How Transformers See
# Roadmap: Days 5-6
# Prerequisite: 02_transformer_encoder_decoder.py

# %% [markdown]
# 
# ### What this file implements:
# 1. Patch embedding (image → sequence of patches → embeddings)
# 2. CLS token and positional embeddings
# 3. ViT forward pass
# 4. Train tiny ViT on CIFAR-10
# 5. Visualize attention maps and position embeddings


# %% [markdown]
# #### Patch Embedding
# Patch Embedding is basically a function that takes an image
# and outputs the patches of that image where each patch 
# is encoded into an internal representation (embedding).
# Basically, we are trying to transform (Batch, Channel, Heigh, Width)
# to sth like (Batch, PatchSequence, EmbeddedRepresentationOfThatPatch)




# %%
import torch

class PatchEmbedding:
    def __init__(
        self,
        channels: int = 3,
        img_height: int = 32,
        img_width: int = 32,
        patch_height: int = 4,
        patch_width: int = 4,
        embedding_dim: int = 32
    ):
        self.channels = channels
        self.img_height = img_height
        self.img_width = img_width
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.embedding_dim = embedding_dim

        # assert the constraints
        assert img_height % patch_height == 0
        assert img_width % patch_width == 0

        # derive patch num and dim
        self.num_patches = int((img_height * img_width) / (patch_height * patch_width))
        self.patch_vector_dim = channels * patch_height * patch_width

        self.embedding = (torch.randn(embedding_dim, self.patch_vector_dim) * 1 / self.patch_vector_dim ** 0.5).requires_grad_(True)
    
    def forward(
        self,
        batch_images: torch.Tensor,
    ) -> torch.Tensor:
        B, C, _, _ = batch_images.shape

        patches = []

        h_start = 0
        h_end = self.patch_height

        for _ in range(self.img_height // self.patch_height):
            w_start = 0
            w_end = self.patch_width
            for __ in range(self.img_width // self.patch_width):
                patches.append(batch_images[:, :, h_start:h_end, w_start:w_end].reshape(B, C * self.patch_height * self.patch_width))
                w_start += self.patch_width
                w_end += self.patch_width
            h_start += self.patch_height
            h_end += self.patch_height

        # list of (B, C * PH * PW)        
        assert len(patches) == self.num_patches

        patched = torch.stack(patches, dim=0) # (NumPatches, B, C*PH*PW)
        patched = patched.transpose(1, 0) 
        assert patched.shape == (B, self.num_patches, C * self.patch_height * self.patch_width)

        embedded = patched @ self.embedding.T

        return embedded

    @property
    def parameters(self):
        return [
            self.embedding
        ]



# %% [markdown]
# Test it out


# %%
random_img = torch.randn((1, 3, 32, 32))
pe = PatchEmbedding(3, 32, 32, 4, 4, 10)
embedded_img = pe.forward(random_img)
print(embedded_img.shape)  # must be (1, 64, 10) 

# %% [markdown]
# #### Build the ViT

# %% [markdown]
# #### Copy-paste previous lesson's components
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

class TransformerBlock:
    def __init__(
        self,
        input_vocab_dim: int,
        attention_dim: int,
        value_dim: int,
        num_heads: int,
        expansion: int = 4,
        causal_mask: bool = True,
    ):
        self.input_vocab_dim = input_vocab_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.expansion = expansion
        self.causal_mask = causal_mask

        # for input normalization
        self.ln1 = LayerNorm(input_vocab_dim=input_vocab_dim)

        # for each token to attend to each other from different perspectives
        self.mh_attention = MultiHeadAttention(
            input_vocab_dim=input_vocab_dim,
            attention_dim=attention_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            enable_causal_mask=causal_mask,
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


# %%
class VisionTransformer:
    def __init__(
        self,
        img_shape: tuple[int, int, int],
        patch_shape: tuple[int, int],
        embedding_dim: int,
        attention_dim: int,
        value_dim: int,
        num_heads: int,
        expansion: int,
        num_blocks: int,
        num_classes: int
    ):
        # Image Related Setup
        self.img_c, self.img_h, self.img_w = img_shape
        self.patch_h, self.patch_w = patch_shape
        self.num_patches = (self.img_h * self.img_w) // (self.patch_h * self.patch_w)
        
        # Embedding related
        self.embedding_dim = embedding_dim
        
        # Transformer related
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.expansion = expansion  # the hidden layer dim of the FFN 

        # Classification related
        self.num_classes = num_classes

        # Components initializations

        ## patches the image to (B, T, E)
        self.patcher = PatchEmbedding(
            channels=self.img_c,
            img_height=self.img_h,
            img_width=self.img_w,
            patch_height=self.patch_h,
            patch_width=self.patch_w,
            embedding_dim=self.embedding_dim
        )

        ## gives the positional info for each Token in a patch sequence (NP + 1, E).
        ### "+1" is needed to be able to consider the summariser token.
        self.position_embedding = (torch.randn(self.num_patches + 1, self.embedding_dim) * 1 / self.embedding_dim ** 0.5).requires_grad_(True)

        ## this is the token we inject into a sequence
        ## so that after attentions, it gathers whole summary of the image (from image patches)
        self.summarizer_token = (torch.randn(1, self.embedding_dim) * 1 / self.embedding_dim ** 0.5).requires_grad_(True)

        ## Transformer as usual enriches each token with the context of all other tokens in a sequence
        self.transformers = [
            TransformerBlock(
                input_vocab_dim=self.embedding_dim,
                attention_dim=self.attention_dim,
                value_dim=self.value_dim,
                num_heads=self.num_heads,
                expansion=self.expansion,
                causal_mask=False,
            )
            for _ in range(self.num_blocks)
        ]

        ## normalize the output of the transformer
        self.ln = LayerNorm(input_vocab_dim=self.embedding_dim)

        ## classifier
        self.classifier = (torch.randn(self.num_classes, self.embedding_dim) * 1 / self.embedding_dim ** 0.5).requires_grad_(True)
    
    def forward(self, batch_images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_images.shape

        internalized = self.patcher.forward(batch_images=batch_images)
        assert internalized.shape == (B, self.num_patches, self.embedding_dim)


        token_sequences = torch.cat([self.summarizer_token.expand(B, -1, -1), internalized], dim=1)
        assert token_sequences.shape == (B, self.num_patches + 1, self.embedding_dim)

        sequence = token_sequences + self.position_embedding

        for tb in self.transformers:
            sequence = tb.forward(sequence)
        
        normalized = self.ln.forward(sequence)
        assert normalized.shape == (B, self.num_patches + 1, self.embedding_dim)

        logits = normalized[:, 0, :] @ self.classifier.T
        assert logits.shape == (B, self.num_classes)

        return logits

    @property
    def parameters(self) -> list[torch.Tensor]:
        params = []
        params.extend(self.patcher.parameters)       # list → extend
        params.append(self.position_embedding)        # tensor → append (NOT extend!)
        params.append(self.summarizer_token)          # tensor → append
        for tb in self.transformers:
            params.extend(tb.parameters)              # list → extend
        params.extend(self.ln.parameters)             # list → extend
        params.append(self.classifier)                # tensor → append
        return params


# %% [markdown]
# ## CIFAR-10 data loader
#
# 50,000 train + 10,000 test images, 32x32 RGB, 10 classes.
# We load the raw uint8 arrays, convert to (N, 3, 32, 32) float tensors,
# normalize per-channel, then sample random batches (same get_batch pattern
# as the nanoGPT char data).

# %%
import torchvision

_train = torchvision.datasets.CIFAR10(root="exploration/data", train=True, download=True)
_test = torchvision.datasets.CIFAR10(root="exploration/data", train=False, download=True)

CIFAR_CLASSES = _train.classes   # ['airplane', 'automobile', 'bird', ...]

# .data is (N, 32, 32, 3) uint8 → permute to (N, 3, 32, 32) float in [0, 1]
X_train = torch.tensor(_train.data).permute(0, 3, 1, 2).float() / 255.0
X_test = torch.tensor(_test.data).permute(0, 3, 1, 2).float() / 255.0
y_train = torch.tensor(_train.targets)
y_test = torch.tensor(_test.targets)

# per-channel normalize (standard CIFAR-10 mean/std) — helps training stability
_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
X_train = (X_train - _mean) / _std
X_test = (X_test - _mean) / _std

print(f"X_train {tuple(X_train.shape)}  y_train {tuple(y_train.shape)}")
print(f"X_test  {tuple(X_test.shape)}  y_test  {tuple(y_test.shape)}")
print(f"classes: {CIFAR_CLASSES}")


def get_batch(split: str, batch_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = (X_train, y_train) if split == "train" else (X_test, y_test)
    ix = torch.randint(len(X), (batch_size,))
    return X[ix], y[ix]   # (B, 3, 32, 32) images, (B,) integer labels 0-9


xb, yb = get_batch("train")
print(f"\nbatch: images {tuple(xb.shape)}  labels {tuple(yb.shape)}")
print("label names:", [CIFAR_CLASSES[i] for i in yb[:5].tolist()])


# %% [markdown]
# ### Peek at a few images (sanity check the data looks right)

# %%
import matplotlib.pyplot as plt


fig, axes = plt.subplots(1, 6, figsize=(14, 3))
imgs, labels = get_batch("train", batch_size=6)
for ax, img, lab in zip(axes, imgs, labels):
    # un-normalize for display: x*std + mean, then clamp to [0,1]
    disp = (img * _std[0] + _mean[0]).permute(1, 2, 0).clamp(0, 1)
    ax.imshow(disp)
    ax.set_title(CIFAR_CLASSES[lab.item()], fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.show()

# %%
import torch.nn.functional as F

vit = VisionTransformer(
    img_shape=(3, 32, 32),
    patch_shape=(4, 4),
    embedding_dim=64,
    attention_dim=128,
    value_dim=128,
    num_heads=4,
    expansion=4,
    num_blocks=4,
    num_classes=len(CIFAR_CLASSES)
)

epochs = 5_000
eval_after = 500
lr = 1e-3

params = vit.parameters
print(f"parameter tensors: {len(params)}")
print(f"scalar params: {sum(p.numel() for p in params):,}")

# manual Adam state (same pattern as nanoGPT)
m = [torch.zeros_like(p) for p in params]
v = [torch.zeros_like(p) for p in params]
beta1, beta2, eps = 0.9, 0.999, 1e-8


@torch.no_grad()
def estimate(split: str, batches: int = 10) -> tuple[float, float]:
    losses, accs = [], []
    for _ in range(batches):
        xb, yb = get_batch(split)
        logits = vit.forward(xb)
        losses.append(F.cross_entropy(logits, yb).item())
        accs.append((logits.argmax(dim=-1) == yb).float().mean().item())
    return sum(losses) / len(losses), sum(accs) / len(accs)


for step in range(1, epochs + 1):
    batch_images, batch_labels = get_batch("train")

    logits = vit.forward(batch_images=batch_images)        # (B, 10) RAW logits
    # F.cross_entropy applies log_softmax INTERNALLY — feed raw logits, NOT softmax'd probs
    loss = F.cross_entropy(logits, batch_labels)

    loss.backward()

    # manual Adam update
    with torch.no_grad():
        for j, p in enumerate(params):
            g = p.grad
            m[j] = beta1 * m[j] + (1 - beta1) * g
            v[j] = beta2 * v[j] + (1 - beta2) * g * g
            m_hat = m[j] / (1 - beta1 ** step)
            v_hat = v[j] / (1 - beta2 ** step)
            p -= lr * m_hat / (v_hat.sqrt() + eps)
            p.grad = None

    if step % eval_after == 0 or step == 1:
        tr_loss, tr_acc = estimate("train")
        te_loss, te_acc = estimate("test")
        print(f"step {step:>5}: train loss {tr_loss:.3f} acc {tr_acc:.1%} | "
              f"test loss {te_loss:.3f} acc {te_acc:.1%}")


# %% [markdown]
# ## PEEK 1: Where does the model LOOK? (CLS attention overlay)
#
# The CLS token's attention over the patches = "which regions did the model
# use to classify?" We grab it from the LAST block (most semantic), average
# over heads, reshape back to the 8x8 patch grid, and overlay on the image.
#
# This is a baby version of EXPLANATION GROUNDING — the core of your thesis.

# %%
imgs, labels = get_batch("test", batch_size=6)
with torch.no_grad():
    logits = vit.forward(imgs)
preds = logits.argmax(dim=-1)

# CLS attention from the LAST transformer block, averaged over heads
attn = vit.transformers[-1].mh_attention.last_attention   # (B, H, 65, 65)
cls_attn = attn[:, :, 0, 1:].mean(dim=1)                   # (B, 64): CLS→patches, avg heads
grid_side = vit.patcher.img_height // vit.patcher.patch_height   # 8

# plt.subplots(rows, cols) makes a grid of sub-plots. Returns:
#   fig  = the whole figure (the canvas)
#   axes = a (2, 6) array of Axes objects — axes[row, col] is one sub-plot
# figsize=(width, height) in inches.
fig, axes = plt.subplots(2, 6, figsize=(15, 5))

for i in range(6):                              # loop over the 6 images (columns)
    # --- prepare the image for display ---
    # imgs[i] is (C, H, W) and NORMALIZED. Un-normalize: x*std + mean → back to [0,1]-ish.
    # .permute(1,2,0): (C,H,W) → (H,W,C) because imshow wants channels LAST.
    # .clamp(0,1): squash any out-of-range values into [0,1] so colors render correctly.
    disp = (imgs[i] * _std[0] + _mean[0]).permute(1, 2, 0).clamp(0, 1)

    # --- TOP ROW: the original image ---
    axes[0, i].imshow(disp)                     # draw the image in sub-plot [row 0, col i]
    correct = "✓" if preds[i] == labels[i] else "✗"   # tick if prediction matched truth
    axes[0, i].set_title(                       # title above the sub-plot
        f"{CIFAR_CLASSES[labels[i]]} | pred:{CIFAR_CLASSES[preds[i]]} {correct}", fontsize=7)
    axes[0, i].axis("off")                       # hide the x/y axis ticks & frame

    # --- BOTTOM ROW: attention heatmap overlaid on the image ---
    # cls_attn[i] is (64,) — one attention weight per patch. Reshape to the 8×8 patch grid.
    heat = cls_attn[i].reshape(grid_side, grid_side)
    # the heat grid is 8×8 but the image is 32×32. Upsample each patch-cell to its
    # patch_height × patch_width pixel block so the heatmap lines up with the image.
    # repeat_interleave(n, dim) repeats each element n times along that dim:
    #   (8,8) --repeat rows ×4--> (32,8) --repeat cols ×4--> (32,32)
    heat = heat.repeat_interleave(vit.patcher.patch_height, 0).repeat_interleave(vit.patcher.patch_width, 1)
    axes[1, i].imshow(disp)                      # draw the image first (the base layer)
    # draw the heatmap ON TOP. cmap="jet" = blue→red colormap. alpha=0.5 = 50% transparent
    # so the image shows through. .detach() because heat still carries grad history.
    axes[1, i].imshow(heat.detach(), cmap="jet", alpha=0.5)
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("image", fontsize=9)       # label on the leftmost top sub-plot
fig.suptitle("CLS attention: top = image, bottom = where the model looked", fontsize=12)  # title for the WHOLE figure
plt.tight_layout()                               # auto-adjust spacing so nothing overlaps
plt.show()                                       # render it


# %% [markdown]
# ## PEEK 2: Did position embeddings learn the 2D spatial grid?
#
# The famous ViT result: each patch's position vector ends up most similar to
# its SPATIAL NEIGHBORS — the model recreates the 8x8 grid layout with no
# explicit spatial info given. Each subplot = one patch's similarity to all 64.

# %%
P = vit.position_embedding[1:].detach()        # (64, E) — skip CLS slot at index 0
P = P / P.norm(dim=-1, keepdim=True)            # normalize for cosine sim
sim = P @ P.T                                   # (64, 64)

fig, axes = plt.subplots(grid_side, grid_side, figsize=(8, 8))
for i in range(grid_side * grid_side):
    ax = axes[i // grid_side, i % grid_side]
    ax.imshow(sim[i].reshape(grid_side, grid_side), cmap="viridis")
    ax.axis("off")
fig.suptitle("Position embedding similarity\n(each cell = one patch's similarity to all others — bright spot should match its location)", fontsize=11)
plt.tight_layout()
plt.show()
# %%
