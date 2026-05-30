# %% [markdown]
# # 04: CLIP — Contrastive Vision-Language Alignment
# Roadmap: Days 7-8
# Prerequisite: 03_vision_transformer.py

# %%
# TODO: Build after completing 03
# This file will implement:
# 1. Load pre-trained CLIP, trace the architecture
# 2. Encode images + texts, compute similarity matrix
# 3. Test CLIP on documents/charts (where it struggles)
# 4. Extract patch-level features (what VLMs use)
# 5. Understand contrastive loss (InfoNCE)

# %% [markdown]
# ### Imports

# %%
import torch
import torch.nn.functional as F


# %% [markdown]
# ### Prev Lesson artifacts

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
# ### [NEW] Image Encoder

# %%
class ImageEncoder:
    def __init__(
        self,
        img_shape: tuple[int, int, int] = (3, 224, 224),
        patch_shape: tuple[int, int] = (4, 4),
        embedding_dim: int = 128,
        attention_dim: int = 32,
        value_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 4,
        expansion: int = 4,
        projection_dim: int = 128
    ):
        self.img_c, self.img_h, self.img_w = img_shape
        self.patch_h, self.patch_w = patch_shape
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.expansion = expansion

        # This is the dimensionality of the common space where we gonna encode our images AND the texts.
        self.projection_dim = projection_dim

        # COMPONENTS
        # tokenizes the image
        self.patcher = PatchEmbedding(
            channels=self.img_c,
            img_height=self.img_h,
            img_width=self.img_w,
            patch_height=self.patch_h,
            patch_width=self.patch_w,
            embedding_dim=embedding_dim
        )

        # summarizer token
                ## gives the positional info for each Token in a patch sequence (NP + 1, E).
        ### "+1" is needed to be able to consider the summariser token.
        self.position_embedding = (torch.randn(self.patcher.num_patches + 1, self.embedding_dim) * 1 / self.embedding_dim ** 0.5).requires_grad_(True)

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

        ## Project the summary token to shared embedding space
        self.projector: torch.Tensor = (torch.randn(self.projection_dim, self.embedding_dim) * 1 / self.embedding_dim ** 0.5).requires_grad_(True)


    def forward(self, batch_images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_images.shape

        internalized = self.patcher.forward(batch_images=batch_images)
        assert internalized.shape == (B, self.patcher.num_patches, self.embedding_dim)


        token_sequences = torch.cat([self.summarizer_token.expand(B, -1, -1), internalized], dim=1)
        assert token_sequences.shape == (B, self.patcher.num_patches + 1, self.embedding_dim)

        sequence = token_sequences + self.position_embedding

        for tb in self.transformers:
            sequence = tb.forward(sequence)
        
        normalized = self.ln.forward(sequence)
        assert normalized.shape == (B, self.patcher.num_patches + 1, self.embedding_dim)

        summarizer_token = normalized[:, 0, :]

        projected = summarizer_token @ self.projector.T
        assert projected.shape == (B, self.projection_dim)

        return projected


    @property
    def parameters(self) -> list[torch.Tensor]:
        params = []
        params.extend(self.patcher.parameters)       # list → extend
        params.append(self.position_embedding)        # tensor → append (NOT extend!)
        params.append(self.summarizer_token)          # tensor → append
        for tb in self.transformers:
            params.extend(tb.parameters)              # list → extend
        params.extend(self.ln.parameters)             # list → extend
        params.append(self.projector)
        return params

IMAGE_ENCODER_DEFAULT_PARAMS = dict(
    img_shape=(3, 224, 224),
    patch_shape=(4, 4),
    embedding_dim=128,
    attention_dim=32,
    value_dim=32,
    num_heads=4,
    num_blocks=4,
    expansion=4,
    # NOTE: projection_dim is owned by CLIP — don't set here (avoid duplicate kwarg).
)

class TextEncoder:
    def __init__(
        self,
        block_size: int = 32,
        vocab_size: int = 64,
        embedding_dim: int = 64,
        attention_dim: int = 16,
        value_dim: int = 16,
        num_heads: int = 4,
        num_blocks: int = 4,
        expansion: int = 4,
        causal_mask: bool = True,
        projection_dim: int = 128,
    ):
        self.block_size = block_size  # this is the "context size"
        self.vocab_size = vocab_size
        self.input_vocab_dim = embedding_dim  # this is "d_model"
        self.attention_dim = attention_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.ffn_expansion = expansion
        self.projection_dim = projection_dim


        # token embedding: (vocab_size, E) — lookup table, row k = vector for token id k
        self.token_embedding = (torch.randn(vocab_size, embedding_dim) * 0.1).requires_grad_(True)
        # position embedding: (block_size, E) — lookup table, row p = vector for position p
        self.positional_embedding = (torch.randn(block_size, embedding_dim) * 0.1).requires_grad_(True)


        self.transformer_blocks = [TransformerBlock(
            input_vocab_dim=embedding_dim,
            attention_dim=attention_dim,
            value_dim=value_dim,
            expansion=expansion,
            num_heads=num_heads,
            causal_mask=causal_mask
        ) for _ in range(num_blocks)]

        self.ln = LayerNorm(input_vocab_dim=embedding_dim)
        self.projector = (torch.randn(projection_dim, embedding_dim) * 1 / embedding_dim ** 0.5).requires_grad_(True)
    

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # Step1: Embed the input tokens
        # Step2: Add the positional embedding
        # Step3: Run the transformer blocks
        # Step4: Normalize the output
        # Step5: Run the projection

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

        last_token_emb = normalized[:, -1, :] @ self.projector.T
        assert last_token_emb.shape == (B, self.projection_dim)

        return last_token_emb

    @property
    def parameters(self) -> list[torch.Tensor]:
        # own tensors + every block's parameters (which recurse into attn/ffn/ln)
        params = [
            self.token_embedding,
            self.positional_embedding,
            self.projector,
            *self.ln.parameters,
        ]
        for block in self.transformer_blocks:
            params += block.parameters
        return params

TEXT_ENCODER_DEFAULT_PARAMS = dict(
    block_size=32,
    vocab_size=64,
    embedding_dim=64,
    attention_dim=16,
    value_dim=16,
    num_heads=4,
    num_blocks=4,
    expansion=4,
    causal_mask=True,
    # NOTE: projection_dim is owned by CLIP — don't set here (avoid duplicate kwarg).
)

class CLIP:
    def __init__(
        self,
        projector_dim: int = 128,
        img_encoder_params: dict | None = None,
        text_encoder_params: dict | None = None,
    ):
        img_encoder_params = IMAGE_ENCODER_DEFAULT_PARAMS | (img_encoder_params or {})
        text_encoder_params = TEXT_ENCODER_DEFAULT_PARAMS | (text_encoder_params or {})

        self.projector_dim = projector_dim
        self.image_encoder = ImageEncoder(**img_encoder_params, projection_dim=projector_dim)
        self.text_encoder = TextEncoder(**text_encoder_params, projection_dim=projector_dim)

        # Learnable temperature, stored in LOG-space so it stays positive after exp().
        # CLIP paper init: log(1/0.07) ≈ 2.6593 → temperature starts at ~14.29.
        # Higher temperature == sharper softmax == stronger pull on the diagonal.
        self.log_temperature = torch.tensor(2.6593, requires_grad=True)

    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """Returns the temperature-scaled similarity matrix (B, B).

        Each ROW i = how similar image i is to every text in the batch.
        Each COL j = how similar text  j is to every image in the batch.
        Diagonal = matching pairs (the positives).
        """
        B = images.shape[0]

        image_meaning = self.image_encoder.forward(images)
        text_meaning = self.text_encoder.forward(texts)
        assert image_meaning.shape == text_meaning.shape == (B, self.projector_dim)

        # L2-normalize each row so the dot product = cosine similarity ∈ [-1, 1].
        # Without this, the model could cheat by inflating vector magnitudes.
        image_unit = image_meaning / image_meaning.norm(dim=-1, keepdim=True)
        text_unit = text_meaning / text_meaning.norm(dim=-1, keepdim=True)

        # Temperature scales the logits BEFORE softmax. Clamp prevents runaway (paper detail).
        temperature = self.log_temperature.exp().clamp(max=100.0)
        similarity_matrix = (image_unit @ text_unit.T) * temperature

        assert similarity_matrix.shape == (B, B)
        return similarity_matrix

    def loss(self, images: torch.Tensor, texts: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE loss.

        For each ROW (image): the "correct text" is at column = row index.
            -> cross_entropy(sim, [0, 1, 2, ..., B-1])
        For each COL (text): the "correct image" is at row = col index.
            -> cross_entropy(sim.T, [0, 1, 2, ..., B-1])
        Average the two directions — that's why it's "symmetric" InfoNCE.
        """
        sim = self.forward(images, texts)
        B = sim.shape[0]
        labels = torch.arange(B, device=sim.device)

        loss_img2txt = F.cross_entropy(sim, labels)     # rows = images
        loss_txt2img = F.cross_entropy(sim.T, labels)   # rows = texts
        return (loss_img2txt + loss_txt2img) / 2

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [
            *self.image_encoder.parameters,
            *self.text_encoder.parameters,
            self.log_temperature,
        ]


# %% [markdown]
# ### CIFAR-10 + char-level tokenizer
#
# We need (image, caption) pairs. CIFAR gives us class IDs; we turn each ID
# into the string `"a photo of a {classname}"` and tokenize at char level
# (same tiny-vocab trick as in nanoGPT — keeps the text encoder small).

# %%
import torchvision

_train = torchvision.datasets.CIFAR10(root="exploration/data", train=True, download=True)
_test = torchvision.datasets.CIFAR10(root="exploration/data", train=False, download=True)
CIFAR_CLASSES = _train.classes  # ['airplane', 'automobile', 'bird', ...]

# Images -> (N, C, H, W) floats in [0, 1] -> per-channel normalized
X_train = torch.tensor(_train.data).permute(0, 3, 1, 2).float() / 255.0
X_test = torch.tensor(_test.data).permute(0, 3, 1, 2).float() / 255.0
y_train = torch.tensor(_train.targets)
y_test = torch.tensor(_test.targets)

_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
X_train = (X_train - _mean) / _std
X_test = (X_test - _mean) / _std

print(f"X_train {tuple(X_train.shape)}  X_test {tuple(X_test.shape)}")
print(f"classes: {CIFAR_CLASSES}")


# %% [markdown]
# ### Build captions + char-level vocab
#
# Reserve token id 0 for PAD. Everything else gets a sequential id.

# %%
CAPTIONS = [f"a photo of a {c}" for c in CIFAR_CLASSES]
print("captions:")
for i, cap in enumerate(CAPTIONS):
    print(f"  {i}: {cap!r}")

_chars = sorted(set("".join(CAPTIONS)))
PAD_ID = 0
char_to_id = {ch: i + 1 for i, ch in enumerate(_chars)}
id_to_char = {i + 1: ch for i, ch in enumerate(_chars)}
VOCAB_SIZE = len(char_to_id) + 1               # +1 for PAD
BLOCK_SIZE = max(len(c) for c in CAPTIONS)     # longest caption fits exactly

print(f"vocab_size: {VOCAB_SIZE}, block_size: {BLOCK_SIZE}")


def encode(text: str, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """char-level text -> (block_size,) token ids, right-padded with PAD_ID."""
    ids = [char_to_id[c] for c in text][:block_size]
    ids += [PAD_ID] * (block_size - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# Pre-tokenize all 10 class captions: (NUM_CLASSES, BLOCK_SIZE)
CAPTION_IDS = torch.stack([encode(c) for c in CAPTIONS])
print(f"CAPTION_IDS shape: {tuple(CAPTION_IDS.shape)}")


# %% [markdown]
# ### Batch sampler — ONE image per class
#
# Wrinkle: CIFAR has only 10 classes. A random batch of, say, 32 would have
# DUPLICATE captions in it -> the diagonal is no longer the *unique* positive,
# which breaks `labels = arange(N)`.
#
# Mini-build fix: sample exactly one image per class -> batch_size = 10,
# all 10 captions distinct, clean diagonal. (Real CLIP solves this by
# having millions of unique captions, so collisions are rare.)

# %%
_train_by_class = [torch.where(y_train == k)[0] for k in range(len(CIFAR_CLASSES))]
_test_by_class = [torch.where(y_test == k)[0] for k in range(len(CIFAR_CLASSES))]


def get_clip_batch(split: str = "train") -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (images (10, 3, 32, 32), texts (10, BLOCK_SIZE)).
    Row k = one random image of class k + the caption for class k."""
    by_class = _train_by_class if split == "train" else _test_by_class
    X = X_train if split == "train" else X_test
    chosen = torch.tensor([
        int(by_class[k][int(torch.randint(0, len(by_class[k]), (1,)).item())].item())
        for k in range(len(CIFAR_CLASSES))
    ])
    images = X[chosen]
    texts = CAPTION_IDS  # already in class order 0..9
    return images, texts


xb, tb = get_clip_batch("train")
print(f"images: {tuple(xb.shape)}, texts: {tuple(tb.shape)}")
print(f"caption[0] decoded: {''.join(id_to_char.get(int(i.item()), '_') for i in tb[0] if int(i.item()) != PAD_ID)!r}")


# %% [markdown]
# ### Build CLIP — sized for CIFAR

# %%
torch.manual_seed(42)

clip = CLIP(
    projector_dim=64,
    img_encoder_params={
        "img_shape": (3, 32, 32),
        "patch_shape": (4, 4),
        "embedding_dim": 64,
        "attention_dim": 32,
        "value_dim": 32,
        "num_heads": 4,
        "num_blocks": 3,
        "expansion": 4,
    },
    text_encoder_params={
        "block_size": BLOCK_SIZE,
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": 64,
        "attention_dim": 32,
        "value_dim": 32,
        "num_heads": 4,
        "num_blocks": 3,
        "expansion": 4,
        "causal_mask": True,
    },
)
print(f"total scalar params: {sum(p.numel() for p in clip.parameters):,}")


# %% [markdown]
# ### Manual Adam training loop
#
# Same Adam-from-scratch template as ViT/nanoGPT. Random-init loss should sit
# near log(10) ≈ 2.30 (uniform over 10 classes); we want it to drop well below.

# %%
import time

lr = 3e-4
beta1, beta2, eps = 0.9, 0.999, 1e-8
step = 0
m_state = [torch.zeros_like(p) for p in clip.parameters]
v_state = [torch.zeros_like(p) for p in clip.parameters]

losses: list[float] = []
temps: list[float] = []
test_losses: list[tuple[int, float]] = []

NUM_STEPS = 5_000
EVAL_EVERY = 200
EVAL_BATCHES = 20
t0 = time.time()

for it in range(NUM_STEPS):
    images, texts = get_clip_batch("train")
    loss = clip.loss(images, texts)

    # zero grads
    for p in clip.parameters:
        if p.grad is not None:
            p.grad.zero_()

    loss.backward()

    # Adam update
    step += 1
    for i, p in enumerate(clip.parameters):
        if p.grad is None:
            continue
        m_state[i] = beta1 * m_state[i] + (1 - beta1) * p.grad
        v_state[i] = beta2 * v_state[i] + (1 - beta2) * p.grad ** 2
        m_hat = m_state[i] / (1 - beta1 ** step)
        v_hat = v_state[i] / (1 - beta2 ** step)
        with torch.no_grad():
            p -= lr * m_hat / (v_hat.sqrt() + eps)

    losses.append(loss.item())
    temps.append(clip.log_temperature.exp().item())

    if (it + 1) % EVAL_EVERY == 0:
        with torch.no_grad():
            tl = 0.0
            for _ in range(EVAL_BATCHES):
                xi, ti = get_clip_batch("test")
                tl += clip.loss(xi, ti).item()
            tl /= EVAL_BATCHES
        test_losses.append((it + 1, tl))
        elapsed = time.time() - t0
        print(f"step {it+1:4d} | train {loss.item():.4f} | test {tl:.4f} | "
              f"temp {clip.log_temperature.exp().item():.2f} | {elapsed:.1f}s")

print(f"\ntraining done in {time.time() - t0:.1f}s")


# %% [markdown]
# ### Viz 1 — loss curve + temperature evolution
#
# Loss should drop from ~log(10)=2.30 toward 0. Temperature usually drifts
# (model self-tunes softmax sharpness as confidence changes).

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

# smoothed train loss
window = 50
smoothed = [sum(losses[max(0, i - window):i + 1]) / min(i + 1, window)
            for i in range(len(losses))]

axes[0].plot(losses, alpha=0.25, label="raw train")
axes[0].plot(smoothed, label=f"smoothed (w={window})", color="orange", linewidth=2)
axes[0].axhline(y=float(torch.log(torch.tensor(10.0))), color="red", linestyle="--",
                label="log(10) = random baseline")
if test_losses:
    xs, ys = zip(*test_losses)
    axes[0].plot(xs, ys, "o-", color="green", label="test loss (eval)", markersize=4)
axes[0].set_xlabel("step")
axes[0].set_ylabel("symmetric InfoNCE loss")
axes[0].set_title("Loss — diagonal pulling ahead of the noise")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(temps, color="purple")
axes[1].set_xlabel("step")
axes[1].set_ylabel("temperature  =  exp(log_temperature)")
axes[1].set_title("Learnable temperature — softmax sharpness self-tuning")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# %% [markdown]
# ### Viz 2 — the similarity matrix (the money shot)
#
# Each cell = `image_i · text_j` (temperature-scaled).
# A trained CLIP has the **diagonal lit up** and off-diagonals dark.
# This is literally the figure from the CLIP paper.

# %%
with torch.no_grad():
    images, texts = get_clip_batch("test")
    sim = clip.forward(images, texts)  # (10, 10), temperature-scaled

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(sim.numpy(), cmap="viridis")
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(CIFAR_CLASSES, fontsize=9)
ax.set_xlabel("text caption")
ax.set_ylabel("image")
ax.set_title("CLIP similarity matrix (test) — diagonal should glow brighter than its row & column")

vmax = sim.max().item()
for i in range(10):
    for j in range(10):
        v = sim[i, j].item()
        ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                color="white" if v < vmax * 0.55 else "black", fontsize=7)

plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()


# %% [markdown]
# ### Viz 3 — zero-shot classification (CLIP's superpower)
#
# CLIP can classify images WITHOUT a classifier head.
#   1. Encode all 10 captions once    -> (10, proj_dim)
#   2. Encode test images              -> (N, proj_dim)
#   3. argmax over `images @ texts.T`  picks the closest caption
#
# The captions ARE the classifier — no labelled training was done on test images.

# %%
NUM_EVAL = 2000

with torch.no_grad():
    # Text side: encode the 10 prototypical captions, L2-normalize.
    text_emb = clip.text_encoder.forward(CAPTION_IDS)
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    # Image side: pick a random chunk of test images, encode in mini-batches.
    idx = torch.randperm(len(X_test))[:NUM_EVAL]
    images_eval = X_test[idx]
    labels_eval = y_test[idx]

    BATCH = 100
    img_emb_chunks = []
    for i in range(0, NUM_EVAL, BATCH):
        e = clip.image_encoder.forward(images_eval[i:i + BATCH])
        e = e / e.norm(dim=-1, keepdim=True)
        img_emb_chunks.append(e)
    img_emb = torch.cat(img_emb_chunks, dim=0)  # (N, proj_dim)

    sim_zs = img_emb @ text_emb.T  # (N, 10)
    preds = sim_zs.argmax(dim=-1)
    acc = (preds == labels_eval).float().mean().item()

print(f"Zero-shot accuracy on {NUM_EVAL} test images: {acc * 100:.1f}%  (random = 10%)")


# %% [markdown]
# ### Viz 4 — per-class confusion (what does it mix up?)
#
# Row = true class, Col = predicted class. Diagonal = correct.
# Off-diagonal hotspots tell you WHICH pairs the model can't tell apart
# (often visually-similar: cat↔dog, car↔truck, deer↔horse).

# %%
confusion = torch.zeros(10, 10, dtype=torch.long)
for t, p in zip(labels_eval.tolist(), preds.tolist()):
    confusion[t, p] += 1

# Row-normalize so class imbalance doesn't bias colors.
conf_norm = confusion.float() / confusion.sum(dim=1, keepdim=True).clamp(min=1)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(conf_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(CIFAR_CLASSES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(CIFAR_CLASSES, fontsize=9)
ax.set_xlabel("predicted")
ax.set_ylabel("true")
ax.set_title(f"Zero-shot confusion (row-normalized) — diagonal acc = {acc*100:.1f}%")
for i in range(10):
    for j in range(10):
        v = conf_norm[i, j].item()
        if v > 0.05:
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=7)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()

# %%
