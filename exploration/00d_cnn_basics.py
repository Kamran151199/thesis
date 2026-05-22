# %% [markdown]
# # Day 0 (part D): CNNs — Where "Channel" Came From
#
# Before transformers ate the world, CNNs ruled vision for a decade.
# You need to understand them because:
#   1. The word "channel" (C in your tensors) comes from CNNs
#   2. Vision Transformers (ViT) still USE a conv layer at the very start
#   3. CLIP's "ResNet backbone" was a CNN before they switched to ViT
#
# Core idea in one sentence:
#   A CNN slides little "detectors" (kernels) across an image.
#   Each detector looks at a small neighborhood, reports a number.
#   Stack many detectors → many "channels" of feature maps.

# %%
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

# %% [markdown]
# ## 1. A convolution is just a sliding kernel
#
# Imagine a small 3×3 stamp. You slide it across the image.
# At each position, you multiply the stamp's 9 numbers by the 9 pixels
# under it, then sum them up. That single number = the output at that position.
#
# Slide the stamp to every position → get a new image (the "feature map").

# %%
# Make a simple 10×10 image with a vertical line in it
image = np.zeros((10, 10))
image[:, 5] = 1.0  # vertical line at column 5

# A "vertical edge detector" kernel — fires when pixel on left is dark, right is bright
edge_kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
], dtype=np.float32)

# Manual 2D convolution — slide the kernel over the image
def conv2d_manual(img, kernel):
    # H = 10, W = 10,
    H, W = img.shape
    
    # kH = 3, kW = 3
    kH, kW = kernel.shape

    # → output shape will be (8, 8)
    out = np.zeros((H - kH + 1, W - kW + 1))
    
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            # Pluck the 3×3 patch under the kernel
            patch = img[i:i+kH, j:j+kW]
            # Multiply element-wise and sum — ONE output number
            out[i, j] = (patch * kernel).sum()
    return out

feature_map = conv2d_manual(image, edge_kernel)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Input image (10×10)\nvertical line at col 5")

axes[1].imshow(edge_kernel, cmap='RdBu', vmin=-1, vmax=1)
axes[1].set_title("Kernel (3×3)\n'vertical edge detector'")
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{edge_kernel[i,j]:.0f}", ha='center', va='center', fontsize=12)

axes[2].imshow(feature_map, cmap='viridis')
axes[2].set_title(f"Feature map {feature_map.shape}\n(kernel slid across image)")
plt.tight_layout()
plt.show()

print("Where the feature map LIGHTS UP, the kernel 'matched' — it found an edge.")
print("The kernel only LIGHTS UP near col 5 where the vertical line is. Beautiful.")

# %% [markdown]
# ## 2. Multiple kernels = multiple CHANNELS
#
# One kernel detects one thing (vertical edges).
# If you want to also detect horizontal edges, diagonals, corners, etc.,
# you use MULTIPLE kernels in parallel. Each produces its own feature map.
#
# Stack them → you have a multi-channel output.
# THIS IS WHERE "CHANNELS" IN ML COMES FROM.

# %%
# Three different detectors
kernels = {
    "Vertical edge": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "Horizontal edge": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "Diagonal (\\)": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
}

# An image with a cross (vertical + horizontal lines)
cross = np.zeros((10, 10))
cross[:, 5] = 1.0
cross[5, :] = 1.0

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

axes[0, 0].imshow(cross, cmap='gray')
axes[0, 0].set_title("Input: cross shape")

for idx, (name, k) in enumerate(kernels.items(), start=1):
    axes[0, idx].imshow(k, cmap='RdBu', vmin=-1, vmax=1)
    axes[0, idx].set_title(f"Kernel: {name}")
    for i in range(3):
        for j in range(3):
            axes[0, idx].text(j, i, f"{k[i,j]:.0f}", ha='center', va='center', fontsize=11)

# Apply each kernel → get 3 feature maps (3 channels of output)
feature_maps = [conv2d_manual(cross, k) for k in kernels.values()]
axes[1, 0].axis('off')
axes[1, 0].text(0.5, 0.5, "3 channels of output\n= 3 different 'views'\nof the same image",
                ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)

for idx, (name, fmap) in enumerate(zip(kernels.keys(), feature_maps), start=1):
    axes[1, idx].imshow(fmap, cmap='viridis')
    axes[1, idx].set_title(f"Channel {idx}: '{name}' feature map")

plt.suptitle("MULTIPLE KERNELS = MULTIPLE CHANNELS\n"
             "Each channel = one 'detector' reporting at every spatial location",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print("The output has 3 CHANNELS. Shape would be: (3, H_out, W_out)")
print("In your thesis tensors: the C dimension IS this.")
print("In a real CNN, there are 64 or 128 or 512 channels per layer.")

# %% [markdown]
# ## 3. PyTorch does all this in one line
#
# nn.Conv2d(in_channels, out_channels, kernel_size)
#   in_channels  = how many channels the input has (RGB image = 3)
#   out_channels = how many detectors you want (= channels in output)
#   kernel_size  = the "stamp size" (3×3 is typical)

# %%
# Simulate: 1 image, 3 input channels (RGB), 64 output channels (detectors)
fake_image = torch.randn(1, 3, 32, 32)   # (B=1, C_in=3, H=32, W=32)

conv_layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
output = conv_layer(fake_image)

print(f"Input shape:  {fake_image.shape}  ← 1 RGB image")
print(f"Output shape: {output.shape}  ← now has 64 CHANNELS = 64 feature detectors")
print(f"\nLearnable parameters: {sum(p.numel() for p in conv_layer.parameters())}")
print("  = 3 input channels × 64 output channels × 3 × 3 kernel = 1728")
print("  + 64 biases = 1792 total")
print("\nEach of those 64 channels learned to detect something different during training.")

# %% [markdown]
# ## 4. The deep stack: each layer builds on the last
#
# Layer 1 detects edges and colors.
# Layer 2 combines those → simple textures.
# Layer 3 → object parts (eyes, wheels, noses).
# Layer 4 → whole objects (faces, cars).
#
# Visual intuition: each layer "sees" a larger region of the input
# because it looks at features that themselves looked at neighborhoods.
# This growing "receptive field" is how CNNs understand images.

# %%
# A tiny 3-layer CNN to illustrate
tiny_cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),  nn.ReLU(),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
)

x = torch.randn(1, 3, 32, 32)
print("Input:", x.shape, "← RGB image")

for i, layer in enumerate(tiny_cnn):
    x = layer(x)
    if isinstance(layer, nn.Conv2d):
        print(f"After Conv layer {i}: {x.shape}  ← more channels, same spatial size")

# %% [markdown]
# ## 5. CNN limitations (why transformers eventually took over for vision)
#
# 1. **Locality:** A 3×3 kernel only sees 3×3 neighborhood at a time.
#    To see the whole image, you need MANY layers (growing receptive field).
#    Global reasoning is slow.
#
# 2. **Translation equivariance:** A CNN treats "dog at top-left" and
#    "dog at bottom-right" similarly. Good for some tasks, but it means
#    the model doesn't directly encode "where" things are relative to each other.
#
# 3. **Hand-designed inductive bias:** "Nearby pixels matter more" is BAKED IN.
#    Great for natural images. But for documents with structured layouts,
#    or charts with arbitrary positions, this bias can hurt.
#
# ViT (Vision Transformer) removed these biases:
#   - Splits image into patches
#   - Treats each patch as a token
#   - Uses attention → any patch can interact with any other patch instantly
#   - No "nearby" assumption
#
# Why your thesis uses ViT-based VLMs (CLIP, LLaVA, Qwen2-VL):
#   Documents and charts have LONG-RANGE structure (header to footer,
#   legend to data points). CNNs struggle with this. ViTs nail it.

# %% [markdown]
# ## Summary
#
# - **Kernel** = small pattern detector that slides over input
# - **Channel** = output of ONE kernel = ONE feature map
# - **Multiple channels** = multiple parallel detectors
# - **C in your tensors** comes directly from this
# - CNNs see LOCAL patterns well, GLOBAL patterns poorly → why ViT replaced them for VLMs

# %%
