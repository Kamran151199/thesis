# %% [markdown]
# # The Journey to Matrix Duality
#
# *(A synthesis of the Q&A that built this mental model — from the
#   "hack, I think I got it" moment through every refinement and
#   side-question, ending with the unified picture.)*
#
# ---
#
# ## Where this started
#
# You'd been pushing on the question: *is there a fundamental LINK
# between the two readings of matrix multiplication?* — the
# "transformation" reading (where do basis vectors land) and the
# "dot product" reading (each entry is a row · column).
#
# Then you wrote this, and it was almost exactly right:
#
# > *"So, the matrix is not only showing the landing zone of the input
# > space's basis vectors, but it's also saying how much importance each
# > basis vector of the input space is carrying to each dimension of the
# > output space. We are doing the dot product with the input vector and
# > the 'imaginary vector' (which happens to be the row vector) of a
# > matrix that carries the info about the importance of each input
# > space's dim for the single dimension of the output space."*
#
# That sentence contained the whole picture, with three small things to
# refine. Let's lock them in.

# %% [markdown]
# ## 1. The refined core insight
#
# A matrix `W` of shape `(out_dim, in_dim)` simultaneously stores **two
# dual pieces of information**:
#
# 1. **Columns: landings.** Each column is where one input basis vector
#    ends up in the output space.
# 2. **Rows: signed contributions per output dimension.** Each row says,
#    for one specific output coordinate, *how much each input dimension
#    contributes to it* — and that contribution can be positive
#    (boosts) or negative (subtracts).
#
# Each output coordinate is computed by dot-producting the input vector
# with one row:
#
# ```
#   output[i]  =  dot(row_i, input)
#              =  W[i, 0]·input[0]  +  W[i, 1]·input[1]  +  W[i, 2]·input[2] + ...
# ```
#
# ### Three small refinements to your original sentence
#
# - **Rows aren't "imaginary."** They are *real* vectors. You can draw
#   them as arrows. They just happen to live in the *input* space (along
#   with the input vector itself), playing the role of probes — not of
#   "input vectors."
#
# - **"Importance" is *signed*.** A row entry can be **negative**,
#   meaning that input dim *hurts* that output dim. So a better word is
#   "signed contribution" or "weight." (E.g., in a cat/dog classifier,
#   the cat-row might have `-1.0` on bark_volume — barking hurts the
#   cat-score.)
#
# - **"Kinda similar to the dot product" undersells it.** It *is* the
#   dot product. Each output coordinate is born from exactly one dot
#   product: `output[i] = dot(row_i, input)`.

# %% [markdown]
# ## 2. Dot product = projection × magnitude  (the next layer of nuance)
#
# You asked: *"but dot product is a projection, no?"* — close, with one
# careful nuance.
#
# The full geometric formula:
#
# ```
#   dot(a, b)  =  |a| · |b| · cos(θ)
# ```
#
# Read it two ways:
#
# ```
#   = |b| · (|a| · cos θ)    →  "magnitude of b" × "projection of a onto b's direction"
#   = |a| · (|b| · cos θ)    →  "magnitude of a" × "projection of b onto a's direction"
# ```
#
# So `dot(row, input)` is **NOT pure projection**. It's:
#
# ```
#   "projection of input onto row's DIRECTION"   ×   "MAGNITUDE of row"
#                  ↑                                          ↑
#         angle / alignment info                  "loudness" / gain
# ```
#
# ### What the row's direction vs magnitude do for a neuron
#
# - **Direction of row** = the pattern the neuron looks for (alignment).
# - **Magnitude of row** = how loudly that neuron speaks when its
#   pattern matches.
#
# Two neurons with the same direction but different magnitudes fire on
# the same inputs — they just produce different output sizes. Training
# learns both the direction AND the magnitude of each row.
#
# ### When dot product IS pure projection
#
# - If you normalize the row to unit length: `dot(row_unit, input)` =
#   pure projection of input onto the row's direction.
# - **Cosine similarity** = `dot(a, b) / (|a|·|b|)` — both normalized,
#   so only the angle survives. Pure direction match.
#
# In ML:
#
# - Standard `Linear` layers: magnitudes matter (loudness baked in).
# - Cosine-similarity attention / contrastive losses (e.g., CLIP):
#   both sides normalized → pure direction match.

# %% [markdown]
# ## 3. The bridge: matrix as linear transformation = stack of project-and-scale
#
# Next you asked: *"if dot product is projection × magnitude, then what
# does that have to do with the matrix being a linear transformation?"*
#
# Here's the connective tissue:
#
# > **A linear transformation IS just a stack of "project-and-scale"
# > operations.** Each output coordinate is one such operation. The whole
# > transformation is many of them done in parallel on the same input.
#
# Using `A = [[3, 2], [1, 5]]` and `v = [1, 2]`:
#
# ```
#   out[0]  =  dot(row 0, v)  =  |v| · |row 0| · cos(angle between v and row 0)
#           =  "how much of v lies along row 0's direction" × "|row 0|"
#           =  7
#
#   out[1]  =  dot(row 1, v)  =  |v| · |row 1| · cos(angle between v and row 1)
#           =  "how much of v lies along row 1's direction" × "|row 1|"
#           =  11
#
#   output  =  [7, 11]   ← just the two readings stacked
# ```
#
# Visually:
#
# ```
#         v (input)
#             │
#             ├──→  projected onto row 0 direction → scaled by |row 0|  →  out[0]
#             │
#             └──→  projected onto row 1 direction → scaled by |row 1|  →  out[1]
#
#                           ↓
#                      output = [out[0], out[1]]
# ```
#
# ### Reconciling with "transformation warps space"
#
# Two equivalent stories about what the transformation does:
#
# | Lens | What it emphasizes |
# |---|---|
# | **Column view** | Where the input basis vectors **land** in output space → describes the warp geometrically. |
# | **Row view (project + scale)** | What aspects of the input get **measured** by each output coordinate → describes the warp coordinate-by-coordinate. |
#
# Both produce the same output. The "warp" IS the multi-angle
# measurement. No contradiction — they're two descriptions of the same
# operation.

# %% [markdown]
# ## 4. Three small questions that came up
#
# ### Q: What's "warp"?
#
# Just an informal word for "the geometric distortion the transformation
# does to space" — stretching, rotating, shearing, squashing. Not a
# technical term. Could have just said "transformation."
#
# ### Q: Is a row vector called a co-vector / dual vector?
#
# **Yes.** Formal names:
#
# - **Vector**: lives in `V` (the input space). Just an arrow.
# - **Co-vector** / **dual vector** / **linear functional**: lives in
#   `V*`. It's a *function* that takes a vector in `V` and returns a
#   scalar.
# - Rows of a matrix `V → W` are elements of `V*`. Each row eats a
#   vector in `V` and spits out one scalar (one output coordinate).
#
# More on this in section 5.
#
# ### Q: If we want the output, why do projections happen in INPUT space?
#
# This was the conceptual unlock. The fix:
#
# > **The projections HAPPEN in input space. The RESULTS (numbers) become
# > the coordinates of the output vector, which lives in output space.**
#
# We don't "display the input vector in output space." We **transform** it:
#
# ```
#   Step 1: take input vector v (lives in input space)
#   Step 2: project v onto row 0 (a probe in input space) × |row 0|  →  one scalar
#   Step 3: project v onto row 1 (a probe in input space) × |row 1|  →  one scalar
#   ...
#   Step N: stack all those scalars → that stack IS the output vector (in output space)
# ```
#
# **Analogy:** measuring an object with multiple rulers.
#
# ```
#    physical object (in 3D room space)
#          │
#          ├──→  ruler 1 measures height      →   17 cm    \
#          └──→  ruler 2 measures width       →   42 cm     ─→  [17, 42]  (output vector)
#                                                           /
#    Measurements happen IN the room (input space).
#    The result [17, 42] is a NEW vector in "measurement space" (output space).
# ```
#
# Input space = the **lab** where measurements occur.
# Output space = the **report** where readings are collected.

# %% [markdown]
# ## 5. The dual space (V* ≅ V) — why we can draw rows as arrows
#
# You asked: *"what do you mean V* ≅ V in `Rⁿ` with the standard dot product?"*
#
# ### The formal picture (no inner product)
#
# A co-vector is **abstractly a function** `f: V → R`. It eats a vector
# and returns a number. NOT a vector. NOT an arrow. Co-vectors form
# their own space `V*`. Different beasts from elements of `V`.
#
# ### The R^n-with-dot-product magic (Riesz representation)
#
# In `Rⁿ` with the standard dot product, every linear function
# `f: Rⁿ → R` can be **uniquely written** as:
#
# ```
#   f(v)  =  a · v       for some unique vector a ∈ Rⁿ
# ```
#
# So every co-vector `f` has a **unique vector twin `a`** in the same
# `Rⁿ`. Concretely:
#
# ```
#   The row [3, 2] is formally the function f(v) = 3·v[0] + 2·v[1].
#   It is ALSO the vector [3, 2] you can draw as an arrow.
#   These are the SAME object because dot([3, 2], v) = 3·v[0] + 2·v[1].
# ```
#
# Function evaluation = dot product. The "function" and "the vector that
# represents the function" are collapsed into one drawable object.
#
# ### Why this matters for our drawings
#
# Without an inner product, co-vectors would live in a parallel universe
# and you couldn't draw them alongside input vectors. **The dot product
# is the bridge that lets us identify them and draw rows as arrows in
# the same input-space picture.**

# %% [markdown]
# ## 6. Matrix as a stack of polynomials
#
# You connected: *"then the matrix is a stack of polynomials stacked
# row by row?"* — **yes, sharp connection.**
#
# Each row of `W` is a polynomial — specifically a **homogeneous linear
# polynomial** (also called a **linear form**):
#
# ```
#   row i of W  =  [w_i0, w_i1, w_i2, ...]
#
#   p_i(x)  =  w_i0 · x_0  +  w_i1 · x_1  +  w_i2 · x_2  +  ...
# ```
#
# Every term has **degree 1**, and there's **no constant term**.
#
# A matrix `W` is then a **stack of N linear polynomials**, one per
# output coordinate:
#
# ```
#   W @ v  =  [ p_0(v),
#               p_1(v),
#               p_2(v),
#                 ⋮     ]   ← each coord = value of one row's polynomial at v
# ```
#
# Concretely with `W = [[3, 2], [1, 5]]` and `v = [1, 2]`:
#
# ```
#   row 0 → polynomial  p_0(x) = 3x_0 + 2x_1     p_0([1, 2]) = 3 + 4 = 7
#   row 1 → polynomial  p_1(x) = 1x_0 + 5x_1     p_1([1, 2]) = 1 + 10 = 11
#
#   output = [7, 11]  ← polynomial evaluations stacked
# ```
#
# **Output space = the space where the stacked-but-independent
# polynomial evaluations live.** Each output coordinate is one
# polynomial's answer; they don't talk to each other, they just run in
# parallel and the results get organized as a vector.
#
# ### What changes with bias
#
# `y = W @ x + b` makes each row become an **affine polynomial** — degree
# 1 plus a constant:
#
# ```
#   p_i(x)  =  w_i0·x_0 + w_i1·x_1 + ... + b_i
# ```
#
# Still polynomial, just no longer "homogeneous."
#
# ### Higher-degree polynomials need TENSORS (math sense)
#
# A matrix only expresses **degree-1** polynomials. For quadratic
# (degree 2) or cubic (degree 3) polynomials, you'd need a higher-rank
# **tensor** (math sense — a multi-dimensional array of coefficients):
#
# ```
#   degree 2 (quadratic):  w_ij · x_i · x_j + ...   ← needs a 3-index tensor
#   degree 3 (cubic):      w_ijk · x_i · x_j · x_k  ← needs a 4-index tensor
# ```
#
# In neural networks, instead of using higher-rank tensors, we get
# expressivity from **stacking linear layers with nonlinearities in
# between** (ReLU, GELU, etc.). This is more parameter-efficient and is
# what powers the universal approximation theorem.
#
# **Note on "tensor"**: in math it specifically means a multi-rank
# object with specific transformation rules. In PyTorch land, "tensor"
# just means *any* multi-dimensional array regardless of math role —
# the word was borrowed but loosened.

# %% [markdown]
# ## 7. The ML payoffs
#
# ### What a Linear layer's weights actually learn
#
# `nn.Linear(in=5, out=3)` has `W.shape = (3, 5) = (out, in)`. That's
# 15 weights total. Each `W[i, j]` represents:
#
# > *"How much should input dim j contribute (signed) to output dim i?"*
#
# Training (gradient descent) adjusts these 15 numbers so the resulting
# polynomials best produce the labels you want from the inputs.
#
# Two honest caveats:
#
# 1. **Signed contribution, not pure "importance."** Weights can be
#    negative — meaning input dim j *hurts* output dim i.
#
# 2. **They're learned, not hand-designed.** You don't tell the network
#    *"whiskers matter for cats"* — the gradient updates figure out the
#    15 numbers by minimizing loss on training data. The semantic
#    interpretations ("this row detects cats") emerge from the data.
#
# ### Why `x @ W.T` instead of `W @ x`?
#
# Pure **shape-alignment trick** for batched inputs, no semantic change.
#
# PyTorch stores `W` with shape `(out, in)` because that's the natural
# "one row per neuron" layout. But inputs come in batches:
# `x.shape = (B, in)`.
#
# ```
#   W @ x        →  (out, in) @ (B, in)         shape mismatch ✗
#   x @ W.T      →  (B, in)  @ (in, out)  =  (B, out)   ✓ batch stays at axis 0
# ```
#
# `(x @ W.T)[b, i] = dot(W[i], x[b])` — every output coordinate is still
# **"row i of W dotted with the input"**, the original semantic. The
# transpose just realigns axes for batching.
#
# ### Each neuron learns one feature of the output space
#
# - **Final classifier layer**: the meaning is predefined by the task.
#   Neuron 0 = "cat_score," neuron 1 = "dog_score," etc.
# - **Hidden layer**: the meaning **emerges from training**. The
#   network learns what each hidden neuron should detect via gradient
#   descent. You don't go in with *"neuron 7 will detect ears"* — you go
#   in with random rows, and after training, neuron 7 *happens* to fire
#   on ear-like inputs because that turned out useful for prediction.
#
# **Interpretability** is the reverse-engineering task of figuring out
# what the trained rows ended up detecting. Done **after** training,
# not designed before.

# %% [markdown]
# ## 8. The unified mental model (the picture you now own)
#
# Putting it all together — a single matrix `W` of shape `(out, in)`:
#
# ```
#   ╔═══════════════════════════════════════════════════════════════════╗
#   ║                                                                   ║
#   ║   W is ONE object that simultaneously:                            ║
#   ║                                                                   ║
#   ║     - Has COLUMNS living in OUTPUT space:                         ║
#   ║         each column = where one input basis vector lands.         ║
#   ║                                                                   ║
#   ║     - Has ROWS living in INPUT space:                             ║
#   ║         each row = a polynomial / probe / co-vector / neuron.     ║
#   ║                                                                   ║
#   ║     - Computes output[i] = dot(row_i, input)                      ║
#   ║                          = projection of input onto row_i's dir   ║
#   ║                            × |row_i|.                             ║
#   ║                                                                   ║
#   ║     - Entry W[i, j] does DOUBLE DUTY:                             ║
#   ║         column reading: how much of input dim j does ĵ contribute ║
#   ║                         to output dim i?                          ║
#   ║         row reading:    how much weight does the output-i probe   ║
#   ║                         put on input dim j?                       ║
#   ║         SAME number. SAME answer. Two questions.                  ║
#   ║                                                                   ║
#   ╚═══════════════════════════════════════════════════════════════════╝
# ```
#
# ### Equivalent ways to read what `y = W @ v` does
#
# 1. *"`v`'s coordinates weight a linear combination of W's columns."*
#    (column reading — geometric warp)
# 2. *"Each output coordinate = one of W's row-polynomials evaluated at `v`."*
#    (row reading — measurement)
# 3. *"Each output coordinate = projection of v onto a row's direction,
#    scaled by the row's magnitude."*
#    (row reading, geometric form)
# 4. *"`y` is the multi-angle readout of `v` through each of W's rows."*
#    (row reading, intuitive)
#
# All four describe the same arithmetic.

# %% [markdown]
# ## 9. Hey-dude summary
#
# > Hey dude, a matrix is **one map between two spaces** that simultaneously
# > tells you (a) where the input basis vectors land in the output space
# > (the *columns*) and (b) the signed-weight recipes that produce each
# > output coordinate via a dot product with the input (the *rows*).
# > Rows are **real, drawable vectors that live in the input space** —
# > they're called co-vectors / dual vectors, but the standard dot
# > product lets us identify them with regular input-space arrows. Each
# > row is a **linear polynomial / probe / neuron** whose job is to
# > read one number out of the input: the projection of the input onto
# > the row's direction, scaled by the row's magnitude. The whole
# > transformation is just **N such polynomials running in parallel**
# > on the same input, stacking their answers into a new vector in
# > output space.
# >
# > In ML, a `Linear` layer is exactly this: `W.shape = (out, in)` stores
# > `out × in` learnable signed contributions. Each row is one neuron,
# > each output coordinate is one polynomial's value, each weight is one
# > "how much does input dim j contribute to output dim i?" learned by
# > gradient descent. We use `x @ W.T` instead of `W @ x` only to keep
# > the batch axis on axis 0 — same numbers, same semantics, just
# > better aligned with batched inputs. Neurons in the final layer have
# > predefined meanings (cat, dog, etc.); neurons in hidden layers have
# > meanings that **emerge from training**, which is why interpretability
# > is a reverse-engineering job done after the fact.
# >
# > Stack many such layers with nonlinearities between them and you get
# > a neural network: a deep composition of polynomial-stacks bent at
# > every layer by ReLU/GELU/etc., capable of approximating any
# > continuous function. **All of it is built on this single duality:
# > rows in input space, columns in output space, every entry doing
# > double duty.**

# %% [markdown]
# ## 10. The mental checklist to never forget
#
# When you next see a weight matrix, run this checklist:
#
# 1. **What's its shape `(out, in)`?** Which is the output dim, which
#    is the input dim?
# 2. **Where do the columns live?** Output space. They are landings of
#    input basis vectors.
# 3. **Where do the rows live?** Input space. They are probes / neurons /
#    co-vectors / polynomials.
# 4. **For each row, what's its direction and magnitude?** The direction
#    is the pattern it looks for; the magnitude is its loudness.
# 5. **What does an entry `W[i, j]` mean?**
#    - As part of column j: how much input basis vector j contributes
#      to output dim i.
#    - As part of row i: how much weight neuron i puts on input dim j.
#    - **Same number, double duty.**
# 6. **What's `W @ x`?** N polynomials evaluated in parallel at x,
#    results stacked into output vector.
# 7. **What does `W.T` do?** Swaps which space owns which arrows. Now
#    the old rows live in the old output's *dual*, and vice versa.
#
# Do this on the next 20 weight matrices you see — `nn.Linear`,
# attention `Q`, `K`, `V`, embedding tables, LoRA `A`/`B`, projection
# heads — and this picture becomes permanent.

# %%
