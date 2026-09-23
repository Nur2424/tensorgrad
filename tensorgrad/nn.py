"""
tensorgrad/nn.py
----------------
Neural network modules built on top of engine.py and functional.py

Every module is a thin Python class, no magic metaclasses or class-level registries

The only convention is:
  __init__ creates parameter Tensors as self.* attributes
  forward() does the computation
  __call__ dispatches to forward (via Module base class)

Architecture: pre-norm transformer (LayerNorm before each sublayer), GPT-2 convention throughout
"""

import numpy as np
from .engine import Tensor
from . import functional as F


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Module:
    """
    Base class for all network modules

    parameters() walks __dict__ and collects every Tensor (requires_grad=True)
    and every sub-Module, including lists/tuples of Modules
    """

    def parameters(self):
        params = []
        for val in self.__dict__.values():
            if isinstance(val, Tensor) and val.requires_grad:
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, (list, tuple)):
                for item in val:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params

    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def num_params(self):
        return sum(p.data.size for p in self.parameters())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Primitive layers
# ---------------------------------------------------------------------------

class Linear(Module):
    """
    y = x @ W + b    (W: in_features x out_features)
    Kaiming uniform initialization
    """

    def __init__(self, in_features, out_features, bias=True):
        k = 1.0 / in_features ** 0.5
        self.weight = Tensor(np.random.uniform(-k, k, (in_features, out_features)))
        self.bias   = Tensor(np.zeros(out_features)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """
    Lookup table, gather rows from a (num_embeddings, embedding_dim) weight matrix
    Backward scatters (accumulates) gradients back via np.add.at
    """

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor(np.random.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, idx):
        return F.embedding(self.weight, idx)


class LayerNorm(Module):
    """
    Normalize over the last dimension with learnable scale (gamma) and shift (beta)
    Uses the one-shot backward from functional.layer_norm
    """

    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = Tensor(np.ones(normalized_shape))
        self.bias   = Tensor(np.zeros(normalized_shape))
        self.eps    = eps

    def forward(self, x):
        return F.layer_norm(x, self.weight, self.bias, self.eps)


class FeedForward(Module):
    """
    Position-wise feed-forward network:
        FFN(x) = relu( x @ W1 + b1 ) @ W2 + b2
    Inner dimension = 4 * n_embd (standard GPT convention)
    """

    def __init__(self, n_embd):
        self.fc1 = Linear(n_embd, 4 * n_embd)
        self.fc2 = Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.fc2(self.fc1(x).relu())


# ---------------------------------------------------------------------------
# Multi-head causal self-attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(Module):
    """
    Causal multi-head self-attention (GPT-2 layout)

    Q, K, V are produced by a single combined linear (c_attn: n_embd => 3*n_embd),
    then split and reshaped into heads, an output projection (c_proj) follows

    The causal mask is a pre-computed boolean array stored as a plain numpy
    attribute (not a Tensor) so it does not appear in parameters()

    After each forward pass, the softmax attention weights are stored in
    self.attn_weights (shape: B, n_head, T, T) as a plain numpy array for
    inspection and debugging
    """

    def __init__(self, n_embd, n_head, block_size):
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.d_head = n_embd // n_head            # dimension per attention head

        self.c_attn = Linear(n_embd, 3 * n_embd)  # combined Q/K/V projection
        self.c_proj = Linear(n_embd, n_embd)       # output projection

        # mask[i, j] = True means token i is NOT allowed to attend to token j
        # k=1, block strictly-upper-triangular positions (future tokens)
        # The diagonal (i == j) is always False each token sees itself
        self.mask = np.triu(np.ones((block_size, block_size), dtype=bool), k=1)

        self.attn_weights = None                   # populated every forward pass

    def forward(self, x):
        B, T, C = x.data.shape                     # C = n_embd

        # ---- Q, K, V projections ----------------------------------------
        # c_attn maps (B, T, C) => (B, T, 3C), then we slice the three projections
        qkv = self.c_attn(x)
        q   = qkv[:, :, :C]                        # (B, T, C)
        k   = qkv[:, :, C:2*C]                     # (B, T, C)
        v   = qkv[:, :, 2*C:]                      # (B, T, C)

        # ---- Reshape into heads -----------------------------------------
        # (B, T, n_head, d_head) => transpose axes 1 and 2 => (B, n_head, T, d_head)
        q = q.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

        # ---- Scaled dot-product attention scores ------------------------
        # k.T swaps the last two axes, (B, n_head, d_head, T)
        # q @ k.T => (B, n_head, T, T)
        scale = self.d_head ** -0.5
        attn  = (q @ k.T) * scale

        # ---- Causal mask ------------------------------------------------
        # masked_fill sets future positions to -inf => softmax maps them to 0
        attn = attn.masked_fill(self.mask[:T, :T], float('-inf'))

        # ---- Softmax and weighted aggregation ---------------------------
        attn_sm = F.softmax(attn, dim=-1)          # (B, n_head, T, T)

        # Store for external inspection (plain numpy, not part of the graph)
        self.attn_weights = attn_sm.data.copy()

        y = attn_sm @ v                            # (B, n_head, T, d_head)

        # ---- Merge heads ------------------------------------------------
        # transpose(1, 2) => (B, T, n_head, d_head) => reshape => (B, T, C)
        y = y.transpose(1, 2).reshape(B, T, C)

        return self.c_proj(y)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class Block(Module):
    """
    One transformer block: pre-norm residual attention + pre-norm residual FFN

        x = x + Attention( LayerNorm(x) )
        x = x + FeedForward( LayerNorm(x) )
    """

    def __init__(self, n_embd, n_head, block_size):
        self.ln1  = LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.ln2  = LayerNorm(n_embd)
        self.ff   = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# GPTNano
# ---------------------------------------------------------------------------

class GPTNano(Module):
    """
    Character-level GPT token embedding + learned positional embedding +
    n_layer transformer blocks + final layer norm + language model head

    Default hyperparameters yield ~400K parameters on a 65-character vocabulary
    (Shakespeare):
        vocab_size=65, n_embd=128, n_head=4, n_layer=2, block_size=64

    Exact count:
        token_emb  :  65 * 128          = 8320
        pos_emb    :  64 * 128          = 8192
        2 x Block  :  2 * 198 272       = 396544
        ln_f       :  2 * 128           = 256
        head       : 128 * 65           = 8320
        Total                           = 421632
    """

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=2, block_size=64):
        self.block_size = block_size
        self.token_emb  = Embedding(vocab_size, n_embd)
        self.pos_emb    = Embedding(block_size, n_embd)
        self.blocks     = [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        self.ln_f       = LayerNorm(n_embd)
        self.head       = Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        """
        idx : integer ndarray (B, T) token indices
        Returns logits (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        tok_emb = self.token_emb(idx)              # (B, T, n_embd)
        pos_emb = self.pos_emb(np.arange(T))       # (T, n_embd) broadcast over B
        x       = tok_emb + pos_emb                # (B, T, n_embd)

        for block in self.blocks:
            x = block(x)

        x      = self.ln_f(x)                     # (B, T, n_embd)
        logits = self.head(x)                     # (B, T, vocab_size)
        return logits
