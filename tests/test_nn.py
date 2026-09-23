"""
tests/test_nn.py
----------------
Tests for every module in nn.py

Structure
---------
1. Attention pattern check: verify shape causal structure, and softmax normalization before any grad checks
2. Gradient checks for each module using central-difference numerical gradients
3. Structural checks: parameter count, output shapes, residual connections

The numpy reference functions are written inline so each test is self-contained
All grad checks use small dimensions to keep the finite-difference sweeps fast
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorgrad import Tensor
from tensorgrad.nn import (
    Linear, Embedding, LayerNorm, FeedForward,
    MultiHeadAttention, Block, GPTNano,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def numerical_grad(f, x, eps=1e-5):
    """Central-difference gradient of scalar function f at array x"""
    x    = x.astype(np.float64)
    grad = np.zeros_like(x)
    it   = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx  = it.multi_index
        orig = float(x[idx])
        x[idx] = orig + eps;  f_plus  = float(f(x))
        x[idx] = orig - eps;  f_minus = float(f(x))
        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def check(tensor, numerical, atol=1e-4, name=''):
    max_err = np.abs(tensor.grad - numerical).max()
    assert np.allclose(tensor.grad, numerical, atol=atol), (
        f"{name} max error: {max_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 1. MultiHeadAttention: isolation test BEFORE any grad checks
# ---------------------------------------------------------------------------

def test_mha_attention_pattern():
    """
    Verify the causal attention pattern in complete isolation
    This test must pass before we trust any gradient computation

    Checks:
      Output shape is (B, n_head, T, T)
      Every row sums to 1.0 (valid probability distribution)
      Every element in the strict upper triangle is 0 (no attending to future)
      Token 0 attends only to itself (single valid key)
    """
    np.random.seed(0)
    n_embd, n_head, block_size = 8, 2, 16
    B, T = 2, 6

    mha = MultiHeadAttention(n_embd, n_head, block_size)
    x   = Tensor(np.random.randn(B, T, n_embd))
    _   = mha(x)

    attn = mha.attn_weights    # (B, n_head, T, T) numpy array

    # Shape
    assert attn.shape == (B, n_head, T, T), f"Wrong shape: {attn.shape}"

    # Every row is a valid probability distribution
    row_sums = attn.sum(axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f"Row sums not 1: max deviation {np.abs(row_sums - 1).max():.2e}"
    )

    # Strict upper triangle must be zero (future positions masked out)
    upper_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    upper_vals = attn[..., upper_mask]
    assert np.allclose(upper_vals, 0.0, atol=1e-8), (
        f"Upper triangle not zero: max {np.abs(upper_vals).max():.2e}"
    )

    # Token 0 can only attend to itself: attn[:, :, 0, 0] == 1
    assert np.allclose(attn[:, :, 0, 0], 1.0, atol=1e-6), (
        f"Token 0 self-attention not 1: {attn[:, :, 0, 0]}"
    )


def test_mha_shorter_than_block():
    """Attention still works when T < block_size (mask sliced to [:T, :T])."""
    np.random.seed(1)
    n_embd, n_head, block_size = 8, 2, 32
    B, T = 1, 3   # T much shorter than block_size

    mha = MultiHeadAttention(n_embd, n_head, block_size)
    out = mha(Tensor(np.random.randn(B, T, n_embd)))

    assert out.data.shape == (B, T, n_embd)
    attn = mha.attn_weights
    assert attn.shape == (B, n_head, T, T)
    assert np.allclose(attn.sum(axis=-1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 2. Linear
# ---------------------------------------------------------------------------

def test_linear_shape():
    np.random.seed(2)
    lin = Linear(6, 10)
    x   = Tensor(np.random.randn(4, 6))
    out = lin(x)
    assert out.data.shape == (4, 10)


def test_linear_no_bias_shape():
    lin = Linear(6, 10, bias=False)
    assert lin.bias is None
    out = lin(Tensor(np.random.randn(3, 6)))
    assert out.data.shape == (3, 10)


def test_linear_grad():
    """Grad check: x, weight, bias for a batched linear layer"""
    np.random.seed(3)
    x_d = np.random.randn(5, 8)
    lin  = Linear(8, 12)
    w_d  = lin.weight.data.copy()
    b_d  = lin.bias.data.copy()

    X = Tensor(x_d.copy())
    lin.weight = Tensor(w_d.copy())
    lin.bias   = Tensor(b_d.copy())
    lin(X).sum().backward()

    check(X,          numerical_grad(lambda x: (x @ w_d + b_d).sum(), x_d.copy()), name='linear x')
    check(lin.weight, numerical_grad(lambda w: (x_d @ w + b_d).sum(), w_d.copy()), name='linear W')
    check(lin.bias,   numerical_grad(lambda b: (x_d @ w_d + b).sum(), b_d.copy()), name='linear b')


# ---------------------------------------------------------------------------
# 3. Embedding module
# ---------------------------------------------------------------------------

def test_embedding_shape():
    emb = Embedding(20, 8)
    idx = np.array([[0, 3, 7], [2, 5, 1]])   # (B=2, T=3)
    out = emb(idx)
    assert out.data.shape == (2, 3, 8)


def test_embedding_grad():
    np.random.seed(4)
    V, D = 10, 6
    emb  = Embedding(V, D)
    w_d  = emb.weight.data.copy()
    idx  = np.array([0, 3, 0, 5])   # row 0 repeated tests accumulation

    emb.weight = Tensor(w_d.copy())
    emb(idx).sum().backward()

    def f(w): return w[idx].sum()
    check(emb.weight, numerical_grad(f, w_d.copy()), name='embedding W')


# ---------------------------------------------------------------------------
# 4. LayerNorm module
# ---------------------------------------------------------------------------

def test_layernorm_shape():
    ln  = LayerNorm(8)
    out = ln(Tensor(np.random.randn(3, 5, 8)))
    assert out.data.shape == (3, 5, 8)


def test_layernorm_grad():
    """LayerNorm module wraps functional.layer_norm, spot-check here"""
    np.random.seed(5)
    N, D = 4, 8
    x_d  = np.random.randn(N, D)
    ln   = LayerNorm(D)
    w_d  = ln.weight.data.copy()
    b_d  = ln.bias.data.copy()

    X = Tensor(x_d.copy())
    ln.weight = Tensor(w_d.copy())
    ln.bias   = Tensor(b_d.copy())
    ln(X).sum().backward()

    def _ref(x, w, b):
        mu   = x.mean(-1, keepdims=True)
        var  = ((x - mu) ** 2).mean(-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + 1e-5)
        return (w * xhat + b).sum()

    check(X,         numerical_grad(lambda x: _ref(x, w_d, b_d), x_d.copy()), name='ln x')
    check(ln.weight, numerical_grad(lambda w: _ref(x_d, w, b_d), w_d.copy()), name='ln W')
    check(ln.bias,   numerical_grad(lambda b: _ref(x_d, w_d, b), b_d.copy()), name='ln b')


# ---------------------------------------------------------------------------
# 5. FeedForward
# ---------------------------------------------------------------------------

def test_feedforward_shape():
    ff  = FeedForward(8)
    out = ff(Tensor(np.random.randn(2, 5, 8)))
    assert out.data.shape == (2, 5, 8), "FFN must preserve the embedding dimension"


def test_feedforward_grad():
    """Grad check for input x through the two-layer MLP"""
    np.random.seed(6)
    N, D = 3, 8
    ff   = FeedForward(D)
    x_d  = np.random.randn(N, D)

    # Fix weights for the numerical reference
    w1 = ff.fc1.weight.data.copy()
    b1 = ff.fc1.bias.data.copy()
    w2 = ff.fc2.weight.data.copy()
    b2 = ff.fc2.bias.data.copy()

    X = Tensor(x_d.copy())
    ff.zero_grad()
    ff(X).sum().backward()

    def f(x):
        h = np.maximum(0, x @ w1 + b1)
        return (h @ w2 + b2).sum()

    check(X, numerical_grad(f, x_d.copy()), name='ffn x')


# ---------------------------------------------------------------------------
# 6. MultiHeadAttention: gradient checks (AFTER pattern test passes)
# ---------------------------------------------------------------------------

def _mha_np(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b, mask, n_head):
    """Pure numpy MHA forward for use inside numerical_grad"""
    B, T, C = x.shape
    d = C // n_head

    qkv = x @ c_attn_w + c_attn_b
    q   = qkv[..., :C  ].reshape(B, T, n_head, d).transpose(0, 2, 1, 3)
    k   = qkv[..., C:2*C].reshape(B, T, n_head, d).transpose(0, 2, 1, 3)
    v   = qkv[..., 2*C:  ].reshape(B, T, n_head, d).transpose(0, 2, 1, 3)

    att = (q @ k.swapaxes(-2, -1)) * (d ** -0.5)   # (B, n_head, T, T)

    # Apply causal mask (broadcast over B and n_head)
    m = mask[:T, :T]
    att = att.copy()
    att[..., m] = -np.inf

    # Numerically stable softmax
    att = att - att.max(-1, keepdims=True)
    exp_att = np.exp(att)
    sm = exp_att / exp_att.sum(-1, keepdims=True)

    y = (sm @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
    return (y @ c_proj_w + c_proj_b).sum()


def test_mha_grad_input():
    """Numerical gradient check for the input tensor x through MHA"""
    np.random.seed(7)
    n_embd, n_head, block_size = 8, 2, 16
    B, T = 1, 4

    mha = MultiHeadAttention(n_embd, n_head, block_size)
    x_d = np.random.randn(B, T, n_embd)

    # Analytical gradient
    X = Tensor(x_d.copy())
    mha.zero_grad()
    mha(X).sum().backward()
    analytical = X.grad.copy()

    # Numerical gradient (weights held fixed)
    caw = mha.c_attn.weight.data
    cab = mha.c_attn.bias.data
    cpw = mha.c_proj.weight.data
    cpb = mha.c_proj.bias.data
    mask = mha.mask

    num = numerical_grad(
        lambda x: _mha_np(x, caw, cab, cpw, cpb, mask, n_head),
        x_d.copy()
    )
    assert np.allclose(analytical, num, atol=1e-4), (
        f"MHA input grad max error: {np.abs(analytical - num).max():.2e}"
    )


def test_mha_grad_c_attn_weight():
    """Numerical gradient check for the c_attn weight matrix"""
    np.random.seed(8)
    n_embd, n_head, block_size = 8, 2, 16
    B, T = 1, 4

    mha = MultiHeadAttention(n_embd, n_head, block_size)
    x_d = np.random.randn(B, T, n_embd)

    mha.zero_grad()
    mha(Tensor(x_d.copy())).sum().backward()

    caw_d = mha.c_attn.weight.data.copy()
    cab   = mha.c_attn.bias.data
    cpw   = mha.c_proj.weight.data
    cpb   = mha.c_proj.bias.data
    mask  = mha.mask

    num = numerical_grad(
        lambda w: _mha_np(x_d, w, cab, cpw, cpb, mask, n_head),
        caw_d
    )
    check(mha.c_attn.weight, num, atol=1e-4, name='mha c_attn.W')


# ---------------------------------------------------------------------------
# 7. Block
# ---------------------------------------------------------------------------

def test_block_shape():
    """Output shape must equal input shape (residual connection preserves dims)"""
    np.random.seed(9)
    n_embd, n_head, block_size = 8, 2, 16
    B, T = 2, 5

    block = Block(n_embd, n_head, block_size)
    x     = Tensor(np.random.randn(B, T, n_embd))
    out   = block(x)
    assert out.data.shape == (B, T, n_embd)


def test_block_backward_runs():
    """Full backward through a Block must not crash and must produce non-zero grads"""
    np.random.seed(10)
    n_embd, n_head, block_size = 8, 2, 16
    B, T = 1, 4

    block = Block(n_embd, n_head, block_size)
    x_d   = np.random.randn(B, T, n_embd)
    X     = Tensor(x_d.copy())
    block(X).sum().backward()

    # Every parameter must have received a gradient
    for p in block.parameters():
        assert np.any(p.grad != 0), "A parameter has zero gradient likely disconnected"


def test_block_grad_input():
    """Grad check for input x through a single Block"""
    np.random.seed(11)
    n_embd, n_head, block_size = 8, 2, 16
    B, T = 1, 3

    block = Block(n_embd, n_head, block_size)

    # Snapshot all weight arrays for the numpy reference
    def _block_np(x):
        # We run the Tensor forward with a fresh input but fixed weights
        # This is valid: numerical_grad only perturbs x, not the weights
        block.zero_grad()
        X_ = Tensor(x)
        return float(block(X_).data.sum())

    x_d = np.random.randn(B, T, n_embd)
    X   = Tensor(x_d.copy())
    block.zero_grad()
    block(X).sum().backward()

    num = numerical_grad(_block_np, x_d.copy())
    check(X, num, atol=1e-4, name='block x')


# ---------------------------------------------------------------------------
# 8. GPTNano
# ---------------------------------------------------------------------------

def test_gptnano_forward_shape():
    """Logits must have shape (B, T, vocab_size)"""
    np.random.seed(12)
    vocab_size = 65
    model = GPTNano(vocab_size, n_embd=32, n_head=2, n_layer=1, block_size=16)

    B, T  = 2, 8
    idx   = np.random.randint(0, vocab_size, size=(B, T))
    logits = model(idx)
    assert logits.data.shape == (B, T, vocab_size), f"Wrong shape: {logits.data.shape}"


def test_gptnano_param_count():
    """
    Default GPTNano (vocab=65, n_embd=128, n_head=4, n_layer=2, block_size=64)
    should have ~400K parameters  
    Exact expected count: 421 632
    """
    model = GPTNano(vocab_size=65, n_embd=128, n_head=4, n_layer=2, block_size=64)
    n = model.num_params()
    expected = 421_632
    assert n == expected, f"Parameter count {n} != expected {expected}"


def test_gptnano_backward_runs():
    """Full forward + backward through GPTNano (small config) must not crash"""
    np.random.seed(13)
    vocab_size = 10
    model = GPTNano(vocab_size, n_embd=16, n_head=2, n_layer=1, block_size=8)
    model.zero_grad()

    B, T    = 2, 6
    idx     = np.random.randint(0, vocab_size, size=(B, T))
    targets = np.random.randint(0, vocab_size, size=(B, T))

    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from tensorgrad.functional import cross_entropy
    logits = model(idx)
    loss   = cross_entropy(logits, targets)
    loss.backward()

    # Every parameter must have a non-zero gradient after the backward
    for p in model.parameters():
        assert np.any(p.grad != 0), "A parameter has zero gradient likely disconnected"


def test_gptnano_grad_head_weight():
    """Grad check for the final linear head weight (small vocab, tiny model)"""
    np.random.seed(14)
    vocab_size = 6
    model = GPTNano(vocab_size, n_embd=8, n_head=2, n_layer=1, block_size=8)

    B, T  = 1, 3
    idx   = np.random.randint(0, vocab_size, size=(B, T))

    # Analytical gradient of sum(logits)
    model.zero_grad()
    model(idx).sum().backward()
    analytical = model.head.weight.grad.copy()

    # Numerical gradient only perturb head.weight, hold everything else fixed
    w_d = model.head.weight.data.copy()

    def f(w):
        model.head.weight.data[:] = w
        logits = model(idx)
        return float(logits.data.sum())

    num = numerical_grad(f, w_d.copy())
    # Restore
    model.head.weight.data[:] = w_d

    assert np.allclose(analytical, num, atol=1e-4), (
        f"head.weight grad max error: {np.abs(analytical - num).max():.2e}"
    )
