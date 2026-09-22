"""
tests/test_functional.py
------------------------
Finite-difference gradient checks for every function in functional.py

For each function:
  Run forward + backward with Tensor inputs
  Compare analytical gradients against central-difference estimates

Special cases
-------------
  cross_entropy: only logits have a gradient, targets are integer indices
  embedding:     only weight has a gradient, idx is integer
  softmax:       sum(softmax(x)) == 1 always, so we contract with a fixed
                 weight vector to get a non-trivial scalar loss
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorgrad import Tensor
from tensorgrad.functional import softmax, cross_entropy, layer_norm, embedding


# ---------------------------------------------------------------------------
# Shared helpers (same as test_engine.py, duplicated to keep files self-contained)
# ---------------------------------------------------------------------------

def numerical_grad(f, x, eps=1e-5):
    """Central-difference gradient of scalar f at array x"""
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
# Reference numpy implementations (used inside numerical_grad lambdas)
# ---------------------------------------------------------------------------

def _softmax_np(x, dim=-1):
    shifted = x - x.max(axis=dim, keepdims=True)
    exp_x   = np.exp(shifted)
    return exp_x / exp_x.sum(axis=dim, keepdims=True)


def _cross_entropy_np(logits, targets):
    orig     = logits.shape
    l2d      = logits.reshape(-1, orig[-1])
    t_flat   = targets.reshape(-1)
    N        = len(t_flat)
    shifted  = l2d - l2d.max(axis=1, keepdims=True)
    exp_l    = np.exp(shifted)
    p        = exp_l / exp_l.sum(axis=1, keepdims=True)
    return -np.log(np.clip(p[np.arange(N), t_flat], 1e-12, None)).mean()


def _layer_norm_np(x, w, b, eps=1e-5):
    mu   = x.mean(axis=-1, keepdims=True)
    var  = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    xhat = (x - mu) / np.sqrt(var + eps)
    return (w * xhat + b).sum()


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

def test_softmax_2d():
    """
    Loss = (softmax(x) * W_fixed).sum()
    W_fixed is a fixed array, contracting with it gives a non-trivial scalar
    """
    np.random.seed(0)
    x_d = np.random.randn(5, 8)
    w   = np.random.randn(5, 8)   # fixed weights, not differentiated

    X = Tensor(x_d.copy())
    loss = (softmax(X, dim=-1) * Tensor(w, requires_grad=False)).sum()
    loss.backward()

    def f(x): return (_softmax_np(x, dim=-1) * w).sum()
    check(X, numerical_grad(f, x_d.copy()), name='softmax_2d')


def test_softmax_3d():
    """Batched softmax over dim=-1 on a (B, T, C) tensor"""
    np.random.seed(1)
    x_d = np.random.randn(2, 4, 6)
    w   = np.random.randn(2, 4, 6)

    X = Tensor(x_d.copy())
    loss = (softmax(X, dim=-1) * Tensor(w, requires_grad=False)).sum()
    loss.backward()

    def f(x): return (_softmax_np(x, dim=-1) * w).sum()
    check(X, numerical_grad(f, x_d.copy()), name='softmax_3d')


def test_softmax_dim0():
    """Softmax along axis 0 (column-wise)"""
    np.random.seed(2)
    x_d = np.random.randn(4, 6)
    w   = np.random.randn(4, 6)

    X = Tensor(x_d.copy())
    loss = (softmax(X, dim=0) * Tensor(w, requires_grad=False)).sum()
    loss.backward()

    def f(x): return (_softmax_np(x, dim=0) * w).sum()
    check(X, numerical_grad(f, x_d.copy()), name='softmax_dim0')


# ---------------------------------------------------------------------------
# cross_entropy
# ---------------------------------------------------------------------------

def test_cross_entropy_2d():
    """2-D logits (N, C), gradient on logits only"""
    np.random.seed(3)
    N, C     = 8, 10
    logits_d = np.random.randn(N, C)
    targets  = np.random.randint(0, C, size=(N,))

    L = Tensor(logits_d.copy())
    cross_entropy(L, targets).backward()

    def f(x): return _cross_entropy_np(x, targets)
    check(L, numerical_grad(f, logits_d.copy()), name='ce_2d logits')


def test_cross_entropy_3d():
    """3-D logits (B, T, C) language model use case"""
    np.random.seed(4)
    B, T, C  = 2, 6, 12
    logits_d = np.random.randn(B, T, C)
    targets  = np.random.randint(0, C, size=(B, T))

    L = Tensor(logits_d.copy())
    cross_entropy(L, targets).backward()

    def f(x): return _cross_entropy_np(x, targets)
    check(L, numerical_grad(f, logits_d.copy()), name='ce_3d logits')


def test_cross_entropy_known_gradient():
    """
    Closed-form check: for a 1-sample, 3-class problem the fused gradient is
    exactly (p - e_y) where e_y is the one-hot vector of the true class
    """
    np.random.seed(5)
    logits_d = np.array([[2.0, 1.0, 0.1]])
    targets  = np.array([0])

    # Compute softmax by hand
    shifted = logits_d - logits_d.max()
    p       = np.exp(shifted) / np.exp(shifted).sum()
    expected_grad = p.copy()
    expected_grad[0, targets[0]] -= 1.0
    # N=1, so divide by 1

    L = Tensor(logits_d.copy())
    cross_entropy(L, targets).backward()

    assert np.allclose(L.grad, expected_grad, atol=1e-6), (
        f"Expected {expected_grad}, got {L.grad}"
    )


# ---------------------------------------------------------------------------
# layer_norm
# ---------------------------------------------------------------------------

def test_layer_norm_2d():
    """2-D input (N, D): check gradients of x, weight, and bias"""
    np.random.seed(6)
    N, D = 5, 8
    x_d = np.random.randn(N, D)
    w_d = np.random.randn(D)
    b_d = np.random.randn(D)

    X, W, B = Tensor(x_d.copy()), Tensor(w_d.copy()), Tensor(b_d.copy())
    layer_norm(X, W, B).sum().backward()

    check(X, numerical_grad(lambda x: _layer_norm_np(x, w_d, b_d), x_d.copy()), name='ln2d x')
    check(W, numerical_grad(lambda w: _layer_norm_np(x_d, w, b_d), w_d.copy()), name='ln2d w')
    check(B, numerical_grad(lambda b: _layer_norm_np(x_d, w_d, b),  b_d.copy()), name='ln2d b')


def test_layer_norm_3d():
    """3-D input (B, T, D): transformer use case"""
    np.random.seed(7)
    B, T, D = 2, 4, 6
    x_d = np.random.randn(B, T, D)
    w_d = np.random.randn(D)
    b_d = np.random.randn(D)

    X, W, B = Tensor(x_d.copy()), Tensor(w_d.copy()), Tensor(b_d.copy())
    layer_norm(X, W, B).sum().backward()

    check(X, numerical_grad(lambda x: _layer_norm_np(x, w_d, b_d), x_d.copy()), name='ln3d x')
    check(W, numerical_grad(lambda w: _layer_norm_np(x_d, w, b_d), w_d.copy()), name='ln3d w')
    check(B, numerical_grad(lambda b: _layer_norm_np(x_d, w_d, b),  b_d.copy()), name='ln3d b')


def test_layer_norm_nonunit_weight():
    """Weight not all-ones ensures weight grad is computed correctly"""
    np.random.seed(8)
    N, D = 4, 10
    x_d = np.random.randn(N, D)
    w_d = np.random.randn(D) * 2.0   # non-trivial scale
    b_d = np.random.randn(D) * 0.5

    X, W, B = Tensor(x_d.copy()), Tensor(w_d.copy()), Tensor(b_d.copy())
    layer_norm(X, W, B).sum().backward()

    check(X, numerical_grad(lambda x: _layer_norm_np(x, w_d, b_d), x_d.copy()), name='lnW x')
    check(W, numerical_grad(lambda w: _layer_norm_np(x_d, w, b_d), w_d.copy()), name='lnW w')
    check(B, numerical_grad(lambda b: _layer_norm_np(x_d, w_d, b),  b_d.copy()), name='lnW b')


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------

def test_embedding_basic():
    """Each token index used exactly once"""
    np.random.seed(9)
    V, D = 10, 8
    w_d  = np.random.randn(V, D)
    idx  = np.array([0, 3, 7, 2])    # 4 distinct tokens

    W = Tensor(w_d.copy())
    embedding(W, idx).sum().backward()

    def f(w): return w[idx].sum()
    check(W, numerical_grad(f, w_d.copy()), name='emb basic')


def test_embedding_repeated():
    """
    Repeated token indices: gradient must accumulate (scatter_add) not overwrite
    Row 2 appears three times 
    """
    np.random.seed(10)
    V, D = 6, 4
    w_d  = np.random.randn(V, D)
    idx  = np.array([2, 0, 2, 5, 2])   # row 2 appears 3 times

    W = Tensor(w_d.copy())
    embedding(W, idx).sum().backward()

    def f(w): return w[idx].sum()
    check(W, numerical_grad(f, w_d.copy()), name='emb repeated')


def test_embedding_2d_idx():
    """2-D index array (B, T) token indices as in a language model"""
    np.random.seed(11)
    V, D = 20, 8
    w_d  = np.random.randn(V, D)
    idx  = np.random.randint(0, V, size=(3, 5))   # (B, T)

    W = Tensor(w_d.copy())
    embedding(W, idx).sum().backward()

    def f(w): return w[idx].sum()
    check(W, numerical_grad(f, w_d.copy()), name='emb 2d idx')


# ---------------------------------------------------------------------------
# Integration: embedding => layer_norm => linear => cross_entropy
# ---------------------------------------------------------------------------

def test_integration_embed_ln_ce():
    """
    Minimal transformer column:
        tokens => embedding => layer_norm => linear projection => cross_entropy
    Checks that gradients flow correctly through all four functions in sequence 
    """
    np.random.seed(12)
    B, T, D, C = 2, 4, 8, 16

    # Fixed data
    tokens  = np.random.randint(0, 20, size=(B, T))
    targets = np.random.randint(0, C,  size=(B, T))

    # Parameters
    Wemb  = Tensor(np.random.randn(20, D) * 0.02)
    gamma = Tensor(np.ones(D))
    beta  = Tensor(np.zeros(D))
    Wout  = Tensor(np.random.randn(D, C) * 0.02)
    bout  = Tensor(np.zeros(C))

    # Forward
    x      = embedding(Wemb, tokens)           # (B, T, D)
    x      = layer_norm(x, gamma, beta)        # (B, T, D)
    logits = x @ Wout + bout                   # (B, T, C)
    loss   = cross_entropy(logits, targets)    # scalar
    loss.backward()

    # Spot check: Wout gradient via finite differences
    def f_wout(w):
        x_  = Wemb.data[tokens]
        mu_ = x_.mean(axis=-1, keepdims=True)
        v_  = ((x_ - mu_) ** 2).mean(axis=-1, keepdims=True)
        xh_ = (x_ - mu_) / np.sqrt(v_ + 1e-5)
        lg_ = xh_ @ w + bout.data
        return _cross_entropy_np(lg_, targets)

    num_grad = numerical_grad(f_wout, Wout.data.copy())
    check(Wout, num_grad, atol=1e-4, name='integration Wout')
