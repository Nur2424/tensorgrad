"""
tests/test_engine.py
--------------------
Finite-difference gradient checks for every operation in engine.py

Strategy
--------
For each op:
  1. Run the forward pass with Tensor inputs
  2. Call .backward() to get analytical gradients
  3. Compute numerical gradients via central differences
  4. Assert they match within atol=1e-4
"""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from tensorgrad import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def numerical_grad(f, x, eps=1e-5):
    """
    Central-difference gradient of scalar function f at array x
    f : ndarray => scalar
    Returns an array the same shape as x
    """
    x = x.astype(np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = float(x[idx])

        x[idx] = orig + eps
        f_plus = float(f(x))

        x[idx] = orig - eps
        f_minus = float(f(x))

        x[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad


def check(tensor, numerical, atol=1e-4, name=''):
    """Assert that tensor.grad matches the numerical gradient"""
    max_err = np.abs(tensor.grad - numerical).max()
    assert np.allclose(tensor.grad, numerical, atol=atol), (
        f"{name} max error: {max_err:.2e}"
    )


# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

def test_add():
    np.random.seed(0)
    a_d, b_d = np.random.randn(3, 4), np.random.randn(3, 4)

    A, B = Tensor(a_d.copy()), Tensor(b_d.copy())
    (A + B).sum().backward()

    check(A, numerical_grad(lambda x: (x + b_d).sum(), a_d.copy()), name='add A')
    check(B, numerical_grad(lambda x: (a_d + x).sum(), b_d.copy()), name='add B')


def test_add_broadcast():
    """A has shape (3,4), B has shape (4,), B is broadcast along axis 0"""
    np.random.seed(1)
    a_d = np.random.randn(3, 4)
    b_d = np.random.randn(4)

    A, B = Tensor(a_d.copy()), Tensor(b_d.copy())
    (A + B).sum().backward()

    check(A, numerical_grad(lambda x: (x + b_d).sum(), a_d.copy()), name='add_bc A')
    check(B, numerical_grad(lambda x: (a_d + x).sum(), b_d.copy()), name='add_bc B')


# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

def test_mul():
    np.random.seed(2)
    a_d, b_d = np.random.randn(2, 3), np.random.randn(2, 3)

    A, B = Tensor(a_d.copy()), Tensor(b_d.copy())
    (A * B).sum().backward()

    check(A, numerical_grad(lambda x: (x * b_d).sum(), a_d.copy()), name='mul A')
    check(B, numerical_grad(lambda x: (a_d * x).sum(), b_d.copy()), name='mul B')


def test_mul_broadcast():
    """Scalar multiplication: B is shape ()"""
    np.random.seed(3)
    a_d = np.random.randn(3, 4)
    b_d = np.array(2.5)

    A, B = Tensor(a_d.copy()), Tensor(b_d.copy())
    (A * B).sum().backward()

    check(A, numerical_grad(lambda x: (x * float(b_d)).sum(), a_d.copy()), name='mul_bc A')
    check(B, numerical_grad(lambda x: (a_d * x).sum(), b_d.copy()), name='mul_bc B')


# ---------------------------------------------------------------------------
# Negation / subtraction
# ---------------------------------------------------------------------------

def test_neg():
    np.random.seed(4)
    a_d = np.random.randn(3, 3)

    A = Tensor(a_d.copy())
    (-A).sum().backward()

    check(A, numerical_grad(lambda x: (-x).sum(), a_d.copy()), name='neg')


def test_sub():
    np.random.seed(5)
    a_d, b_d = np.random.randn(2, 4), np.random.randn(2, 4)

    A, B = Tensor(a_d.copy()), Tensor(b_d.copy())
    (A - B).sum().backward()

    check(A, numerical_grad(lambda x: (x - b_d).sum(), a_d.copy()), name='sub A')
    check(B, numerical_grad(lambda x: (a_d - x).sum(), b_d.copy()), name='sub B')


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------

def test_pow():
    np.random.seed(6)
    a_d = np.abs(np.random.randn(3, 4)) + 0.1  # keep positive

    A = Tensor(a_d.copy())
    (A ** 3).sum().backward()

    check(A, numerical_grad(lambda x: (x ** 3).sum(), a_d.copy()), name='pow3')


def test_pow_negative():
    """Fractional / negative exponents"""
    np.random.seed(7)
    a_d = np.abs(np.random.randn(3, 4)) + 0.5

    A = Tensor(a_d.copy())
    (A ** -2).sum().backward()

    check(A, numerical_grad(lambda x: (x ** -2).sum(), a_d.copy()), name='pow-2')


# ---------------------------------------------------------------------------
# Matrix multiplication
# ---------------------------------------------------------------------------

def test_matmul_2d():
    np.random.seed(8)
    a_d = np.random.randn(4, 8)
    b_d = np.random.randn(8, 6)

    A, B = Tensor(a_d.copy()), Tensor(b_d.copy())
    (A @ B).sum().backward()

    check(A, numerical_grad(lambda x: (x @ b_d).sum(), a_d.copy()), name='mm A')
    check(B, numerical_grad(lambda x: (a_d @ x).sum(), b_d.copy()), name='mm B')


def test_matmul_batched():
    """Batched matmul: (B,T,C) @ (C,H) weight matrix is shared across batch"""
    np.random.seed(9)
    x_d = np.random.randn(2, 5, 8)   # (batch, seq, in)
    w_d = np.random.randn(8, 4)       # (in, out) broadcast over batch

    X, W = Tensor(x_d.copy()), Tensor(w_d.copy())
    (X @ W).sum().backward()

    check(X, numerical_grad(lambda x: (x @ w_d).sum(), x_d.copy()), name='bmm X')
    check(W, numerical_grad(lambda x: (x_d @ x).sum(), w_d.copy()), name='bmm W')


def test_matmul_chain():
    """
    Three weight matrices in sequence: loss = ((A @ B) @ C).sum()
    All three gradients must pass finite-difference checks
    """
    np.random.seed(10)
    a_d = np.random.randn(4, 8)
    b_d = np.random.randn(8, 16)
    c_d = np.random.randn(16, 4)

    A, B, C = Tensor(a_d.copy()), Tensor(b_d.copy()), Tensor(c_d.copy())
    ((A @ B) @ C).sum().backward()

    check(A, numerical_grad(lambda x: ((x @ b_d) @ c_d).sum(), a_d.copy()), name='chain A')
    check(B, numerical_grad(lambda x: ((a_d @ x) @ c_d).sum(), b_d.copy()), name='chain B')
    check(C, numerical_grad(lambda x: ((a_d @ b_d) @ x).sum(), c_d.copy()), name='chain C')


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def test_sum_all():
    np.random.seed(11)
    a_d = np.random.randn(3, 4)

    A = Tensor(a_d.copy())
    A.sum().backward()

    check(A, numerical_grad(lambda x: x.sum(), a_d.copy()), name='sum_all')


def test_sum_dim():
    np.random.seed(12)
    a_d = np.random.randn(3, 4, 5)

    A = Tensor(a_d.copy())
    A.sum(dim=1).sum().backward()

    check(A, numerical_grad(lambda x: x.sum(axis=1).sum(), a_d.copy()), name='sum_dim')


def test_sum_dim_negative():
    np.random.seed(13)
    a_d = np.random.randn(3, 4, 5)

    A = Tensor(a_d.copy())
    A.sum(dim=-1).sum().backward()

    check(A, numerical_grad(lambda x: x.sum(axis=-1).sum(), a_d.copy()), name='sum_dim_neg')


def test_mean_all():
    np.random.seed(14)
    a_d = np.random.randn(3, 4)

    A = Tensor(a_d.copy())
    A.mean().backward()

    check(A, numerical_grad(lambda x: x.mean(), a_d.copy()), name='mean_all')


def test_mean_dim():
    np.random.seed(15)
    a_d = np.random.randn(3, 4, 5)

    A = Tensor(a_d.copy())
    A.mean(dim=2).sum().backward()

    check(A, numerical_grad(lambda x: x.mean(axis=2).sum(), a_d.copy()), name='mean_dim')


# ---------------------------------------------------------------------------
# Element-wise ops
# ---------------------------------------------------------------------------

def test_exp():
    np.random.seed(16)
    a_d = np.random.randn(3, 4)

    A = Tensor(a_d.copy())
    A.exp().sum().backward()

    check(A, numerical_grad(lambda x: np.exp(x).sum(), a_d.copy()), name='exp')


def test_log():
    np.random.seed(17)
    a_d = np.abs(np.random.randn(3, 4)) + 0.1

    A = Tensor(a_d.copy())
    A.log().sum().backward()

    check(A, numerical_grad(lambda x: np.log(x).sum(), a_d.copy()), name='log')


def test_relu():
    np.random.seed(18)
    a_d = np.random.randn(4, 4)

    A = Tensor(a_d.copy())
    A.relu().sum().backward()

    check(A, numerical_grad(lambda x: np.maximum(0, x).sum(), a_d.copy()), name='relu')


def test_tanh():
    np.random.seed(19)
    a_d = np.random.randn(3, 4)

    A = Tensor(a_d.copy())
    A.tanh().sum().backward()

    check(A, numerical_grad(lambda x: np.tanh(x).sum(), a_d.copy()), name='tanh')


# ---------------------------------------------------------------------------
# Shape ops
# ---------------------------------------------------------------------------

def test_reshape():
    np.random.seed(20)
    a_d = np.random.randn(3, 4)

    A = Tensor(a_d.copy())
    A.reshape(2, 6).sum().backward()

    check(A, numerical_grad(lambda x: x.reshape(2, 6).sum(), a_d.copy()), name='reshape')


def test_transpose():
    np.random.seed(21)
    a_d = np.random.randn(3, 4, 5)

    A = Tensor(a_d.copy())
    A.transpose(0, 2).sum().backward()

    check(A, numerical_grad(lambda x: x.swapaxes(0, 2).sum(), a_d.copy()), name='transpose')


def test_T_property():
    """2D matrix transpose via .T property"""
    np.random.seed(22)
    a_d = np.random.randn(4, 6)

    A = Tensor(a_d.copy())
    A.T.sum().backward()

    check(A, numerical_grad(lambda x: x.T.sum(), a_d.copy()), name='T')


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def test_getitem_basic():
    np.random.seed(23)
    a_d = np.random.randn(5, 4)

    A = Tensor(a_d.copy())
    A[[0, 2, 4]].sum().backward()

    check(A, numerical_grad(lambda x: x[[0, 2, 4]].sum(), a_d.copy()), name='getitem')


def test_getitem_repeated():
    """Repeated indices gradient must accumulate, not overwrite"""
    np.random.seed(24)
    a_d = np.random.randn(5, 4)
    idx = np.array([0, 0, 2, 2, 2])  # row 0 twice, row 2 three times

    A = Tensor(a_d.copy())
    A[idx].sum().backward()

    check(A, numerical_grad(lambda x: x[idx].sum(), a_d.copy()), name='getitem_repeated')


# ---------------------------------------------------------------------------
# Masked fill
# ---------------------------------------------------------------------------

def test_masked_fill():
    """
    We use fill value 0.0 (not -inf) so the function stays finite and
    central differences are numerically stable
    """
    np.random.seed(25)
    a_d = np.random.randn(3, 5)
    mask = np.array([[True, False, True, False, True],
                     [False, True, False, True, False],
                     [True, True, False, False, False]])

    A = Tensor(a_d.copy())
    A.masked_fill(mask, 0.0).sum().backward()

    def f(x):
        xc = x.copy()
        xc[mask] = 0.0
        return xc.sum()

    check(A, numerical_grad(f, a_d.copy()), name='masked_fill')


# ---------------------------------------------------------------------------
# Concatenation
# ---------------------------------------------------------------------------

def test_cat():
    np.random.seed(26)
    a_d = np.random.randn(3, 4)
    b_d = np.random.randn(2, 4)
    c_d = np.random.randn(5, 4)

    A, B, C = Tensor(a_d.copy()), Tensor(b_d.copy()), Tensor(c_d.copy())
    Tensor.cat([A, B, C], dim=0).sum().backward()

    check(A, numerical_grad(lambda x: np.concatenate([x, b_d, c_d], axis=0).sum(), a_d.copy()), name='cat A')
    check(B, numerical_grad(lambda x: np.concatenate([a_d, x, c_d], axis=0).sum(), b_d.copy()), name='cat B')
    check(C, numerical_grad(lambda x: np.concatenate([a_d, b_d, x], axis=0).sum(), c_d.copy()), name='cat C')


# ---------------------------------------------------------------------------
# Compound expression: x @ W + b  (the building block of every linear layer)
# ---------------------------------------------------------------------------

def test_compound_linear():
    """
    Forward: out = x @ W + b
    This exercises matmul + add + broadcast in a single backward pass
    """
    np.random.seed(27)
    x_d = np.random.randn(8, 16)
    w_d = np.random.randn(16, 32)
    b_d = np.random.randn(32)

    X, W, B = Tensor(x_d.copy()), Tensor(w_d.copy()), Tensor(b_d.copy())
    (X @ W + B).sum().backward()

    check(X, numerical_grad(lambda x: (x @ w_d + b_d).sum(), x_d.copy()), name='linear X')
    check(W, numerical_grad(lambda x: (x_d @ x + b_d).sum(), w_d.copy()), name='linear W')
    check(B, numerical_grad(lambda x: (x_d @ w_d + x).sum(), b_d.copy()), name='linear B')
