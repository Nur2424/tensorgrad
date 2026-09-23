"""
tests/test_optim.py
-------------------
Tests for tensorgrad/optim.py (AdamW)

Test structure
--------------
1. test_zero_grad_resets_all
     After accumulating arbitrary gradients through a forward+backward pass,
     zero_grad() must set every parameter's .grad back to zeros

2. test_adamw_first_step_manual
     At t=1 with a single scalar parameter, compute the expected update by
     hand and compare against AdamW.step() 
     The closed form at t=1 is:
         m = (1 - beta1) * g
         v = (1 - beta2) * g^2
         m^ = m / (1 - beta1)  = g              (bc1 cancels)
         v^ = v / (1 - beta2)  = g^2            (bc2 cancels)
         delta_p = lr * (g / (|g| + eps) + wd * p0)

3. test_adamw_weight_decay
     With gradient fixed at zero, the only update is -lr * wd * p
     After n steps: p_n = p_0 * (1 - lr * wd)^n
     Verified numerically

4. test_adamw_linear_regression
     The mandated sanity check: fit y = x @ w_true with MSE loss
     Loss must be strictly decreasing over 10 steps

5. test_adamw_moment_accumulation
     After multiple steps with a constant gradient g, the first moment
     converges toward g and the second moment toward g^2 (geometric series)

     Checks that the bias-corrected estimates are close to g and g^2 after enough steps
"""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tensorgrad import Tensor
from tensorgrad.optim import AdamW


# ---------------------------------------------------------------------------
# 1. zero_grad resets all parameter gradients
# ---------------------------------------------------------------------------

def test_zero_grad_resets_all():
    """
    User-mandated pre-condition: zero_grad() must return every .grad to zeros
    before the linear regression sanity check runs
    """
    np.random.seed(0)
    # Three parameters of varying shapes
    params = [
        Tensor(np.random.randn(4, 8)),
        Tensor(np.random.randn(8)),
        Tensor(np.random.randn(3, 3)),
    ]
    opt = AdamW(params, lr=1e-3)

    # Simulate a backward pass: put nonzero grads on every param
    for p in params:
        p.grad = np.random.randn(*p.shape)

    # Confirm grads are nonzero before reset
    assert any(np.any(p.grad != 0) for p in params), (
        "Pre-condition: at least one grad should be nonzero before zero_grad"
    )

    opt.zero_grad()

    # Every grad must be exactly zeros now
    for i, p in enumerate(params):
        assert np.all(p.grad == 0.0), (
            f"Parameter {i} still has nonzero grad after zero_grad(): "
            f"max |grad| = {np.abs(p.grad).max()}"
        )


# ---------------------------------------------------------------------------
# 2. Manual first-step verification
# ---------------------------------------------------------------------------

def test_adamw_first_step_manual():
    """
    At t=1 the bias-correction denominators cancel the EMA coefficients,
    so the Adam direction simplifies to g / (|g| + eps)

    Closed form (scalar, t=1):
        p_new = p_0 - lr * (g / (sqrt(g^2) + eps) + wd * p_0)
              = p_0 - lr * (sign(g) * |g|/(|g|+eps) + wd * p_0)
    """
    np.random.seed(1)
    p0   = 2.5
    g    = 0.8
    lr   = 0.1
    wd   = 0.01
    eps  = 1e-8
    b1, b2 = 0.9, 0.999

    param = Tensor(np.array([p0]))
    param.grad = np.array([g])

    opt = AdamW([param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    opt.step()

    # At t=1:  m=g*(1-b1), v=g^2*(1-b2); bc1=(1-b1), bc2=(1-b2)
    # m_hat = g, v_hat = g^2
    m_hat    = g                  # (1-b1)*g / (1-b1)
    v_hat    = g ** 2             # (1-b2)*g^2 / (1-b2)
    expected = p0 - lr * (m_hat / (np.sqrt(v_hat) + eps) + wd * p0)

    assert np.isclose(float(param.data[0]), expected, atol=1e-9), (
        f"First step: expected {expected:.8f}, got {float(param.data[0]):.8f}"
    )


# ---------------------------------------------------------------------------
# 3. Weight decay without gradient
# ---------------------------------------------------------------------------

def test_adamw_weight_decay():
    """
    With gradient = 0, Adam direction = 0, so the only update is:
        p <= p - lr * wd * p  =>  p_n = p_0 * (1 - lr * wd)^n

    After 20 steps the parameter must match this exponential decay
    """
    p0  = 3.0
    lr  = 0.01
    wd  = 0.1
    n   = 20

    param = Tensor(np.array([p0]))
    opt   = AdamW([param], lr=lr, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay=wd)

    for _ in range(n):
        param.grad = np.zeros_like(param.data)   # gradient is zero every step
        opt.step()

    # With zero gradient m=0, v=0, m_hat=0, v_hat=0
    # The update reduces to:  p <- p - lr * wd * p
    # BUT: v_hat=0 means denominator = sqrt(0) + eps = eps
    # m_hat / (sqrt(v_hat) + eps) = 0 / eps = 0
    # So it truly is pure decay
    expected = p0
    for _ in range(n):
        expected = expected * (1.0 - lr * wd)

    assert np.isclose(float(param.data[0]), expected, atol=1e-9), (
        f"Weight decay: expected {expected:.8f}, got {float(param.data[0]):.8f}"
    )


# ---------------------------------------------------------------------------
# 4. Linear regression sanity check 
# ---------------------------------------------------------------------------

def test_adamw_linear_regression():
    """
    Fit y = x @ w_true with MSE loss via AdamW

    Loss must be strictly decreasing over every one of 10 steps
    """
    np.random.seed(42)
    N, D = 32, 8

    x_np    = np.random.randn(N, D)
    w_true  = np.random.randn(D, 1)
    y_np    = x_np @ w_true + 0.01 * np.random.randn(N, 1)   # tiny noise

    X = Tensor(x_np, requires_grad=False)
    Y = Tensor(y_np, requires_grad=False)
    W = Tensor(np.zeros((D, 1)))   # learnable weight, initialised to zero

    opt = AdamW([W], lr=1e-2, weight_decay=0.0)   # no decay for clean check

    losses = []
    for step in range(10):
        opt.zero_grad()

        pred = X @ W                  # (N, 1)
        diff = pred - Y               # (N, 1)
        loss = (diff * diff).mean()   # scalar MSE
        loss.backward()
        opt.step()

        losses.append(float(loss.data))

    # Every step must be strictly better than the previous
    for i in range(1, len(losses)):
        assert losses[i] < losses[i - 1], (
            f"Loss did not decrease at step {i}: "
            f"{losses[i-1]:.6f} -> {losses[i]:.6f}\n"
            f"Full loss curve: {[f'{l:.4f}' for l in losses]}"
        )


# ---------------------------------------------------------------------------
# 5. Moment accumulation convergence
# ---------------------------------------------------------------------------

def test_adamw_moment_accumulation():
    """
    With a constant gradient g over many steps, the bias-corrected first
    and second moments should converge close to g and g^2 respectively
    (geometric series limit)
    """
    np.random.seed(3)
    g    = 0.5
    lr   = 1e-6   # tiny lr so parameter barely moves
    wd   = 0.0
    b1   = 0.9
    b2   = 0.999
    eps  = 1e-8
    T    = 500    # enough steps for convergence

    param = Tensor(np.array([0.0]))
    opt   = AdamW([param], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)

    for _ in range(T):
        param.grad = np.array([g])
        opt.step()

    # After T steps the bias-corrected estimates converge toward g and g^2
    m_hat = opt.m[0][0] / (1.0 - b1 ** T)
    v_hat = opt.v[0][0] / (1.0 - b2 ** T)

    assert abs(m_hat - g)    < 0.01, f"m_hat {m_hat:.4f} far from g {g}"
    assert abs(v_hat - g**2) < 0.01, f"v_hat {v_hat:.4f} far from g^2 {g**2}"
