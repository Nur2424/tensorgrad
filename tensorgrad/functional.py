"""
tensorgrad/functional.py
------------------------
Differentiable functions that go beyond the primitive ops in engine.py

Each function takes Tensor inputs and returns a Tensor whose ._backward
closure implements the analytically derived gradient.  Where possible we use
the "one-shot" closed-form backward rather than running the chain rule 
through many small intermediate nodes

Functions
---------
  softmax(x, dim)
  cross_entropy(logits, targets)     => fused softmax + NLL, O(1) graph depth
  layer_norm(x, weight, bias, eps)   => one-shot backward (BatchNorm formula, dim=-1)
  embedding(weight, idx)             => gather + scatter_add backward
"""

import numpy as np
from .engine import Tensor


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(x, dim=-1):
    """
    Numerically stable softmax along `dim`

    Forward:
        p = exp(x - max(x, dim)) / sum(exp(x - max(x, dim)), dim)

    Backward (Jacobian-vector product for softmax):
        dx = p * (dout - sum(dout * p, dim, keepdims=True))

    The Jacobian of softmax is  J_ij = p_i * (delta_ij - p_j)
    The JVP collapses to the expression above
    """
    # Subtract row-max for numerical stability does not change output
    shifted = x.data - x.data.max(axis=dim, keepdims=True)
    exp_x   = np.exp(shifted)
    p       = exp_x / exp_x.sum(axis=dim, keepdims=True)

    out = Tensor(p, (x,), 'softmax')

    def _backward():
        # dx = p * (dout - (dout * p).sum(dim))
        dot      = (out.grad * p).sum(axis=dim, keepdims=True)
        x.grad  += p * (out.grad - dot)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Cross-entropy (fused softmax + NLL)
# ---------------------------------------------------------------------------

def cross_entropy(logits, targets):
    """
    Fused softmax + negative log-likelihood loss, averaged over the batch

    Forward:
        p    = softmax(logits, dim=-1)
        loss = -mean( log p[i, y_i] )

    Backward:
        dlogits = (p - one_hot(targets)) / N

    Accepts 2-D logits (N, C) or 3-D logits (B, T, C)
    targets must be an integer array of shape (N,) or (B, T)
    Gradient is deposited into logits.grad, targets has no gradient
    """
    orig_shape  = logits.data.shape          # (N, C) or (B, T, C)
    logits_2d   = logits.data.reshape(-1, orig_shape[-1])
    targets_flat = np.asarray(targets).reshape(-1)
    N, C         = logits_2d.shape

    # Numerically stable softmax (max subtracted per row)
    shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
    exp_l   = np.exp(shifted)
    p       = exp_l / exp_l.sum(axis=1, keepdims=True)   # (N, C)

    # Loss: mean of -log p[i, y_i]
    log_probs = -np.log(np.clip(p[np.arange(N), targets_flat], 1e-12, None))
    loss      = log_probs.mean()

    out = Tensor(loss, (logits,), 'cross_entropy')

    def _backward():
        # dlogits = (p - 1[j == y_i]) / N
        dlogits = p.copy()
        dlogits[np.arange(N), targets_flat] -= 1.0
        dlogits /= N
        logits.grad += dlogits.reshape(orig_shape)

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Layer normalization
# ---------------------------------------------------------------------------

def layer_norm(x, weight, bias, eps=1e-5):
    """
    Layer normalization over the last dimension

    Forward:
        mu   = mean(x, dim=-1, keepdims=True)
        var  = mean((x - mu)^2, dim=-1, keepdims=True)
        rstd = 1 / sqrt(var + eps)
        xhat = (x - mu) * rstd
        out  = weight * xhat + bias

    Backward:
    The BatchNorm one-shot formula generalizes to LayerNorm by replacing:
        - normalization axis 0 (batch)  =>  axis -1 (features)
        - n = batch size                =>  D = feature dimension

        dxhat = dout * weight
        dx    = (1/D) * rstd * (D * dxhat
                                - dxhat.sum(-1, keepdims=True)
                                - xhat * (dxhat * xhat).sum(-1, keepdims=True))

    x      : Tensor (..., D)
    weight : Tensor (D,)    - learnable scale (gamma)
    bias   : Tensor (D,)    - learnable shift (beta)
    """
    D  = x.data.shape[-1]

    mu   = x.data.mean(axis=-1, keepdims=True)                    # (..., 1)
    xmu  = x.data - mu                                            # (..., D)
    var  = (xmu ** 2).mean(axis=-1, keepdims=True)                # (..., 1)
    rstd = 1.0 / np.sqrt(var + eps)                               # (..., 1)
    xhat = xmu * rstd                                             # (..., D)
    out_data = weight.data * xhat + bias.data                     # (..., D)

    out = Tensor(out_data, (x, weight, bias), 'layer_norm')

    def _backward():
        dout = out.grad                                            # (..., D)

        # weight and bias gradients: sum over all leading (batch/seq) dims
        reduce_axes  = tuple(range(dout.ndim - 1))
        weight.grad += (dout * xhat).sum(axis=reduce_axes)        # (D,)
        bias.grad   += dout.sum(axis=reduce_axes)                 # (D,)

        # One-shot x backward:
        #   dxhat = dout * gamma
        #   dx    = (1/D) * rstd * (D*dxhat - sum(dxhat,-1) - xhat*sum(dxhat*xhat,-1))
        dxhat  = dout * weight.data                               # (..., D)
        x.grad += (1.0 / D) * rstd * (
            D * dxhat
            - dxhat.sum(axis=-1, keepdims=True)
            - xhat  * (dxhat * xhat).sum(axis=-1, keepdims=True)
        )

    out._backward = _backward
    return out


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embedding(weight, idx):
    """
    Embedding table lookup

    Forward:
        out = weight[idx]      => gather rows

    Backward:
        dweight[idx] += dout   => scatter_add 

    np.add.at handles repeated token indices by accumulating gradients

    weight : Tensor (V, D)  => embedding matrix
    idx    : integer ndarray of any shape  => token indices
    """
    idx = np.asarray(idx)
    out = Tensor(weight.data[idx], (weight,), 'embedding')

    def _backward():
        # dweight[idx] += dout, accumulate, not overwrite, for repeated idx
        np.add.at(weight.grad, idx, out.grad)

    out._backward = _backward
    return out
