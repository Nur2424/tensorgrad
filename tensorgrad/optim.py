"""
tensorgrad/optim.py
-------------------
AdamW optimizer

AdamW decouples weight decay from the gradient update, applying it
directly to the parameter (L2 regularization in parameter space) rather
than folding it into the gradient
This is the GPT-2 / nanoGPT default

Reference:
  Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
  Algorithm 2 (AdamW)
"""

import numpy as np


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------

class AdamW:
    """
    AdamW optimizer

    Update rule (per parameter p, with gradient g):
        m  = beta1 * m  + (1 - beta1) * g          # 1st moment (momentum)
        v  = beta2 * v  + (1 - beta2) * g^2        # 2nd moment (variance)
        m^ = m  / (1 - beta1^t)                     # bias-corrected 1st moment
        v^ = v  / (1 - beta2^t)                     # bias-corrected 2nd moment
        p  = p  - lr * (m^ / (sqrt(v^) + eps)      # Adam step
                        + weight_decay * p)         # AdamW decay on param

    Parameters
    ----------
    params       : iterable of Tensor  => parameters to optimise
    lr           : float               => learning rate
    betas        : (float, float)      => (beta1, beta2) EMA coefficients
    eps          : float               => denominator stability term
    weight_decay : float               => L2 penalty coefficient (decoupled)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01):
        self.params       = list(params)
        self.lr           = lr
        self.beta1, self.beta2 = betas
        self.eps          = eps
        self.weight_decay = weight_decay
        self.t            = 0                        # step counter

        # First and second moment buffers, one per parameter
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def step(self):
        """Apply one AdamW update to all parameters"""
        self.t += 1

        # Bias-correction denominators (scalars, same for all params this step)
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        for i, p in enumerate(self.params):
            g = p.grad  # current gradient

            # Moment updates (EMA of gradient and squared gradient)
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g ** 2

            # Bias-corrected moments
            m_hat = self.m[i] / bc1
            v_hat = self.v[i] / bc2

            # AdamW parameter update:
            #   p <= p - lr * (adam_direction + weight_decay * p)
            p.data -= self.lr * (m_hat / (np.sqrt(v_hat) + self.eps)
                                 + self.weight_decay * p.data)

    # ------------------------------------------------------------------
    # Gradient reset
    # ------------------------------------------------------------------

    def zero_grad(self):
        """Reset all parameter gradients to zero before the next forward pass"""
        for p in self.params:
            p.grad = np.zeros_like(p.data)
