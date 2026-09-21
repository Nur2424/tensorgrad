"""
tensorgrad/engine.py
--------------------
Core Tensor class: a thin wrapper around a NumPy array that records a
computation graph and can run reverse-mode autodiff through it.

Every operation returns a new Tensor whose ._backward closure implements the
chain rule for that operation.  Backward closures carry a comment showing the
matrix expression they implement.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Broadcasting helper
# ---------------------------------------------------------------------------

def _unbroadcast(grad, shape):
    """
    Sum grad over every axis that was broadcast to produce it, so that the
    result has the same shape as `shape`

    Forward broadcasting pads dims on the left and stretches size-1 dims.
    The adjoint (backward) sums over those same axes
    """
    # 1. If grad has more dimensions than the target, sum over the leading axes
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)

    # 2. Sum over any axis where the target was size 1 (keepdims to preserve rank)
    for i, (g, t) in enumerate(zip(grad.shape, shape)):
        if t == 1 and g != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


# ---------------------------------------------------------------------------
# Tensor
# ---------------------------------------------------------------------------

class Tensor:
    """
    A multi-dimensional array that tracks a computation graph for autodiff

    Attributes
    ----------
    data          : np.ndarray  - the forward value
    grad          : np.ndarray  - accumulated gradient (same shape as data)
    requires_grad : bool        - whether to build the graph for this tensor
    _backward     : callable    - closure that deposits gradients into inputs
    _prev         : set         - input Tensors this node depends on
    _op           : str         - name of the operation (for debugging)
    """

    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def T(self):
        """Transpose last two axes (matrix transpose convention)"""
        return self.transpose(-2, -1)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self):
        """
        Topological sort of the graph, then call each node's ._backward
        in reverse order (from output back to inputs)
        """
        topo, visited = [], set()

        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        # Seed: dL/dL = 1
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    # ------------------------------------------------------------------
    # Utility: wrap scalars / arrays as Tensors when needed
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure(x):
        if isinstance(x, Tensor):
            return x
        return Tensor(x, requires_grad=False)

    # ------------------------------------------------------------------
    # Arithmetic ops
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            # C = A + B  =>  dA = dC, dB = dC  (sum over broadcast dims)
            self.grad  += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            # C = A * B  =>  dA = dC * B, dB = dC * A
            self.grad  += _unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += _unbroadcast(out.grad * self.data,  other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-Tensor._ensure(other))

    def __rsub__(self, other):
        return Tensor._ensure(other) + (-self)

    def __truediv__(self, other):
        return self * Tensor._ensure(other) ** -1

    def __rtruediv__(self, other):
        return Tensor._ensure(other) * self ** -1

    def __pow__(self, exp):
        assert isinstance(exp, (int, float)), "exponent must be a scalar"
        out = Tensor(self.data ** exp, (self,), f'**{exp}')

        def _backward():
            # d/dA [A^n] = n * A^(n-1)
            self.grad += exp * (self.data ** (exp - 1)) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------

    def __matmul__(self, other):
        other = Tensor._ensure(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            # C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
            # Works for batched matmul: swapaxes(-2,-1) transposes the last two dims
            # _unbroadcast handles the case where B was broadcast (e.g. shared weight)
            self.grad  += out.grad @ other.data.swapaxes(-2, -1)
            other.grad += _unbroadcast(self.data.swapaxes(-2, -1) @ out.grad,
                                       other.data.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Reduction ops
    # ------------------------------------------------------------------

    def sum(self, dim=None, keepdims=False):
        out = Tensor(self.data.sum(axis=dim, keepdims=keepdims), (self,), 'sum')

        def _backward():
            # Summing collapses axes; the adjoint broadcasts dC back to A's shape
            grad = out.grad
            if dim is not None and not keepdims:
                # Normalize negative dim to positive so expand_dims is unambiguous
                d = dim if dim >= 0 else self.data.ndim + dim
                grad = np.expand_dims(grad, axis=d)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, dim=None, keepdims=False):
        out = Tensor(self.data.mean(axis=dim, keepdims=keepdims), (self,), 'mean')

        def _backward():
            # mean = sum / n  =>  gradient is dC / n, broadcast back
            grad = out.grad
            if dim is not None and not keepdims:
                d = dim if dim >= 0 else self.data.ndim + dim
                grad = np.expand_dims(grad, axis=d)
            # n = number of elements averaged over
            if dim is None:
                n = self.data.size
            else:
                n = self.data.shape[dim]
            self.grad += np.broadcast_to(grad, self.data.shape) / n

        out._backward = _backward
        return out

    def max(self, dim=None, keepdims=False):
        out = Tensor(self.data.max(axis=dim, keepdims=keepdims), (self,), 'max')

        def _backward():
            # Gradient flows only to the argmax element
            # For ties, we pick one winner (same as PyTorch)
            grad = out.grad
            if dim is not None and not keepdims:
                d = dim if dim >= 0 else self.data.ndim + dim
                grad = np.expand_dims(grad, axis=d)
            val = np.expand_dims(out.data, axis=dim) if (dim is not None and not keepdims) else out.data
            mask = (self.data == np.broadcast_to(val, self.data.shape))
            # Normalise so ties share the gradient equally
            mask = mask / mask.sum(axis=dim, keepdims=True)
            self.grad += mask * np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Element-wise nonlinearities
    # ------------------------------------------------------------------

    def exp(self):
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            # d/dA [e^A] = e^A = out
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            # d/dA [log A] = 1/A
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')

        def _backward():
            # ReLU gate: gradient passes through where input > 0
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            # d/dA [tanh A] = 1 - tanh^2(A)
            self.grad += (1.0 - out.data ** 2) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Shape ops
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            # Reshape is its own inverse: just reshape the gradient back
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self, dim0, dim1):
        out = Tensor(np.swapaxes(self.data, dim0, dim1), (self,), 'transpose')

        def _backward():
            # Transpose is self-inverse: swap the same axes in the gradient
            self.grad += np.swapaxes(out.grad, dim0, dim1)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Indexing / gathering
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), '[]')

        def _backward():
            # Scatter: add dC at each index position back into dA
            # np.add.at handles repeated indices correctly (accumulates)
            np.add.at(self.grad, idx, out.grad)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Masking
    # ------------------------------------------------------------------

    def masked_fill(self, mask, value):
        """
        Return a new Tensor where positions where `mask` is True are replaced
        by `value`.  Gradient is zeroed at masked positions

        mask  : boolean array broadcastable to self.shape
                It is broadcast to the full data shape here so that a 2-D
                causal mask (T, T) works correctly on a 4-D attention tensor
                (B, n_head, T, T) without the caller having to expand dims
        value : scalar
        """
        # Broadcast mask to the full data shape so boolean indexing is
        # unambiguous regardless of how many leading batch/head dims exist.
        mask = np.broadcast_to(np.asarray(mask), self.data.shape)
        # np.broadcast_to returns a read-only view; copy so we can reuse it
        mask = mask.copy()

        new_data = self.data.copy()
        new_data[mask] = value
        out = Tensor(new_data, (self,), 'masked_fill')

        def _backward():
            # Gradient does not flow through masked positions
            grad = out.grad.copy()
            grad[mask] = 0.0
            self.grad += grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    @staticmethod
    def cat(tensors, dim=0):
        """
        Concatenate a list of Tensors along `dim`
        The backward splits the gradient back into the original chunk sizes
        """
        arrays = [t.data for t in tensors]
        out_data = np.concatenate(arrays, axis=dim)
        out = Tensor(out_data, tuple(tensors), 'cat')

        # Record the size of each tensor along `dim` so we can split later
        sizes = [t.data.shape[dim] for t in tensors]

        def _backward():
            # np.split returns views; we accumulate into each input's .grad
            grads = np.split(out.grad, np.cumsum(sizes[:-1]), axis=dim)
            for t, g in zip(tensors, grads):
                t.grad += g

        out._backward = _backward
        return out