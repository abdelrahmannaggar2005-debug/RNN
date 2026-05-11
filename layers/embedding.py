"""
layers/embedding.py
===================
Embedding Layer and Linear Output Projection

═══════════════════════════════════════════════════════
EMBEDDING LAYER
═══════════════════════════════════════════════════════

Maps integer token indices to dense vectors:
    E : (vocab_size, embed_dim) — the embedding matrix

Forward:
    x_embedded[n, t, :] = E[token_idx[n, t], :]   (simple row lookup)

Backward:
    dE[token_idx[n,t], :] += d_out[n, t, :]
    (accumulate gradients for each row that was accessed)

This is cheaper than a one-hot @ E matrix multiply:
    - Forward: O(N*T) lookups vs O(N*T*V) multiplications
    - Backward: sparse scatter-add instead of dense matrix multiply

═══════════════════════════════════════════════════════
LINEAR OUTPUT PROJECTION
═══════════════════════════════════════════════════════

Maps hidden states to vocabulary logits:
    logits = H_out @ W_out + b_out
    shape:  (N*T, H) @ (H, V) = (N*T, V)

Standard dense layer applied to every time step simultaneously.
"""

import numpy as np


class EmbeddingLayer:
    """
    Learnable embedding table that maps token IDs to vectors.

    Parameters
    ----------
    vocab_size : V — number of unique tokens
    embed_dim  : E — dimensionality of embedding vectors
    """

    def __init__(self, vocab_size, embed_dim):
        self.V = vocab_size
        self.E = embed_dim

        # Initialize embeddings with small uniform values
        # Alternatively: N(0, 1/sqrt(E))
        self.weight = np.random.randn(vocab_size, embed_dim) * 0.1  # (V, E)
        self.d_weight = np.zeros_like(self.weight)                  # (V, E)

        self._token_ids = None  # cache for backward

    def forward(self, token_ids):
        """
        Parameters
        ----------
        token_ids : np.ndarray (int), shape (N, T) — integer token indices

        Returns
        -------
        out : np.ndarray, shape (N, T, E) — embedding vectors
        """
        self._token_ids = token_ids
        # Fancy indexing: select rows from embedding matrix
        return self.weight[token_ids]  # (N, T, E)

    def backward(self, d_out):
        """
        Accumulate gradients for each accessed embedding row.

        Parameters
        ----------
        d_out : np.ndarray, shape (N, T, E)

        Returns
        -------
        None (no gradient flows to token_ids — they're integer indices)
        """
        self.d_weight = np.zeros_like(self.weight)
        N, T, E = d_out.shape
        # Scatter-add: for each (n,t), add d_out[n,t,:] to row token_ids[n,t]
        np.add.at(self.d_weight, self._token_ids, d_out)

    def get_params(self): return {'weight': self.weight}
    def set_params(self, p): self.weight = p['weight']
    def get_grads(self): return {'weight': self.d_weight}


class LinearLayer:
    """
    Linear projection: out = X @ W + b

    Used to project hidden states (N, H) → vocabulary logits (N, V).
    Supports 2D and 3D inputs (reshapes internally).
    """

    def __init__(self, input_dim, output_dim):
        self.D = input_dim
        self.O = output_dim

        # He init (used even for output layer for consistency)
        std = np.sqrt(2.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * std   # (D, O)
        self.b = np.zeros(output_dim)                            # (O,)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._input = None
        self._input_shape = None

    def forward(self, X):
        """
        Parameters
        ----------
        X : shape (N, D) or (N, T, D)

        Returns
        -------
        out : shape (N, O) or (N, T, O)
        """
        self._input_shape = X.shape
        X_flat = X.reshape(-1, self.D)   # (N*T, D) or (N, D)
        self._input = X_flat
        out_flat = X_flat @ self.W + self.b  # (N*T, O)
        return out_flat.reshape(X.shape[:-1] + (self.O,))

    def backward(self, d_out):
        d_flat = d_out.reshape(-1, self.O)
        self.dW = self._input.T @ d_flat
        self.db = d_flat.sum(axis=0)
        dX_flat = d_flat @ self.W.T
        return dX_flat.reshape(self._input_shape)

    def get_params(self): return {'W': self.W, 'b': self.b}
    def set_params(self, p): self.W = p['W']; self.b = p['b']
    def get_grads(self): return {'W': self.dW, 'b': self.db}
