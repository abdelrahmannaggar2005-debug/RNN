"""
layers/rnn_cell.py
==================
Vanilla RNN Cell — Implemented from Scratch with NumPy

═══════════════════════════════════════════════════════
MATHEMATICAL BACKGROUND
═══════════════════════════════════════════════════════

A Recurrent Neural Network processes sequences by maintaining a hidden state
h_t that is updated at every time step t:

    h_t = tanh( W_hh @ h_{t-1}  +  W_xh @ x_t  +  b_h )
    y_t = W_hy @ h_t  +  b_y          (output — optional at each step)

Tensor shapes:
    x_t  : (N, D)        — input at time t    (N=batch, D=input_dim)
    h_t  : (N, H)        — hidden state       (H=hidden_dim)
    W_xh : (D, H)        — input-to-hidden weights
    W_hh : (H, H)        — hidden-to-hidden weights  ← the recurrent weights
    b_h  : (H,)          — hidden bias
    W_hy : (H, V)        — hidden-to-output weights  (V=vocab_size / output_dim)
    b_y  : (V,)          — output bias
    y_t  : (N, V)        — output logits at time t

The key insight: h_t depends on h_{t-1}, which depends on h_{t-2}, etc.
This creates a computational graph that unfolds through time, enabling the
network to model sequential dependencies.

═══════════════════════════════════════════════════════
BACKPROPAGATION THROUGH TIME (BPTT)
═══════════════════════════════════════════════════════

Because the network is unrolled through T time steps, gradients must flow
backwards through time as well as through the standard layer connections.

For time step t, define:
    z_t = W_hh @ h_{t-1} + W_xh @ x_t + b_h   (pre-activation)
    h_t = tanh(z_t)

Loss is summed over all time steps:
    L = Σ_t  L_t

Gradient w.r.t. h_t has two sources:
    dL/dh_t = dL_t/dh_t  +  dL/dh_{t+1} * ∂h_{t+1}/∂h_t    (chain rule)

The second term is the "gradient from the future" — this is what makes
BPTT different from standard backprop.

    ∂h_{t+1}/∂h_t = W_hh^T * diag(1 - h_{t+1}²)    (tanh Jacobian × W_hh^T)

Full BPTT gradient update for W_hh:
    dL/dW_hh = Σ_t  dL/dh_t_incoming * h_{t-1}^T

where dL/dh_t_incoming = d_tanh_t * W_hh^T (from above, times tanh derivative).

═══════════════════════════════════════════════════════
VANISHING GRADIENT IN RNNs
═══════════════════════════════════════════════════════

When backpropagating through T steps, the gradient involves:

    dL/dh_0 ∝ Π_{t=1}^{T}  W_hh^T * diag(1 - h_t²)

Each tanh derivative is in (0,1], and if ||W_hh|| < 1, this product
vanishes exponentially. If ||W_hh|| > 1, it explodes.

This "long-range credit assignment problem" prevents vanilla RNNs from
learning dependencies longer than ~10-20 steps.
Solutions: LSTM (gating), GRU, gradient clipping, orthogonal init.
"""

import numpy as np


class RNNCell:
    """
    Vanilla RNN layer that processes a full sequence via BPTT.

    Usage:
        rnn = RNNCell(input_dim=D, hidden_dim=H)
        outputs, h_final = rnn.forward(X, h0)
        dX, dh0 = rnn.backward(d_outputs)
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Parameters
        ----------
        input_dim  : D — dimensionality of each input vector x_t
        hidden_dim : H — dimensionality of hidden state h_t
        """
        self.D = input_dim
        self.H = hidden_dim

        # ── Weight initialization ─────────────────────────────────────
        # Xavier/Glorot uniform for tanh networks:
        #   W ~ Uniform(-limit, limit),  limit = sqrt(6/(fan_in+fan_out))
        # This keeps the variance of activations roughly constant.
        lim_xh = np.sqrt(6.0 / (self.D + self.H))
        lim_hh = np.sqrt(6.0 / (self.H + self.H))

        self.W_xh = np.random.uniform(-lim_xh, lim_xh, (self.D, self.H))  # (D, H)
        self.W_hh = np.random.uniform(-lim_hh, lim_hh, (self.H, self.H))  # (H, H)
        self.b_h  = np.zeros(self.H)                                        # (H,)

        # Gradient buffers
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h  = np.zeros_like(self.b_h)

        # Cache for BPTT
        self._cache = None

    # ──────────────────────────────────────────────────────────────────
    # Forward Pass
    # ──────────────────────────────────────────────────────────────────
    def forward(self, X, h0=None):
        """
        Process a full sequence through the RNN.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T, D)
             Batch of N sequences, each of length T, each token dimension D
        h0 : np.ndarray, shape (N, H)  or  None
             Initial hidden state. Defaults to zeros.

        Returns
        -------
        H_out : np.ndarray, shape (N, T, H)
                Hidden states at every time step
        h_T   : np.ndarray, shape (N, H)
                Final hidden state (last time step)

        Step-by-step dimensions:
            x_t  : (N, D)
            z_t  : (N, D)@(D,H) + (N,H)@(H,H) + (H,) = (N, H)
            h_t  : tanh(z_t) → (N, H)
        """
        N, T, D = X.shape
        H = self.H

        if h0 is None:
            h0 = np.zeros((N, H))  # start from all-zeros state

        # Allocate output: hidden states at every time step
        H_out = np.zeros((N, T, H))

        # Cache: store x_t, h_{t-1}, h_t for every step (needed for BPTT)
        xs = np.zeros((T, N, D))    # inputs
        hs = np.zeros((T+1, N, H))  # hidden states, hs[0] = h0
        hs[0] = h0

        for t in range(T):
            x_t = X[:, t, :]          # (N, D) — input at time t

            # RNN equation:
            # z_t = x_t @ W_xh + h_{t-1} @ W_hh + b_h
            z_t = x_t @ self.W_xh + hs[t] @ self.W_hh + self.b_h  # (N, H)

            # tanh non-linearity: maps z_t to (-1, 1)
            h_t = np.tanh(z_t)  # (N, H)

            xs[t]    = x_t
            hs[t+1]  = h_t
            H_out[:, t, :] = h_t

        # Store for backward
        self._cache = (xs, hs, N, T, D, H)

        return H_out, hs[T]  # hidden states + final state

    # ──────────────────────────────────────────────────────────────────
    # Backward Pass (BPTT)
    # ──────────────────────────────────────────────────────────────────
    def backward(self, dH_out):
        """
        Backpropagation Through Time (BPTT).

        Parameters
        ----------
        dH_out : np.ndarray, shape (N, T, H)
            Upstream gradient of loss w.r.t. every hidden state output

        Returns
        -------
        dX  : np.ndarray, shape (N, T, D)  — gradient w.r.t. input sequence
        dh0 : np.ndarray, shape (N, H)     — gradient w.r.t. initial hidden state
        """
        xs, hs, N, T, D, H = self._cache

        # Reset gradient accumulators
        self.dW_xh = np.zeros_like(self.W_xh)
        self.dW_hh = np.zeros_like(self.W_hh)
        self.db_h  = np.zeros_like(self.b_h)
        dX         = np.zeros((N, T, D))

        # dh_next = gradient flowing from t+1 back into h_t
        # At t=T (last step), there's no future gradient → start at 0
        dh_next = np.zeros((N, H))

        for t in reversed(range(T)):
            # ── Current hidden state and its predecessor ──────────────
            h_t    = hs[t+1]   # h at time t   — shape (N, H)
            h_prev = hs[t]     # h at time t-1 — shape (N, H)
            x_t    = xs[t]     # input at t    — shape (N, D)

            # ── Total gradient flowing into h_t ───────────────────────
            # Two sources:
            #   1. dL_t/dh_t  — from the output loss at time t (e.g. language model loss)
            #   2. dh_next    — from the next time step via W_hh
            dh_t = dH_out[:, t, :] + dh_next   # (N, H)

            # ── Backprop through tanh ─────────────────────────────────
            # h_t = tanh(z_t)  →  dz_t = dh_t * (1 - h_t²)
            # This is the tanh derivative: tanh'(x) = 1 - tanh(x)²
            dtanh = dh_t * (1.0 - h_t ** 2)    # (N, H)

            # ── Gradient w.r.t. bias ──────────────────────────────────
            # z_t = ... + b_h  →  db_h += Σ_n dtanh[n,:]
            self.db_h += dtanh.sum(axis=0)      # (H,)

            # ── Gradient w.r.t. W_xh ─────────────────────────────────
            # z_t = x_t @ W_xh + ...  →  dW_xh += x_t^T @ dtanh
            self.dW_xh += x_t.T @ dtanh         # (D,H)

            # ── Gradient w.r.t. W_hh ─────────────────────────────────
            # z_t = ... + h_{t-1} @ W_hh  →  dW_hh += h_{t-1}^T @ dtanh
            self.dW_hh += h_prev.T @ dtanh      # (H,H)

            # ── Gradient w.r.t. input x_t ────────────────────────────
            # z_t = x_t @ W_xh + ...  →  dx_t = dtanh @ W_xh^T
            dX[:, t, :] = dtanh @ self.W_xh.T  # (N,D)

            # ── Gradient flowing to previous hidden state ─────────────
            # z_t = ... + h_{t-1} @ W_hh  →  dh_{t-1} = dtanh @ W_hh^T
            dh_next = dtanh @ self.W_hh.T      # (N,H)

        dh0 = dh_next  # gradient w.r.t. initial hidden state
        return dX, dh0

    def get_params(self):
        return {'W_xh': self.W_xh, 'W_hh': self.W_hh, 'b_h': self.b_h}

    def set_params(self, p):
        self.W_xh = p['W_xh']; self.W_hh = p['W_hh']; self.b_h = p['b_h']

    def get_grads(self):
        return {'W_xh': self.dW_xh, 'W_hh': self.dW_hh, 'b_h': self.db_h}
