"""
layers/lstm_cell.py
===================
Long Short-Term Memory (LSTM) — Implemented from Scratch with NumPy

═══════════════════════════════════════════════════════
MATHEMATICAL BACKGROUND
═══════════════════════════════════════════════════════

The LSTM (Hochreiter & Schmidhuber, 1997) solves the vanishing gradient
problem of vanilla RNNs by introducing a separate "cell state" c_t that
acts as a conveyor belt, allowing gradients to flow unchanged over many
time steps via an additive update (not multiplicative like tanh+W_hh).

━━━ THE FOUR GATES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At each time step t, given input x_t (D,) and previous hidden h_{t-1} (H,):

Define the concatenated input:
    z = [x_t, h_{t-1}]   shape: (D+H,)

Four linear transforms (computed simultaneously):
    [f_t]   [W_f]         [b_f]
    [i_t] = [W_i] @ z  +  [b_i]
    [g_t]   [W_g]         [b_g]
    [o_t]   [W_o]         [b_o]

where W_f, W_i, W_g, W_o each have shape (H, D+H).

Applied non-linearities:
    f_t = σ(W_f @ z + b_f)   — forget gate    ∈ (0,1)^H
    i_t = σ(W_i @ z + b_i)   — input gate     ∈ (0,1)^H
    g_t = tanh(W_g @ z + b_g) — cell gate     ∈ (-1,1)^H
    o_t = σ(W_o @ z + b_o)   — output gate   ∈ (0,1)^H

Cell state update (the "conveyor belt"):
    c_t = f_t ⊙ c_{t-1}  +  i_t ⊙ g_t
          ───────────────    ─────────────
          "forget old"       "write new"

Hidden state:
    h_t = o_t ⊙ tanh(c_t)

━━━ INTUITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Forget gate f_t: decides what to erase from cell state
      "If we see 'The cat...sat', forget the old subject when we see a new one."

  Input gate i_t: decides what new info to write
      "We just saw a new noun — write it into the relevant cell."

  Cell gate g_t: the actual new content to potentially write
      "Here is the new information, in (-1,1)."

  Output gate o_t: decides what to expose as hidden state
      "Which parts of the memory are relevant to predict the next token?"

━━━ WHY LSTMS DON'T VANISH ━━━━━━━━━━━━━━━━━━━━━━━━━━━

The gradient of the loss w.r.t. c_{t-k} is:
    dL/dc_{t-k} = dL/dc_t * Π_{j=1}^{k} f_{t-j+1}

The forget gates f ∈ (0,1) can be close to 1 for many steps, allowing
gradients to propagate through time nearly unchanged — the "constant error
carousel" (CEC). This contrasts with vanilla RNN where the chain involves
tanh derivatives (≤1) multiplied by W_hh repeatedly.

When f_t ≈ 1: gradient flows backward unchanged → long-range dependencies!
When f_t ≈ 0: gradient blocked → the network has truly "forgotten"

═══════════════════════════════════════════════════════
BACKPROPAGATION DERIVATIONS
═══════════════════════════════════════════════════════

Given upstream gradients dh_t and dc_t (from both the output loss and next step):

Step 1 — Backprop through output gate and tanh(c_t):
    d_tanhc = dh_t * o_t                       # through o_t ⊙ tanh(c_t)
    dc_t   += d_tanhc * (1 - tanh(c_t)^2)     # tanh derivative
    do_t    = dh_t * tanh(c_t)                 # through o_t

Step 2 — Backprop through cell state:
    df_t = dc_t * c_{t-1}                      # forget gate's contribution
    di_t = dc_t * g_t                          # input gate's contribution
    dg_t = dc_t * i_t                          # cell gate's contribution
    dc_{t-1} = dc_t * f_t                      # gradient to previous cell state

Step 3 — Backprop through gate non-linearities:
    d_f_raw = df_t * f_t * (1 - f_t)          # sigmoid derivative
    d_i_raw = di_t * i_t * (1 - i_t)
    d_g_raw = dg_t * (1 - g_t^2)              # tanh derivative
    d_o_raw = do_t * o_t * (1 - o_t)

Step 4 — Backprop through W (all gates share same input z = [x_t, h_{t-1}]):
    d_gates = concat(d_f_raw, d_i_raw, d_g_raw, d_o_raw)   shape: (N, 4H)
    dW      += d_gates.T @ z                                 shape: (4H, D+H)
    db      += d_gates.sum(axis=0)
    dz       = d_gates @ W                                   shape: (N, D+H)
    dx_t    = dz[:, :D]
    dh_{t-1} = dz[:, D:]
"""

import numpy as np


def sigmoid(x):
    """Numerically stable sigmoid."""
    # For x ≥ 0: 1/(1+e^{-x}). For x < 0: e^x/(1+e^x).
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class LSTMCell:
    """
    Long Short-Term Memory (LSTM) layer.

    Processes full sequences via BPTT.
    Uses fused weight matrix W of shape (D+H, 4H) for all 4 gates.
    """

    def __init__(self, input_dim, hidden_dim):
        """
        Parameters
        ----------
        input_dim  : D
        hidden_dim : H
        """
        self.D = input_dim
        self.H = hidden_dim

        # ── Fused weight matrix: all 4 gates in one matrix ─────────
        # W has shape (D+H, 4H). Columns are partitioned as [f | i | g | o]
        # Xavier uniform initialization for tanh/sigmoid networks
        lim = np.sqrt(6.0 / (input_dim + hidden_dim + hidden_dim))
        self.W = np.random.uniform(-lim, lim, (input_dim + hidden_dim, 4 * hidden_dim))
        self.b = np.zeros(4 * hidden_dim)  # (4H,)

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # BPTT cache
        self._cache = None

    # ──────────────────────────────────────────────────────────────────
    # Forward Pass
    # ──────────────────────────────────────────────────────────────────
    def forward(self, X, h0=None, c0=None):
        """
        Forward pass through full sequence.

        Parameters
        ----------
        X  : np.ndarray, shape (N, T, D)
        h0 : np.ndarray, shape (N, H)  or None
        c0 : np.ndarray, shape (N, H)  or None

        Returns
        -------
        H_out : np.ndarray, shape (N, T, H)  — hidden states at all time steps
        h_T   : np.ndarray, shape (N, H)     — final hidden state
        c_T   : np.ndarray, shape (N, H)     — final cell state
        """
        N, T, D = X.shape
        H = self.H

        if h0 is None: h0 = np.zeros((N, H))
        if c0 is None: c0 = np.zeros((N, H))

        # Storage for BPTT cache
        H_out = np.zeros((N, T, H))

        # Store intermediate values at each time step
        zs  = np.zeros((T, N, D + H))   # concatenated inputs [x_t, h_{t-1}]
        fs  = np.zeros((T, N, H))        # forget gate activations
        is_ = np.zeros((T, N, H))        # input gate activations
        gs  = np.zeros((T, N, H))        # cell gate activations
        os  = np.zeros((T, N, H))        # output gate activations
        cs  = np.zeros((T+1, N, H))      # cell states  cs[0] = c0
        hs  = np.zeros((T+1, N, H))      # hidden states hs[0] = h0
        cs[0] = c0
        hs[0] = h0

        for t in range(T):
            x_t    = X[:, t, :]          # (N, D)
            h_prev = hs[t]               # (N, H)
            c_prev = cs[t]               # (N, H)

            # Concatenate input and previous hidden state: (N, D+H)
            z_t = np.concatenate([x_t, h_prev], axis=1)

            # All 4 gates in one matrix multiply: (N, D+H) @ (D+H, 4H) = (N, 4H)
            gates_raw = z_t @ self.W + self.b  # (N, 4H)

            # Partition into 4 gates (each of size H)
            f_raw = gates_raw[:, 0*H : 1*H]
            i_raw = gates_raw[:, 1*H : 2*H]
            g_raw = gates_raw[:, 2*H : 3*H]
            o_raw = gates_raw[:, 3*H : 4*H]

            f_t = sigmoid(f_raw)        # forget gate — (N, H)
            i_t = sigmoid(i_raw)        # input gate  — (N, H)
            g_t = np.tanh(g_raw)        # cell gate   — (N, H)
            o_t = sigmoid(o_raw)        # output gate — (N, H)

            # Cell state update (additive — this is why gradients don't vanish)
            c_t = f_t * c_prev + i_t * g_t   # (N, H)

            # Hidden state
            h_t = o_t * np.tanh(c_t)         # (N, H)

            # Store
            zs[t] = z_t; fs[t] = f_t; is_[t] = i_t
            gs[t] = g_t; os[t] = o_t
            cs[t+1] = c_t; hs[t+1] = h_t
            H_out[:, t, :] = h_t

        self._cache = (zs, fs, is_, gs, os, cs, hs, N, T, D, H)
        return H_out, hs[T], cs[T]

    # ──────────────────────────────────────────────────────────────────
    # Backward Pass (BPTT)
    # ──────────────────────────────────────────────────────────────────
    def backward(self, dH_out, dh_next=None, dc_next=None):
        """
        BPTT through full sequence.

        Parameters
        ----------
        dH_out   : np.ndarray, shape (N, T, H)  — gradient w.r.t. all hidden outputs
        dh_next  : np.ndarray, shape (N, H)      — gradient from next layer/step (optional)
        dc_next  : np.ndarray, shape (N, H)      — gradient of cell state (optional)

        Returns
        -------
        dX   : np.ndarray, shape (N, T, D)
        dh0  : np.ndarray, shape (N, H)
        dc0  : np.ndarray, shape (N, H)
        """
        zs, fs, is_, gs, os, cs, hs, N, T, D, H = self._cache

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dX  = np.zeros((N, T, D))

        dh_t  = dh_next if dh_next is not None else np.zeros((N, H))
        dc_t  = dc_next if dc_next is not None else np.zeros((N, H))

        for t in reversed(range(T)):
            h_t    = hs[t+1]   # (N, H)
            c_t    = cs[t+1]   # (N, H)
            c_prev = cs[t]     # (N, H)
            f_t    = fs[t]     # (N, H)
            i_t    = is_[t]    # (N, H)
            g_t    = gs[t]     # (N, H)
            o_t    = os[t]     # (N, H)
            z_t    = zs[t]     # (N, D+H)

            # Total gradient into h_t from output loss + future step
            dh_total = dH_out[:, t, :] + dh_t   # (N, H)

            # ── Backprop through h_t = o_t * tanh(c_t) ───────────────
            tanh_ct = np.tanh(c_t)               # (N, H)
            do_t    = dh_total * tanh_ct         # (N, H)
            dc_t   += dh_total * o_t * (1.0 - tanh_ct**2)  # tanh derivative

            # ── Backprop through c_t = f_t*c_prev + i_t*g_t ──────────
            df_t   = dc_t * c_prev               # (N, H)
            dc_prev = dc_t * f_t                 # gradient to previous cell state
            di_t   = dc_t * g_t                  # (N, H)
            dg_t   = dc_t * i_t                  # (N, H)

            # ── Backprop through gate non-linearities ─────────────────
            # Sigmoid derivative: σ'(x) = σ(x)*(1-σ(x))
            # tanh derivative: tanh'(x) = 1-tanh(x)^2
            df_raw = df_t * f_t * (1.0 - f_t)   # (N, H)
            di_raw = di_t * i_t * (1.0 - i_t)   # (N, H)
            dg_raw = dg_t * (1.0 - g_t**2)      # (N, H)  ← tanh
            do_raw = do_t * o_t * (1.0 - o_t)   # (N, H)

            # Stack gate gradients: (N, 4H)
            d_gates = np.concatenate([df_raw, di_raw, dg_raw, do_raw], axis=1)

            # ── Gradient w.r.t. W and b ───────────────────────────────
            # gates_raw = z_t @ W + b
            self.dW += z_t.T @ d_gates           # (D+H, 4H)
            self.db += d_gates.sum(axis=0)       # (4H,)

            # ── Gradient w.r.t. z_t = [x_t, h_{t-1}] ────────────────
            dz_t = d_gates @ self.W.T            # (N, D+H)
            dX[:, t, :] = dz_t[:, :D]           # (N, D)
            dh_t = dz_t[:, D:]                   # (N, H) → flows to prev step
            dc_t = dc_prev                        # cell gradient to prev step

        return dX, dh_t, dc_t  # dh0, dc0

    def get_params(self): return {'W': self.W, 'b': self.b}
    def set_params(self, p): self.W = p['W']; self.b = p['b']
    def get_grads(self): return {'W': self.dW, 'b': self.db}
