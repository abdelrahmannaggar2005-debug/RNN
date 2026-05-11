"""
layers/gru_cell.py
==================
Gated Recurrent Unit (GRU) — Cho et al. (2014)

═══════════════════════════════════════════════════════
MATHEMATICAL BACKGROUND
═══════════════════════════════════════════════════════

The GRU is a simplified version of the LSTM that merges the forget and
input gates into a single "update gate", and eliminates the separate
cell state. It uses only 2 gates (vs 4 in LSTM) with fewer parameters.

━━━ EQUATIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Given input x_t (N,D) and previous hidden h_{t-1} (N,H):

    z_t = σ(W_z @ [x_t, h_{t-1}] + b_z)    — update gate ∈ (0,1)^H
    r_t = σ(W_r @ [x_t, h_{t-1}] + b_r)    — reset gate  ∈ (0,1)^H
    n_t = tanh(W_n @ [x_t, r_t ⊙ h_{t-1}] + b_n)  — candidate hidden
    h_t = (1 - z_t) ⊙ h_{t-1}  +  z_t ⊙ n_t

━━━ INTUITION ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Update gate z_t: How much of the new candidate to accept.
      z_t=1 → fully update to n_t (forget old hidden state)
      z_t=0 → keep old h_{t-1} unchanged

  Reset gate r_t: How much of the previous hidden state to use
      when computing the candidate n_t.
      r_t=0 → ignore h_{t-1}, compute n_t only from x_t (like fresh start)
      r_t=1 → use full h_{t-1} for the candidate

  The final update is an interpolation:
      h_t = (1-z_t)*h_{t-1} + z_t*n_t

━━━ WHY GRU vs LSTM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  GRU: fewer parameters (3 weight matrices vs 4), trains faster,
       often comparable accuracy on many tasks, especially with small data.
  LSTM: more expressive (separate c_t), often better on long sequences.
  In practice: try both and validate.

═══════════════════════════════════════════════════════
BACKPROPAGATION DERIVATIONS
═══════════════════════════════════════════════════════

Given dh_t (upstream gradient):

1. Backprop through h_t = (1-z_t)*h_{t-1} + z_t*n_t:
    dz_t   = dh_t * (n_t - h_{t-1})
    dn_t   = dh_t * z_t
    dh_{t-1} += dh_t * (1 - z_t)       ← partial (more below)

2. Backprop through n_t = tanh(W_n @ [x_t, r_t*h_{t-1}]):
    d_n_raw = dn_t * (1 - n_t^2)       ← tanh derivative
    dW_n   += [x_t, r_t*h_{t-1}]^T @ d_n_raw
    d_cat_n = d_n_raw @ W_n^T          shape (N, D+H)
    d_x_from_n   = d_cat_n[:, :D]
    d_rh_from_n  = d_cat_n[:, D:]      ← this is gradient w.r.t. r_t*h_{t-1}

3. Backprop through r_t*h_{t-1}:
    dr_t   = d_rh_from_n * h_{t-1}
    dh_{t-1} += d_rh_from_n * r_t      ← second partial

4. Backprop through z_t and r_t gates (sigmoid):
    dz_raw = dz_t * z_t * (1-z_t)
    dr_raw = dr_t * r_t * (1-r_t)
"""

import numpy as np


def sigmoid(x):
    return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))


class GRUCell:
    """GRU layer processing full sequences via BPTT."""

    def __init__(self, input_dim, hidden_dim):
        self.D = input_dim
        self.H = hidden_dim
        DH = input_dim + hidden_dim

        lim = np.sqrt(6.0 / (DH + hidden_dim))

        # Update gate weights
        self.W_z = np.random.uniform(-lim, lim, (DH, hidden_dim))
        self.b_z = np.zeros(hidden_dim)

        # Reset gate weights
        self.W_r = np.random.uniform(-lim, lim, (DH, hidden_dim))
        self.b_r = np.zeros(hidden_dim)

        # Candidate hidden state weights (note: input is [x_t, r_t*h_{t-1}])
        self.W_n = np.random.uniform(-lim, lim, (DH, hidden_dim))
        self.b_n = np.zeros(hidden_dim)

        # Gradients
        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_n = np.zeros_like(self.W_n)
        self.db_z = np.zeros_like(self.b_z)
        self.db_r = np.zeros_like(self.b_r)
        self.db_n = np.zeros_like(self.b_n)

        self._cache = None

    def forward(self, X, h0=None):
        N, T, D = X.shape; H = self.H
        if h0 is None: h0 = np.zeros((N, H))

        H_out = np.zeros((N, T, H))
        xs = np.zeros((T, N, D))
        hs = np.zeros((T+1, N, H)); hs[0] = h0
        zs  = np.zeros((T, N, H))   # update gates
        rs  = np.zeros((T, N, H))   # reset gates
        ns  = np.zeros((T, N, H))   # candidate hidden

        for t in range(T):
            x_t    = X[:, t, :]
            h_prev = hs[t]
            xh = np.concatenate([x_t, h_prev], axis=1)  # (N, D+H)

            z_t = sigmoid(xh @ self.W_z + self.b_z)    # update gate
            r_t = sigmoid(xh @ self.W_r + self.b_r)    # reset gate

            # Candidate: uses r_t to gate h_prev
            rh = np.concatenate([x_t, r_t * h_prev], axis=1)
            n_t = np.tanh(rh @ self.W_n + self.b_n)

            h_t = (1 - z_t) * h_prev + z_t * n_t

            xs[t] = x_t; zs[t] = z_t; rs[t] = r_t
            ns[t] = n_t; hs[t+1] = h_t
            H_out[:, t, :] = h_t

        self._cache = (xs, hs, zs, rs, ns, N, T, D, H)
        return H_out, hs[T]

    def backward(self, dH_out):
        xs, hs, zs, rs, ns, N, T, D, H = self._cache

        self.dW_z = np.zeros_like(self.W_z)
        self.dW_r = np.zeros_like(self.W_r)
        self.dW_n = np.zeros_like(self.W_n)
        self.db_z = np.zeros_like(self.b_z)
        self.db_r = np.zeros_like(self.b_r)
        self.db_n = np.zeros_like(self.b_n)
        dX = np.zeros((N, T, D))
        dh_next = np.zeros((N, H))

        for t in reversed(range(T)):
            x_t    = xs[t]; h_prev = hs[t]
            z_t = zs[t]; r_t = rs[t]; n_t = ns[t]

            dh_t = dH_out[:, t, :] + dh_next

            # h_t = (1-z)*h_prev + z*n_t
            dz_t    = dh_t * (n_t - h_prev)
            dn_t    = dh_t * z_t
            dh_prev = dh_t * (1 - z_t)          # partial

            # n_t = tanh(W_n @ [x_t, r_t*h_prev])
            dn_raw = dn_t * (1.0 - n_t**2)
            rh = np.concatenate([x_t, r_t * h_prev], axis=1)
            self.dW_n += rh.T @ dn_raw
            self.db_n += dn_raw.sum(axis=0)
            d_rh = dn_raw @ self.W_n.T           # (N, D+H)
            dX[:, t, :] += d_rh[:, :D]
            d_r_h_prev   = d_rh[:, D:]           # d(r_t * h_prev)
            dr_t   = d_r_h_prev * h_prev
            dh_prev += d_r_h_prev * r_t          # second partial

            # Gates backprop (sigmoid)
            xh = np.concatenate([x_t, h_prev], axis=1)
            dz_raw = dz_t * z_t * (1 - z_t)
            dr_raw = dr_t * r_t * (1 - r_t)
            self.dW_z += xh.T @ dz_raw; self.db_z += dz_raw.sum(axis=0)
            self.dW_r += xh.T @ dr_raw; self.db_r += dr_raw.sum(axis=0)
            d_xh_z = dz_raw @ self.W_z.T         # (N, D+H)
            d_xh_r = dr_raw @ self.W_r.T         # (N, D+H)
            dX[:, t, :] += d_xh_z[:, :D] + d_xh_r[:, :D]
            dh_prev += d_xh_z[:, D:] + d_xh_r[:, D:]

            dh_next = dh_prev

        return dX, dh_next

    def get_params(self):
        return {'W_z':self.W_z,'W_r':self.W_r,'W_n':self.W_n,
                'b_z':self.b_z,'b_r':self.b_r,'b_n':self.b_n}
    def set_params(self, p):
        self.W_z=p['W_z']; self.W_r=p['W_r']; self.W_n=p['W_n']
        self.b_z=p['b_z']; self.b_r=p['b_r']; self.b_n=p['b_n']
    def get_grads(self):
        return {'W_z':self.dW_z,'W_r':self.dW_r,'W_n':self.dW_n,
                'b_z':self.db_z,'b_r':self.db_r,'b_n':self.db_n}
