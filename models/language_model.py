"""
models/language_model.py
========================
Character-Level Language Model (RNN / LSTM / GRU)

Full pipeline:
    token_ids → Embedding → RNN/LSTM/GRU → Linear → logits → loss

Architecture:
    Input:   (N, T)  — integer token IDs
    Embed:   (N, T, E)        E = embed_dim
    RNN:     (N, T, H)        H = hidden_dim
    Linear:  (N, T, V)        V = vocab_size
    Loss:    scalar cross-entropy over all N*T token predictions
"""

import numpy as np
import os

from layers.embedding import EmbeddingLayer, LinearLayer
from layers.rnn_cell  import RNNCell
from layers.lstm_cell import LSTMCell
from layers.gru_cell  import GRUCell
from utils.utils      import CrossEntropyLoss


class LanguageModel:
    """
    Char-level LM with switchable RNN/LSTM/GRU backbone.

    Parameters
    ----------
    vocab_size  : V
    embed_dim   : E
    hidden_dim  : H
    cell_type   : 'rnn' | 'lstm' | 'gru'
    """

    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, cell_type='lstm'):
        self.vocab_size  = vocab_size
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.cell_type   = cell_type.lower()

        # ── Layers ────────────────────────────────────────────────────
        self.embedding = EmbeddingLayer(vocab_size, embed_dim)

        if self.cell_type == 'lstm':
            self.rnn = LSTMCell(embed_dim, hidden_dim)
        elif self.cell_type == 'gru':
            self.rnn = GRUCell(embed_dim, hidden_dim)
        else:  # vanilla rnn
            self.rnn = RNNCell(embed_dim, hidden_dim)

        self.output_proj = LinearLayer(hidden_dim, vocab_size)
        self.loss_fn     = CrossEntropyLoss()

        # List of parametric layers (for optimizer)
        self.param_layers = [self.embedding, self.rnn, self.output_proj]

        # Hidden state (persisted across batches for stateful training)
        self.h = None
        self.c = None   # LSTM only

    # ──────────────────────────────────────────────────────────────────
    # Forward Pass
    # ──────────────────────────────────────────────────────────────────
    def forward(self, token_ids, stateful=False):
        """
        Parameters
        ----------
        token_ids : np.ndarray (int32), shape (N, T)
        stateful  : bool — if True, carry hidden state across batches

        Returns
        -------
        logits : np.ndarray, shape (N*T, V)
        """
        N, T = token_ids.shape

        # Reset state if not stateful
        if not stateful or self.h is None:
            self.h = np.zeros((N, self.hidden_dim))
            self.c = np.zeros((N, self.hidden_dim))

        # Step 1 — Embedding lookup: (N, T) → (N, T, E)
        x_embed = self.embedding.forward(token_ids)

        # Step 2 — RNN: (N, T, E) → (N, T, H)
        if self.cell_type == 'lstm':
            H_out, h_T, c_T = self.rnn.forward(x_embed, self.h, self.c)
            if stateful:
                self.h = h_T.copy(); self.c = c_T.copy()
        else:
            H_out, h_T = self.rnn.forward(x_embed, self.h)
            if stateful:
                self.h = h_T.copy()

        # Step 3 — Project to vocab: (N, T, H) → (N, T, V)
        logits_3d = self.output_proj.forward(H_out)   # (N, T, V)

        # Flatten for loss: (N*T, V)
        return logits_3d.reshape(N * T, self.vocab_size)

    # ──────────────────────────────────────────────────────────────────
    # Loss
    # ──────────────────────────────────────────────────────────────────
    def compute_loss(self, logits, targets):
        """
        logits  : (N*T, V)
        targets : (N, T) — integer token IDs
        """
        return self.loss_fn.forward(logits, targets.reshape(-1))

    # ──────────────────────────────────────────────────────────────────
    # Backward Pass
    # ──────────────────────────────────────────────────────────────────
    def backward(self):
        """Full BPTT backward pass."""
        # Gradient from loss: (N*T, V)
        d_logits = self.loss_fn.backward()

        # Backprop through output projection: (N*T, V) → (N, T, H)
        N_T, V = d_logits.shape
        # Recover N and T from output_proj cache
        N = self.rnn._cache[-4]
        T = self.rnn._cache[-3]
        d_logits_3d = d_logits.reshape(N, T, V)
        d_H = self.output_proj.backward(d_logits_3d)   # (N, T, H)

        # Backprop through RNN: (N, T, H) → (N, T, E)
        if self.cell_type == 'lstm':
            d_embed, _, _ = self.rnn.backward(d_H)
        else:
            d_embed, _ = self.rnn.backward(d_H)

        # Backprop through embedding
        self.embedding.backward(d_embed)

    # ──────────────────────────────────────────────────────────────────
    # Text Generation (Sampling)
    # ──────────────────────────────────────────────────────────────────
    def generate(self, seed_text, char2idx, idx2char, length=200, temperature=1.0):
        """
        Autoregressively generate text one character at a time.

        Parameters
        ----------
        seed_text   : str — priming text
        char2idx    : dict
        idx2char    : dict
        length      : int — number of characters to generate
        temperature : float — controls randomness
            T=1.0: sample from model distribution
            T<1.0: sharper, more predictable output
            T>1.0: more random/creative output

        Temperature scaling:
            p_i = exp(logit_i / T) / Σ_j exp(logit_j / T)
        """
        # Reset state
        self.h = np.zeros((1, self.hidden_dim))
        self.c = np.zeros((1, self.hidden_dim))

        # Prime the state with seed text (run forward without saving grad)
        generated = list(seed_text)

        # Encode seed
        seed_ids = np.array([[char2idx.get(c, 0) for c in seed_text]], dtype=np.int32)

        # Run seed through model to build up hidden state
        if len(seed_text) > 0:
            embed = self.embedding.forward(seed_ids)   # (1, T_seed, E)
            if self.cell_type == 'lstm':
                H_out, h_T, c_T = self.rnn.forward(embed, self.h, self.c)
                self.h = h_T; self.c = c_T
            else:
                H_out, h_T = self.rnn.forward(embed, self.h)
                self.h = h_T

            # Use last hidden state to get first prediction
            logits = self.output_proj.forward(H_out[:, -1:, :])   # (1,1,V)
            last_token_logits = logits[0, 0]  # (V,)
        else:
            # No seed — start from random token
            last_token_logits = np.zeros(self.vocab_size)

        # Generate character by character
        for _ in range(length):
            # Apply temperature
            scaled_logits = last_token_logits / max(temperature, 1e-6)
            shifted = scaled_logits - scaled_logits.max()
            probs = np.exp(shifted) / np.exp(shifted).sum()

            # Sample from distribution
            next_id = np.random.choice(self.vocab_size, p=probs)
            generated.append(idx2char[next_id])

            # Feed sampled token back in as next input
            x = np.array([[next_id]], dtype=np.int32)   # (1, 1)
            embed = self.embedding.forward(x)             # (1, 1, E)

            if self.cell_type == 'lstm':
                H_out, self.h, self.c = self.rnn.forward(embed, self.h, self.c)
            else:
                H_out, self.h = self.rnn.forward(embed, self.h)

            logits = self.output_proj.forward(H_out[:, -1:, :])
            last_token_logits = logits[0, 0]

        return ''.join(generated)

    # ──────────────────────────────────────────────────────────────────
    # Save / Load
    # ──────────────────────────────────────────────────────────────────
    def save(self, path='saved_weights/lm_weights.npz'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {}

        # Embedding
        for k, v in self.embedding.get_params().items():
            data[f'embed_{k}'] = v

        # RNN params (different keys for different cell types)
        for k, v in self.rnn.get_params().items():
            data[f'rnn_{k}'] = v

        # Output projection
        for k, v in self.output_proj.get_params().items():
            data[f'proj_{k}'] = v

        np.savez(path, **data)
        print(f"[INFO] Saved weights to {path}")

    def load(self, path='saved_weights/lm_weights.npz'):
        d = np.load(path)

        self.embedding.set_params({k[6:]: d[k] for k in d if k.startswith('embed_')})
        self.rnn.set_params({k[4:]: d[k] for k in d if k.startswith('rnn_')})
        self.output_proj.set_params({k[5:]: d[k] for k in d if k.startswith('proj_')})

        print(f"[INFO] Loaded weights from {path}")
