"""
train.py
========
Training Script — Character-Level Language Model

Trains an LSTM (or RNN/GRU) on character-level text prediction.
Logs loss, perplexity, generates sample text after each epoch,
and produces full visualizations.
"""

import numpy as np
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.text_loader import CharDataset, load_text
from models.language_model import LanguageModel
from utils.utils import SGD, Adam, clip_gradients, perplexity, token_accuracy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════

CELL_TYPE   = 'lstm'     # 'rnn', 'lstm', or 'gru'
EMBED_DIM   = 32
HIDDEN_DIM  = 128
SEQ_LEN     = 50
BATCH_SIZE  = 64
EPOCHS      = 20
LR          = 0.002
GRAD_CLIP   = 5.0
SAVE_PATH   = 'saved_weights/lm_weights.npz'
PLOT_DIR    = 'plots'


# ══════════════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════════════

def plot_training_curves(train_losses, train_ppls, save_dir=PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', lw=2, ms=5)
    ax1.set_xlabel('Epoch', fontsize=12); ax1.set_ylabel('Cross-Entropy Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_ppls, 'r-s', lw=2, ms=5)
    ax2.set_xlabel('Epoch', fontsize=12); ax2.set_ylabel('Perplexity', fontsize=12)
    ax2.set_title('Perplexity (lower = better)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_gradient_norms(grad_norms, save_dir=PLOT_DIR):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(grad_norms, 'g-', lw=1, alpha=0.7, label='Gradient norm')
    ax.axhline(GRAD_CLIP, color='r', linestyle='--', label=f'Clip threshold ({GRAD_CLIP})')
    ax.set_xlabel('Training Step', fontsize=12); ax.set_ylabel('||grad||', fontsize=12)
    ax.set_title('Gradient Norms During Training', fontsize=14, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'gradient_norms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_loss_vs_temperature(model, dataset, save_dir=PLOT_DIR):
    """Show how temperature affects generated text distribution."""
    os.makedirs(save_dir, exist_ok=True)
    temps = [0.3, 0.7, 1.0, 1.5, 2.0]
    seed = "First Citizen:\n"

    fig, axes = plt.subplots(len(temps), 1, figsize=(10, len(temps)*1.6))
    for i, T in enumerate(temps):
        text = model.generate(seed, dataset.char2idx, dataset.idx2char,
                              length=80, temperature=T)
        # Only show generated part (after seed)
        gen = text[len(seed):]
        axes[i].text(0.02, 0.5, f"T={T}: {repr(gen[:70])}",
                     transform=axes[i].transAxes, fontsize=9,
                     verticalalignment='center', fontfamily='monospace',
                     wrap=True)
        axes[i].axis('off')

    plt.suptitle('Generated Text at Different Temperatures\n(Lower T = more predictable)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'temperature_sampling.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_hidden_state_heatmap(model, dataset, text_sample, save_dir=PLOT_DIR):
    """
    Visualize hidden state activations over a text sample.
    Reveals how the RNN's memory evolves as it reads characters.
    """
    os.makedirs(save_dir, exist_ok=True)
    text = text_sample[:80]
    ids  = np.array([[dataset.char2idx.get(c, 0) for c in text]], dtype=np.int32)

    model.h = np.zeros((1, model.hidden_dim))
    model.c = np.zeros((1, model.hidden_dim))
    embed = model.embedding.forward(ids)

    if model.cell_type == 'lstm':
        H_out, _, _ = model.rnn.forward(embed, model.h, model.c)
    else:
        H_out, _ = model.rnn.forward(embed, model.h)

    # H_out: (1, T, H) → select first 40 hidden units for display
    hidden = H_out[0, :, :40].T   # (40, T)

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(hidden, cmap='RdBu_r', aspect='auto',
                   vmin=-1, vmax=1, interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('Character Position', fontsize=11)
    ax.set_ylabel('Hidden Unit Index', fontsize=11)
    ax.set_title('Hidden State Activations Over Input Sequence', fontsize=13, fontweight='bold')

    # Label x-axis with characters (every 5)
    tick_positions = range(0, len(text), 5)
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels([repr(text[i])[1:-1] for i in tick_positions], fontsize=7, rotation=45)

    plt.tight_layout()
    path = os.path.join(save_dir, 'hidden_state_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_embedding_similarity(model, dataset, save_dir=PLOT_DIR):
    """
    Show cosine similarity between learned character embeddings.
    Similar characters should cluster together.
    """
    os.makedirs(save_dir, exist_ok=True)
    E = model.embedding.weight   # (V, E)

    # Compute cosine similarity matrix
    norms  = np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    E_norm = E / norms
    sim    = E_norm @ E_norm.T  # (V, V)

    vocab = dataset.vocab
    V     = len(vocab)

    fig, ax = plt.subplots(figsize=(max(8, V//4), max(7, V//4)))
    im = ax.imshow(sim, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(V)); ax.set_yticks(range(V))
    ax.set_xticklabels([repr(c)[1:-1] for c in vocab], fontsize=5, rotation=90)
    ax.set_yticklabels([repr(c)[1:-1] for c in vocab], fontsize=5)
    ax.set_title('Character Embedding Cosine Similarity', fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'embedding_similarity.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {path}")


def plot_char_predictions(model, dataset, save_dir=PLOT_DIR):
    """
    For a sample text, show predicted probability distribution
    over the next character at each position.
    """
    os.makedirs(save_dir, exist_ok=True)
    text = "First Citizen:\nBefore we proceed"
    ids  = np.array([[dataset.char2idx.get(c,0) for c in text]], dtype=np.int32)

    model.h = np.zeros((1, model.hidden_dim))
    model.c = np.zeros((1, model.hidden_dim))
    logits  = model.forward(ids)                      # (T, V)
    probs   = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  /= probs.sum(axis=1, keepdims=True)

    T_show  = min(20, len(text))
    V_show  = dataset.vocab_size

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(probs[:T_show].T, cmap='Blues', aspect='auto',
                   vmin=0, interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('Input Character Position', fontsize=11)
    ax.set_ylabel('Predicted Next Character', fontsize=11)
    ax.set_title('Next-Character Prediction Probabilities', fontsize=13, fontweight='bold')

    ax.set_xticks(range(T_show))
    ax.set_xticklabels([repr(text[i])[1:-1] for i in range(T_show)], fontsize=8, rotation=45)
    ax.set_yticks(range(V_show))
    ax.set_yticklabels([repr(c)[1:-1] for c in dataset.vocab], fontsize=5)

    plt.tight_layout()
    path = os.path.join(save_dir, 'char_prediction_probs.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] Saved {path}")


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train():
    np.random.seed(42)

    print("=" * 60)
    print(f"  RNN FROM SCRATCH — Char-Level LM ({CELL_TYPE.upper()})")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────
    text    = load_text(data_dir='data')
    dataset = CharDataset(text, seq_len=SEQ_LEN)

    # ── Build model ──────────────────────────────────────────────────
    model = LanguageModel(
        vocab_size  = dataset.vocab_size,
        embed_dim   = EMBED_DIM,
        hidden_dim  = HIDDEN_DIM,
        cell_type   = CELL_TYPE,
    )

    total_params = (
        dataset.vocab_size * EMBED_DIM +           # embedding
        (EMBED_DIM + HIDDEN_DIM) * HIDDEN_DIM * 4 +  # lstm (approx)
        HIDDEN_DIM * dataset.vocab_size             # output
    )
    print(f"\n[INFO] Model: {CELL_TYPE.upper()}, embed={EMBED_DIM}, "
          f"hidden={HIDDEN_DIM}, vocab={dataset.vocab_size}")
    print(f"[INFO] Approx parameters: {total_params:,}")

    # Adam optimizer — much better than SGD for language models
    optimizer = Adam(lr=LR)

    train_losses = []
    train_ppls   = []
    grad_norms   = []
    best_loss    = float('inf')

    print(f"\n[INFO] Training for {EPOCHS} epochs...\n")

    for epoch in range(1, EPOCHS + 1):
        epoch_losses = []
        t0 = time.time()
        step = 0

        for X_batch, Y_batch in dataset.get_batches(BATCH_SIZE, shuffle=True):
            # Forward
            logits = model.forward(X_batch)                  # (N*T, V)
            loss   = model.compute_loss(logits, Y_batch)

            # Backward
            model.backward()

            # Gradient clipping (essential for RNNs!)
            gn = clip_gradients(model.param_layers, max_norm=GRAD_CLIP)
            grad_norms.append(gn)

            # Update
            for layer in model.param_layers:
                optimizer.update(layer)

            epoch_losses.append(loss)
            step += 1

        mean_loss = float(np.mean(epoch_losses))
        ppl       = perplexity(mean_loss)
        dt        = time.time() - t0

        train_losses.append(mean_loss)
        train_ppls.append(ppl)

        print(f"[Epoch {epoch:2d}/{EPOCHS}]  "
              f"Loss: {mean_loss:.4f}  |  PPL: {ppl:.1f}  |  "
              f"Steps: {step}  |  Time: {dt:.1f}s")

        # Save best model
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save(SAVE_PATH)
            print(f"  ✓ Best model saved (loss={mean_loss:.4f})")

        # Generate sample text every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS:
            sample = model.generate(
                seed_text   = "First Citizen:\n",
                char2idx    = dataset.char2idx,
                idx2char    = dataset.idx2char,
                length      = 120,
                temperature = 0.8,
            )
            print(f"\n  ── Sample (T=0.8) ──────────────────────────────────")
            print(f"  {sample[:120].replace(chr(10), ' ↵ ')}")
            print()

    # ── Final evaluation ────────────────────────────────────────────
    model.load(SAVE_PATH)
    print("\n" + "=" * 60)
    print(f"  Training complete! Best loss: {best_loss:.4f} | PPL: {perplexity(best_loss):.1f}")
    print("=" * 60)

    # ── Plots ────────────────────────────────────────────────────────
    plot_training_curves(train_losses, train_ppls)
    plot_gradient_norms(grad_norms)
    plot_loss_vs_temperature(model, dataset)
    plot_hidden_state_heatmap(model, dataset, "First Citizen:\nBefore we proceed")
    plot_embedding_similarity(model, dataset)
    plot_char_predictions(model, dataset)

    # Save results
    os.makedirs('data', exist_ok=True)
    np.savez('data/training_results.npz',
             train_losses=np.array(train_losses),
             train_ppls=np.array(train_ppls),
             grad_norms=np.array(grad_norms[:500]))

    print(f"\n[INFO] All plots saved to '{PLOT_DIR}/'")
    print("[DONE]")

    return model, dataset


if __name__ == '__main__':
    train()
