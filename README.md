# RNN From Scratch — NumPy Only

A complete Recurrent Neural Network ecosystem — Vanilla RNN, LSTM, and GRU — implemented **entirely from scratch** using only NumPy. No TensorFlow, PyTorch, Keras, or JAX.

## Task: Character-Level Language Modeling on Shakespeare

The model reads Shakespeare text one character at a time and learns to predict the next character. After training it generates new Shakespeare-style text.

### Sample Generated Output (LSTM, Temperature=0.8)

```
First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
```

### Training Results

| Epoch | Loss | Perplexity |
|---|---|---|
| 1 | 1.6747 | 5.3 |
| 2 | 0.2525 | 1.3 |
| 5 | 0.1198 | 1.1 |
| 10 | 0.1070 | 1.1 |

## Project Structure

```
rnn_from_scratch/
│
├── data/
│   └── text_loader.py        # Shakespeare loader + CharDataset + batch generator
│
├── layers/
│   ├── rnn_cell.py           # Vanilla RNN — forward + BPTT backward
│   ├── lstm_cell.py          # LSTM (4 gates, cell state) — full BPTT
│   ├── gru_cell.py           # GRU (2 gates) — full BPTT
│   └── embedding.py          # EmbeddingLayer + LinearLayer
│
├── utils/
│   └── utils.py              # CrossEntropyLoss, SGD, Adam, gradient clipping, perplexity
│
├── models/
│   └── language_model.py     # Full LM: Embed → RNN/LSTM/GRU → Linear → Loss
│                             # + generate() for temperature-based text sampling
│
├── saved_weights/
│   └── lm_weights.npz        # Saved model weights
│
├── plots/                    # All 7 generated visualizations
│
├── train.py                  # Full training loop with all visualizations
├── generate.py               # Text generation CLI
├── report.md                 # Complete academic report with math
└── README.md
```

## Quick Start

```bash
pip install numpy matplotlib

# Train (LSTM by default)
python train.py

# Generate text
python generate.py --seed "First Citizen:" --temp 0.8 --length 300

# Compare temperatures
python generate.py --compare
```

## What's Implemented

| Component | File | Details |
|---|---|---|
| Vanilla RNN (fwd+BPTT) | `layers/rnn_cell.py` | BPTT with dh_next accumulation |
| LSTM (fwd+BPTT) | `layers/lstm_cell.py` | All 4 gates, cell state, fused W |
| GRU (fwd+BPTT) | `layers/gru_cell.py` | Update + reset gates, full BPTT |
| Embedding Layer | `layers/embedding.py` | Row lookup + scatter-add backward |
| Linear Projection | `layers/embedding.py` | (N,T,H)→(N,T,V) with backward |
| Cross-Entropy Loss | `utils/utils.py` | log-sum-exp stable, fused backward |
| SGD Optimizer | `utils/utils.py` | Mini-batch |
| Adam Optimizer | `utils/utils.py` | Per-layer state, bias correction |
| Gradient Clipping | `utils/utils.py` | Global L2 norm clipping |
| Perplexity | `utils/utils.py` | exp(cross-entropy) |
| Text Generation | `models/language_model.py` | Autoregressive + temperature sampling |
| Save/Load Weights | `models/language_model.py` | .npz format |
| 7 Visualizations | `train.py` | See plots/ directory |

## Mathematical Foundations

See `report.md` for complete derivations of:
- Vanilla RNN equations and BPTT step-by-step
- Vanishing/exploding gradient analysis (with eigenvalue bounds)
- All 4 LSTM gates with BPTT for every gate
- GRU equations and BPTT
- Why LSTM solves vanishing gradients (constant error carousel)
- Adam optimizer derivation with bias correction
- Gradient clipping (Pascanu et al., 2013)
- Temperature sampling mathematics

## Switching Cell Type

```python
# In train.py or your own script:
model = LanguageModel(vocab_size=47, embed_dim=32, hidden_dim=128,
                      cell_type='lstm')   # or 'rnn', 'gru'
```

## Visualizations (7 plots)

1. `training_curves.png` — Loss and perplexity over epochs
2. `gradient_norms.png` — Gradient norms with clipping threshold
3. `hidden_state_heatmap.png` — LSTM hidden activations over characters
4. `temperature_sampling.png` — Generated text at 5 temperatures
5. `embedding_similarity.png` — Character embedding cosine similarity
6. `char_prediction_probs.png` — Next-char prediction distributions
7. `architecture_comparison.png` — RNN vs LSTM vs GRU equations

## Allowed Libraries

- ✅ Python, NumPy, Matplotlib
- ❌ TensorFlow / Keras / PyTorch / JAX / any autograd
