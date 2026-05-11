"""
utils/utils.py — Loss, Optimizers, Gradient Clipping, Metrics
"""
import numpy as np


class CrossEntropyLoss:
    def __init__(self, epsilon=1e-12):
        self.epsilon = epsilon
        self._probs = None; self._labels = None; self._N = None

    def forward(self, logits, labels):
        logits_2d = logits.reshape(-1, logits.shape[-1])
        labels_1d = labels.reshape(-1)
        M = logits_2d.shape[0]
        shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
        exp_z   = np.exp(shifted)
        probs   = exp_z / exp_z.sum(axis=1, keepdims=True)
        probs   = np.clip(probs, self.epsilon, 1.0)
        self._probs = probs; self._labels = labels_1d; self._N = M
        return float(-np.mean(np.log(probs[np.arange(M), labels_1d])))

    def backward(self):
        p = self._probs.copy()
        p[np.arange(self._N), self._labels] -= 1.0
        return p / self._N


class SGD:
    def __init__(self, lr=0.01): self.lr = lr
    def update(self, layer):
        p = layer.get_params(); g = layer.get_grads()
        layer.set_params({k: p[k] - self.lr * g[k] for k in p})


class Adam:
    """
    Adam optimizer with per-layer unique state to avoid key collisions.
    State dict keys: (layer_id, param_key) tuples.
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr=lr; self.b1=beta1; self.b2=beta2; self.eps=eps
        self._state = {}   # (layer_id, key) -> (m, v, t)

    def update(self, layer):
        lid = id(layer)
        params = layer.get_params()
        grads  = layer.get_grads()
        updated = {}

        for key in params:
            sid = (lid, key)
            if sid not in self._state:
                self._state[sid] = {
                    'm': np.zeros_like(params[key]),
                    'v': np.zeros_like(params[key]),
                    't': 0
                }
            s = self._state[sid]
            s['t'] += 1
            g = grads[key]
            s['m'] = self.b1 * s['m'] + (1 - self.b1) * g
            s['v'] = self.b2 * s['v'] + (1 - self.b2) * g**2
            m_hat  = s['m'] / (1 - self.b1**s['t'])
            v_hat  = s['v'] / (1 - self.b2**s['t'])
            updated[key] = params[key] - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

        layer.set_params(updated)


def clip_gradients(layers, max_norm=5.0):
    total_sq = 0.0; all_grads = []
    for layer in layers:
        for g in layer.get_grads().values():
            if g is not None:
                total_sq += np.sum(g**2)
                all_grads.append(g)
    global_norm = np.sqrt(total_sq)
    if global_norm > max_norm:
        scale = max_norm / (global_norm + 1e-8)
        for g in all_grads: g *= scale
    return float(global_norm)


def perplexity(loss): return float(np.exp(min(loss, 50)))

def token_accuracy(logits, labels):
    preds = np.argmax(logits.reshape(-1, logits.shape[-1]), axis=1)
    return float(np.mean(preds == labels.reshape(-1)))
