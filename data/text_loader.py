"""
data/text_loader.py
===================
Character-Level Language Modeling Dataset

We use a subset of Tiny Shakespeare (Karpathy, 2015) — a concatenation
of Shakespeare's works (~1MB of text). The task is character-level language
modeling: given a sequence of characters, predict the next character.

PREPROCESSING PIPELINE:
    1. Read raw text → string of characters
    2. Build vocabulary: set of unique chars → char-to-int mapping
    3. Encode text as integer sequence
    4. Slice into (input, target) pairs:
         input:  x[t] = token at position t
         target: y[t] = token at position t+1  (next-character prediction)
    5. Batch sequences into (N, T) arrays

WHY CHARACTER-LEVEL?
    - No need for tokenizer
    - Vocabulary is tiny (≈65 characters)
    - Model must learn spelling, grammar, and style from scratch
    - Great for demonstrating RNN capabilities with minimal setup

TEACHER FORCING:
    During training we feed the TRUE previous token as input at each step
    (not the model's prediction). This stabilizes training and is called
    "teacher forcing". During generation we feed the model's own predictions.
"""

import numpy as np
import os
import urllib.request


# Tiny Shakespeare text (embedded as a small sample if download fails)
TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

FALLBACK_TEXT = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you know Caius Marcius is chief enemy to the people.

All:
We know't, we know't.

First Citizen:
Let us kill him, and we'll have corn at our own price.
Is't a verdict?

All:
No more talking on't; let it be done: away, away!

Second Citizen:
One word, good citizens.

First Citizen:
We are accounted poor citizens, the patricians good.
What authority surfeits on would relieve us: if they
would yield us but the superfluity, while it were
wholesome, we might guess they relieved us humanely;
but they think we are too dear: the leanness that
afflicts us, the object of our misery, is as an
inventory to particularise their abundance; our
sufferance is a gain to them. Let us revenge this with
our pikes, ere we become rakes: for the gods know I
speak this in hunger for bread, not in thirst for revenge.

Second Citizen:
Would you proceed especially against Caius Marcius?

All:
Against him first: he's a very dog to the commonalty.

Second Citizen:
Consider you what services he has done for his country?

First Citizen:
Very well; and could be content to give him good
report for't, but that he pays himself with being proud.
"""


def load_text(data_dir='data', min_chars=50000):
    """
    Load or generate text dataset.

    Returns raw text string.
    """
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'shakespeare.txt')

    if os.path.exists(filepath) and os.path.getsize(filepath) > 10000:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[INFO] Loaded text: {len(text):,} characters")
        return text

    # Try downloading
    try:
        print("[INFO] Downloading Tiny Shakespeare...")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"[INFO] Downloaded: {len(text):,} characters")
        return text
    except Exception as e:
        print(f"[WARN] Download failed ({e}). Using built-in sample text.")

    # Use fallback and repeat to get enough data
    text = (FALLBACK_TEXT * 60).strip()
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"[INFO] Using fallback text: {len(text):,} characters")
    return text


class CharDataset:
    """
    Character-level language modeling dataset.

    Attributes
    ----------
    vocab      : list of unique characters (sorted)
    vocab_size : V = |vocab|
    char2idx   : dict mapping char → int
    idx2char   : dict mapping int → char
    data       : np.ndarray (int32) — full encoded text
    """

    def __init__(self, text, seq_len=50):
        """
        Parameters
        ----------
        text    : str — raw text
        seq_len : int — length of each training sequence (T)
        """
        self.seq_len = seq_len

        # Build vocabulary from unique characters
        self.vocab     = sorted(set(text))
        self.vocab_size = len(self.vocab)
        self.char2idx  = {c: i for i, c in enumerate(self.vocab)}
        self.idx2char  = {i: c for i, c in enumerate(self.vocab)}

        # Encode entire text as integer array
        self.data = np.array([self.char2idx[c] for c in text], dtype=np.int32)

        print(f"[INFO] Vocab size: {self.vocab_size} characters")
        print(f"[INFO] Total tokens: {len(self.data):,}")
        print(f"[INFO] Sequence length: {seq_len}")

    def get_batches(self, batch_size, shuffle=True):
        """
        Yield (X_batch, Y_batch) mini-batches for language model training.

        For each position i in the data:
            X[i] = data[i   : i+seq_len]      — input sequence
            Y[i] = data[i+1 : i+seq_len+1]    — target (shifted by 1)

        Parameters
        ----------
        batch_size : int
        shuffle    : bool

        Yields
        ------
        X_batch : np.ndarray (int32), shape (batch_size, seq_len)
        Y_batch : np.ndarray (int32), shape (batch_size, seq_len)
        """
        T  = self.seq_len
        N  = len(self.data) - T - 1   # number of valid start positions

        # Create all valid start indices
        starts = np.arange(N)
        if shuffle:
            np.random.shuffle(starts)

        for i in range(0, N - batch_size + 1, batch_size):
            batch_starts = starts[i : i + batch_size]
            X = np.stack([self.data[s     : s + T]   for s in batch_starts])
            Y = np.stack([self.data[s + 1 : s + T + 1] for s in batch_starts])
            yield X, Y

    def decode(self, indices):
        """Convert integer sequence back to string."""
        return ''.join(self.idx2char.get(int(i), '?') for i in indices)

    def encode(self, text):
        """Convert string to integer sequence."""
        return np.array([self.char2idx.get(c, 0) for c in text], dtype=np.int32)
