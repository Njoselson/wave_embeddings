"""Simple word-level tokenizer for toy experiments."""

from collections import Counter
from dataclasses import dataclass
import re
import string


@dataclass
class Vocab:
    word2idx: dict[str, int]
    idx2word: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.word2idx)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        tokens = tokenize(text)
        ids = [self.word2idx.get(t, self.word2idx["<unk>"]) for t in tokens]
        if max_len is not None:
            ids = ids[:max_len]
            ids += [self.word2idx["<pad>"]] * (max_len - len(ids))
        return ids


# --- Character-level vocabulary ---

_CHARS = (
    list(string.ascii_lowercase)
    + list(string.digits)
    + list(".,;:!?'\"()-/&@ ")
)


def build_char_vocab() -> Vocab:
    """Build a character-level vocabulary (~50 chars + <pad> + <unk>)."""
    specials = ["<pad>", "<unk>"]
    chars = specials + _CHARS
    word2idx = {c: i for i, c in enumerate(chars)}
    idx2word = {i: c for c, i in word2idx.items()}
    return Vocab(word2idx=word2idx, idx2word=idx2word)


def tokenize_words_to_chars(text: str, vocab: Vocab | None = None) -> list[list[int]]:
    """Split text into words and convert each word to a list of char IDs.

    Args:
        text: Input text.
        vocab: Character vocab. If None, builds one.

    Returns:
        List of words, each word is a list of char IDs.
    """
    if vocab is None:
        vocab = build_char_vocab()
    unk_id = vocab.word2idx["<unk>"]
    words = tokenize(text)
    result = []
    for word in words:
        char_ids = [vocab.word2idx.get(c, unk_id) for c in word]
        if char_ids:
            result.append(char_ids)
    return result


def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    return re.findall(r"\w+|[^\w\s]", text)


def build_vocab(texts: list[str], max_size: int = 10000, min_freq: int = 2) -> Vocab:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    specials = ["<pad>", "<unk>"]
    words = [w for w, c in counter.most_common(max_size - len(specials)) if c >= min_freq]

    word2idx = {w: i for i, w in enumerate(specials + words)}
    idx2word = {i: w for w, i in word2idx.items()}
    return Vocab(word2idx=word2idx, idx2word=idx2word)
