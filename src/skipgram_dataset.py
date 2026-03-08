"""Skip-gram dataset for contrastive wave embedding training.

Generates (target, context) pairs from Wikipedia text with negative sampling.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from dataclasses import dataclass

from src.tokenizer import Vocab, tokenize


class SkipGramDataset(Dataset):
    """Generates skip-gram pairs from pre-tokenized corpus.

    Each item returns (target_id, positive_id, [negative_ids]).
    Negatives are sampled from unigram distribution^(3/4) (word2vec convention).
    """

    def __init__(
        self,
        token_ids: list[list[int]],
        vocab: Vocab,
        window_size: int = 5,
        num_negatives: int = 5,
        min_count: int = 1,
    ):
        self.vocab = vocab
        self.window_size = window_size
        self.num_negatives = num_negatives

        # Build skip-gram pairs
        self.pairs: list[tuple[int, int]] = []
        word_counts = np.zeros(vocab.size, dtype=np.float64)

        pad_id = vocab.word2idx.get("<pad>", 0)
        unk_id = vocab.word2idx.get("<unk>", 1)
        skip_ids = {pad_id, unk_id}

        for doc_ids in token_ids:
            for i, target in enumerate(doc_ids):
                if target in skip_ids:
                    continue
                word_counts[target] += 1
                # Random window size (word2vec style)
                actual_window = np.random.randint(1, window_size + 1)
                start = max(0, i - actual_window)
                end = min(len(doc_ids), i + actual_window + 1)
                for j in range(start, end):
                    if j == i:
                        continue
                    context = doc_ids[j]
                    if context in skip_ids:
                        continue
                    self.pairs.append((target, context))

        # Build negative sampling distribution: unigram^(3/4)
        word_counts = np.maximum(word_counts, 0)
        powered = np.power(word_counts, 0.75)
        # Zero out special tokens
        for sid in skip_ids:
            if sid < len(powered):
                powered[sid] = 0.0
        total = powered.sum()
        if total > 0:
            self.neg_probs = powered / total
        else:
            self.neg_probs = np.ones(vocab.size) / vocab.size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target, positive = self.pairs[idx]

        # Sample negatives
        negatives = np.random.choice(
            len(self.neg_probs),
            size=self.num_negatives,
            replace=True,
            p=self.neg_probs,
        )

        return (
            torch.tensor(target, dtype=torch.long),
            torch.tensor(positive, dtype=torch.long),
            torch.tensor(negatives, dtype=torch.long),
        )


class CharSkipGramDataset(Dataset):
    """Word-level skip-gram dataset where each word is represented as char IDs.

    Generates (target_word, context_word) pairs at the word level.
    Words are pre-padded to a global max length at init time so that
    collation is just tensor indexing — no Python loops per batch.
    """

    def __init__(
        self,
        word_sequences: list[list[list[int]]],
        window_size: int = 5,
        num_negatives: int = 5,
    ):
        self.num_negatives = num_negatives
        self.window_size = window_size

        # Build unique word index
        self.all_words: list[list[int]] = []
        word_to_idx: dict[tuple[int, ...], int] = {}
        word_counts: list[int] = []

        # Convert sentences to word-index arrays
        self.sentences: list[np.ndarray] = []
        for sentence_words in word_sequences:
            sentence_indices = []
            for word_chars in sentence_words:
                key = tuple(word_chars)
                if key not in word_to_idx:
                    word_to_idx[key] = len(self.all_words)
                    self.all_words.append(word_chars)
                    word_counts.append(0)
                idx = word_to_idx[key]
                word_counts[idx] += 1
                sentence_indices.append(idx)
            if len(sentence_indices) >= 2:
                self.sentences.append(np.array(sentence_indices, dtype=np.int32))

        # Pre-pad all words to global max length → (num_words, max_word_len) tensor
        self.max_word_len = max(len(w) for w in self.all_words)
        num_words = len(self.all_words)
        self.word_ids = torch.zeros(num_words, self.max_word_len, dtype=torch.long)
        self.word_masks = torch.zeros(num_words, self.max_word_len, dtype=torch.bool)
        for i, w in enumerate(self.all_words):
            self.word_ids[i, :len(w)] = torch.tensor(w, dtype=torch.long)
            self.word_masks[i, :len(w)] = True

        # Flat position index
        self.positions: list[tuple[int, int]] = []
        for si, sent in enumerate(self.sentences):
            for pos in range(len(sent)):
                self.positions.append((si, pos))

        # Negative sampling distribution: unigram^(3/4)
        counts = np.array(word_counts, dtype=np.float64)
        powered = np.power(counts, 0.75)
        total = powered.sum()
        self.neg_probs = powered / total if total > 0 else np.ones(len(counts)) / len(counts)

        # Pre-compute cumulative distribution for fast sampling
        self.neg_cum_probs = np.cumsum(self.neg_probs)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        si, pos = self.positions[idx]
        sent = self.sentences[si]
        target_word_idx = int(sent[pos])

        # Pick a random context word within window
        actual_window = np.random.randint(1, self.window_size + 1)
        start = max(0, pos - actual_window)
        end = min(len(sent), pos + actual_window + 1)
        context_positions = [j for j in range(start, end) if j != pos]
        if not context_positions:
            context_positions = [pos]
        ctx_pos = context_positions[np.random.randint(len(context_positions))]
        pos_word_idx = int(sent[ctx_pos])

        # Negative samples
        neg_indices = np.random.choice(
            len(self.neg_probs),
            size=self.num_negatives,
            replace=True,
            p=self.neg_probs,
        )

        # Return word indices — collate_fn will look up pre-padded tensors
        return target_word_idx, pos_word_idx, neg_indices

    def collate_fn(self, batch):
        """Fast collation via pre-padded word tensor indexing.

        Returns:
            target_chars: (batch, max_word_len) LongTensor
            pos_chars: (batch, max_word_len) LongTensor
            neg_chars: (batch, num_neg, max_word_len) LongTensor
            target_mask: (batch, max_word_len) BoolTensor
            pos_mask: (batch, max_word_len) BoolTensor
            neg_mask: (batch, num_neg, max_word_len) BoolTensor
        """
        targets, positives, negatives = zip(*batch)

        target_idx = torch.tensor(targets, dtype=torch.long)
        pos_idx = torch.tensor(positives, dtype=torch.long)
        neg_idx = torch.tensor(np.array(negatives), dtype=torch.long)  # (batch, num_neg)

        target_chars = self.word_ids[target_idx]      # (batch, max_word_len)
        target_mask = self.word_masks[target_idx]
        pos_chars = self.word_ids[pos_idx]
        pos_mask = self.word_masks[pos_idx]
        neg_chars = self.word_ids[neg_idx]             # (batch, num_neg, max_word_len)
        neg_mask = self.word_masks[neg_idx]

        return target_chars, pos_chars, neg_chars, target_mask, pos_mask, neg_mask


def tokenize_corpus(
    texts: list[str],
    vocab: Vocab,
) -> list[list[int]]:
    """Tokenize a list of texts into token ID sequences."""
    result = []
    for text in texts:
        tokens = tokenize(text)
        ids = [vocab.word2idx.get(t, vocab.word2idx["<unk>"]) for t in tokens]
        result.append(ids)
    return result
