"""WaveEmbeddingModel: end-to-end model for training wave embeddings on language tasks."""

import torch
import torch.nn as nn

from .wave_embedding import WaveTokenEmbedding


class WaveEmbeddingModel(nn.Module):
    """Complete model: wave embeddings -> sequence composition -> task head.

    Composes token spectral embeddings via interference (summation in Fourier space),
    then passes through a simple classifier head.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        num_waves: int = 7,
        signal_length: int = 1024,
        k_max: int = 16,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.wave_embedding = WaveTokenEmbedding(
            vocab_size=vocab_size,
            num_waves=num_waves,
            signal_length=signal_length,
            k_max=k_max,
        )

        embed_dim = self.wave_embedding.embed_dim

        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass: tokens -> wave embeddings -> interference -> classify.

        Args:
            token_ids: (batch, seq_len) tensor of token indices

        Returns:
            Logits of shape (batch, num_classes)
        """
        # Get per-token spectral embeddings: (batch, seq_len, embed_dim)
        spectral_embeds = self.wave_embedding(token_ids)

        # Compose via interference: sum spectra across sequence
        # This is equivalent to summing time-domain signals then taking FFT
        # (by linearity of FFT)
        composed = spectral_embeds.sum(dim=1)  # (batch, embed_dim)

        # Classify
        logits = self.classifier(composed)
        return logits
