import torch
import torch.nn as nn


class SpeakerConditioner(nn.Module):
    def __init__(self, n_speakers: int, hidden_dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_speakers, hidden_dim)

    def forward(self, speaker_ids: torch.Tensor) -> torch.Tensor:
        return self.emb(speaker_ids)
