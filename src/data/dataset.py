from src.utils.io import read_jsonl
import torch
from torch.utils.data import Dataset


class TokenizedTTSDataset(Dataset):
    def __init__(self, manifest_path: str):
        self.rows = list(read_jsonl(manifest_path))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        tokens = torch.load(row['token_path'], map_location='cpu')
        if tokens.ndim > 1:
            tokens = tokens.reshape(-1)
        text = ''
        with open(row['text_path'], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        speaker_id = row.get('speaker_id', 0)
        return {'audio_tokens': tokens.long(), 'text': text, 'speaker_id': speaker_id}
