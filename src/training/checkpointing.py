from pathlib import Path

import torch


def save_checkpoint(model, optimizer, step: int, out_dir: str) -> str:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / f'step_{step}.pt'
    latest = Path(out_dir) / 'latest.pt'
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }
    torch.save(payload, path)
    torch.save(payload, latest)
    return str(path)


def load_checkpoint(model, optimizer, ckpt_path: str):
    payload = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(payload['model'])
    if optimizer is not None and 'optimizer' in payload:
        optimizer.load_state_dict(payload['optimizer'])
    return payload.get('step', 0)
