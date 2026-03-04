import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel
from encodec.utils import convert_audio

from src.utils.io import read_jsonl, write_jsonl


def load_encodec(bandwidth: float, device: str):
    try:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)
        model.to(device)
        model.eval()
        return model
    except Exception:
        return None


def _load_audio(path: str) -> tuple[torch.Tensor, int]:
    wav, sr = sf.read(path, always_2d=True)
    wav = torch.tensor(np.asarray(wav).T, dtype=torch.float32)
    return wav, sr


def _fallback_encode(wav: torch.Tensor) -> torch.Tensor:
    mono = wav.mean(dim=0)
    q = ((mono.clamp(-1, 1) + 1.0) * 511.5).round().long().clamp(0, 1023)
    return q


def encode_manifest(manifest_path: str, out_dir: str, out_manifest: str, bandwidth: float = 6.0, device: str = 'cpu') -> int:
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    model = load_encodec(bandwidth, device)

    rows_out = []
    for i, row in enumerate(read_jsonl(manifest_path)):
        wav, sr = _load_audio(row['audio_path'])
        if model is None:
            codes = _fallback_encode(wav)
            codec_name = 'fallback_quantizer'
        else:
            wav = convert_audio(wav, sr, 24000, model.channels).unsqueeze(0).to(device)
            with torch.no_grad():
                encoded = model.encode(wav)
            codes = torch.cat([frame[0] for frame in encoded], dim=-1).squeeze(0).cpu()
            codec_name = 'encodec_24khz'
        token_path = out_root / f'sample_{i:06d}.pt'
        torch.save(codes, token_path)

        row = dict(row)
        row['token_path'] = str(token_path)
        row['codec_bandwidth'] = bandwidth
        row['codec_name'] = codec_name
        rows_out.append(row)

    write_jsonl(out_manifest, rows_out)
    return len(rows_out)


def main() -> None:
    parser = argparse.ArgumentParser(description='Tokenize dataset audio into EnCodec token .pt files')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--out_manifest', required=True)
    parser.add_argument('--bandwidth', type=float, default=6.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    n = encode_manifest(args.manifest, args.out_dir, args.out_manifest, args.bandwidth, args.device)
    print(f'Encoded {n} samples -> {args.out_manifest}')


if __name__ == '__main__':
    main()
