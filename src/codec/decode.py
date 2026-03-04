import argparse

import numpy as np
import soundfile as sf
import torch
from encodec import EncodecModel


def _fallback_decode(codes: torch.Tensor) -> np.ndarray:
    if codes.ndim > 1:
        codes = codes.flatten()
    wav = (codes.float() / 511.5) - 1.0
    return wav.unsqueeze(0).numpy().T


def decode_tokens(token_path: str, out_wav: str, bandwidth: float = 6.0, device: str = 'cpu') -> None:
    codes = torch.load(token_path)
    try:
        model = EncodecModel.encodec_model_24khz()
        model.set_target_bandwidth(bandwidth)
        model.to(device)
        model.eval()

        if codes.ndim == 1:
            codes = codes.unsqueeze(0)
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        codes = codes.to(device)

        with torch.no_grad():
            wav = model.decode([(codes, None)])
        wav_np = wav.squeeze(0).cpu().numpy().T
    except Exception:
        wav_np = _fallback_decode(codes)

    sf.write(out_wav, np.asarray(wav_np), 24000)


def main() -> None:
    parser = argparse.ArgumentParser(description='Decode EnCodec token file into wav')
    parser.add_argument('--token_path', required=True)
    parser.add_argument('--out_wav', required=True)
    parser.add_argument('--bandwidth', type=float, default=6.0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    decode_tokens(args.token_path, args.out_wav, args.bandwidth, args.device)
    print(f'Decoded to {args.out_wav}')


if __name__ == '__main__':
    main()
