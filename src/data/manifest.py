import argparse
from pathlib import Path

from src.utils.io import write_jsonl

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}


def _iter_audio_files(data_dir: Path):
    for path in sorted(data_dir.rglob('*')):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTS:
            yield path


def build_manifest(data_dir: str, out: str, sample_rate: int = 24000, default_speaker: int | None = None) -> int:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f'data_dir does not exist: {root}')

    rows = []
    for audio_path in _iter_audio_files(root):
        text_path = audio_path.with_suffix('.txt')
        if not text_path.exists():
            continue

        row = {
            'audio_path': str(audio_path),
            'text_path': str(text_path),
            'sample_rate': int(sample_rate),
        }
        if default_speaker is not None:
            row['speaker_id'] = int(default_speaker)
        rows.append(row)

    if not rows:
        raise RuntimeError(f'No wav/txt pairs found under: {root}')

    write_jsonl(out, rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build manifest.jsonl from wav/txt pairs')
    parser.add_argument('--data_dir', required=True, help='Directory containing audio files and matching .txt transcripts')
    parser.add_argument('--out', required=True, help='Output manifest JSONL path')
    parser.add_argument('--sample_rate', type=int, default=24000)
    parser.add_argument('--default_speaker', type=int, default=None)
    args = parser.parse_args()

    n = build_manifest(args.data_dir, args.out, sample_rate=args.sample_rate, default_speaker=args.default_speaker)
    print(f'Wrote {n} entries to {args.out}')


if __name__ == '__main__':
    main()
