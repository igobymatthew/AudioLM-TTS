import argparse

from src.inference.generate import generate_audio
from src.models.transformer import build_model
from src.models.vocab import CombinedVocab
from src.training.checkpointing import load_checkpoint
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--speaker_id', type=int, default=0)
    parser.add_argument('--out_wav', default='outputs/demo.wav')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    vocab = CombinedVocab()
    model = build_model(vocab.total_vocab_size, cfg)
    load_checkpoint(model, None, args.ckpt)
    path = generate_audio(model, args.text, args.out_wav, cfg, args.speaker_id)
    print(path)


if __name__ == '__main__':
    main()
