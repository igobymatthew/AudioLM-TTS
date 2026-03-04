import argparse

from src.models.transformer import build_model
from src.models.vocab import CombinedVocab
from src.models.lora import apply_lora
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    if args.max_steps is not None:
        cfg['training']['max_steps'] = args.max_steps

    set_seed(int(cfg.get('seed', 42)))
    vocab = CombinedVocab()
    model = build_model(vocab.total_vocab_size, cfg)
    if 'lora' in cfg:
        model = apply_lora(model, cfg['lora'])
    trainer = Trainer(model, cfg)
    trainer.train(cfg['paths']['token_manifest'])


if __name__ == '__main__':
    main()
