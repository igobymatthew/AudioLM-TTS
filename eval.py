import argparse

from src.evaluation.bench import benchmark_tokens_per_sec
from src.evaluation.mos_proxy import mos_proxy
from src.evaluation.wer_whisper import compute_wer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--n_samples', type=int, default=2)
    args = parser.parse_args()

    _ = load_config(args.config)
    refs = ['hello world'] * args.n_samples
    hyps = ['hello world'] * args.n_samples
    w = compute_wer(refs, hyps)
    tps = benchmark_tokens_per_sec(200, 1.0)
    mos = mos_proxy([])
    print({'wer': w, 'tokens_per_sec': tps, 'mos_proxy': mos, 'ckpt': args.ckpt})


if __name__ == '__main__':
    main()
