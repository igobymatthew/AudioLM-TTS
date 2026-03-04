from transformers import GPT2Config, GPT2LMHeadModel


def build_model(vocab_size: int, cfg: dict | None = None) -> GPT2LMHeadModel:
    cfg = cfg or {}
    model_cfg = cfg.get('model', {})
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=model_cfg.get('n_layers', 6),
        n_head=model_cfg.get('n_heads', 8),
        n_embd=model_cfg.get('hidden_dim', 512),
        n_positions=model_cfg.get('max_positions', 1024),
        resid_pdrop=model_cfg.get('dropout', 0.1),
        embd_pdrop=model_cfg.get('dropout', 0.1),
        attn_pdrop=model_cfg.get('dropout', 0.1),
    )
    return GPT2LMHeadModel(config)
