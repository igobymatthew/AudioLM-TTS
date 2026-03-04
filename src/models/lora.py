from peft import LoraConfig, TaskType, get_peft_model


def apply_lora(model, lora_cfg: dict):
    config = LoraConfig(
        r=lora_cfg.get('r', 8),
        lora_alpha=lora_cfg.get('alpha', 16),
        lora_dropout=lora_cfg.get('dropout', 0.05),
        target_modules=lora_cfg.get('target_modules', ['c_attn', 'c_proj']),
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, config)
