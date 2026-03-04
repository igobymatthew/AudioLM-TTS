import torch


def build_optimizer(model, lr: float):
    return torch.optim.AdamW(model.parameters(), lr=lr)
