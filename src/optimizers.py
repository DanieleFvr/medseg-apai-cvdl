import torch


def build_adam_optimizer(
    model,
    lr:float,
):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )