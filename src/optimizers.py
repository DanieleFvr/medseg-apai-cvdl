import torch


def build_adam_optimizer(
    model,
    lr:float,
):
    """
    This wrapper builds an adam optimizer for the given model.

    Args:
        model (torch.nn.Module): the model to be optimized.
        lr (float): the learning rate.

    Returns:
        torch.optim.Optimizer: the optimizer for the model.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )