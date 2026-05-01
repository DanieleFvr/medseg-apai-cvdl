def ohem_warmup_schedule(
        epoch: int,
        warmup_active: bool,
        activation_epoch: int | None,
        final_weight: float,
) -> float:
    """
    This is a helper function that sets a schedule for neg_ohem_weight.
    """
    if warmup_active:  # TODO this is kind of crappy, I'll fix it if I have the time
        if epoch < activation_epoch:
            return 0.0
        elif epoch < activation_epoch + 2:
            return final_weight / 3
        elif epoch < activation_epoch + 4:
            return (final_weight / 3) * 2
        else:
            return final_weight
    else:
        return final_weight