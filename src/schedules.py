def ohem_warmup_schedule(
        round_id: int,
        epoch: int,
        activation_epoch: int,
) -> float:
    """
    This is a helper function that sets a schedule for neg_ohem_weight.
    """
    if round_id == 0:  # TODO maybe get rid of this
        if epoch < activation_epoch:
            return 0.0
        elif epoch < (activation_epoch + 2):
            return 0.01
        elif epoch < (activation_epoch + 4):
            return 0.035
        else:
            return 0.05
    else:
        return 0.05
