from contextlib import contextmanager
from torch import nn


@contextmanager
def change_to_eval_mode_temporary(model: nn.Module):
    """Temporarily set the model to evaluation mode and then restore it."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()
