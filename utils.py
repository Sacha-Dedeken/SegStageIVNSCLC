# DEFINE VARIOUS TOOLS FOR TRACKING METRICS

import torch
import resource
import numpy as np
import random


class EarlyStopping:
    """
    EarlyStopping class to perform early stopping during training.

    Args:
        tolerance (int, optional): Number of epochs to wait without improvement. Defaults to 10.
        mode (str, optional): Whether to minimize ("min") or maximize ("max") the loss. Defaults to "min".
    """

    def __init__(self, tolerance: int = 10, mode: str = "min") -> None:
        """
        Initialize the EarlyStopping object.

        Args:
            tolerance (int, optional): Number of epochs to wait without improvement. Defaults to 10.
            mode (str, optional): Whether to minimize ("min") or maximize ("max") the loss. Defaults to "min".
        """
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.best_metric = "no metric"
        self.mode = mode

    def __call__(self, new_metric: float) -> None:
        """
        Perform early stopping check based on the new loss.

        Args:
            new_loss (float): New loss value.
        """
        if self.best_metric == "no metric":
            self.best_metric = new_metric
        else:
            if self.mode == "max":
                if new_metric > self.best_metric:
                    self.best_metric = new_metric
                    self.counter = 0
                else:
                    self.counter += 1
            else:
                if new_metric < self.best_metric:
                    self.best_metric = new_metric
                    self.counter = 0
                else:
                    self.counter += 1
            if self.counter >= self.tolerance:
                self.early_stop = True


def count_parameters(model):
    """
    Return the number of trainable parameters of the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_memory_usage(s=""):
    # Get CPU memory usage
    usage = resource.getrusage(resource.RUSAGE_SELF)
    memory_in_mb = usage.ru_maxrss / 1024  # Convert from KB to MB
    print(f"[{s}] CPU Memory Usage: {memory_in_mb:.2f} MB", flush=True)

    # Get GPU memory usage if a GPU is available
    if torch.cuda.is_available():
        gpu_memory_allocated = (
            torch.cuda.memory_allocated() / 1024**2
        )  # Convert from bytes to MB
        print(f"[{s}] GPU Memory Usage: {gpu_memory_allocated:.2f} MB", flush=True)


def save_random_state_dict():
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }


def load_random_state_dict(checkpoint):
    torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    torch.cuda.set_rng_state_all(
        [state.cpu() for state in checkpoint["cuda_rng_state"]]
    )
    np.random.set_state(checkpoint["numpy_rng_state"])
    random.setstate(checkpoint["python_rng_state"])

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()