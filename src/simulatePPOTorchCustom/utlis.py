from typing import Tuple 
from tensordict import TensorDict
from torch import Tensor

def get_grad_norm(parameters: Tensor) -> float: 
    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def split_trajectories(trajectories: TensorDict, val_ratio: float) -> Tuple[TensorDict, TensorDict]: 
    index_threshold = int(trajectories.shape[0] * (1.0 - val_ratio)) 
    return trajectories[:index_threshold], trajectories[index_threshold:]