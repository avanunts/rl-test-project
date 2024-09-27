from typing import List
from dataclasses import dataclass

@dataclass
class PolicyNetConfig: 
    observation_space_dim: int
    hidden_layers_dims: List[int]
    hidden_layers_activations: List[str]
    action_space_low: List[float]
    action_space_high: List[float]
    scale_min: float
    
@dataclass
class ValueNetConfig: 
    observation_space_dim: int
    hidden_layers_dims: List[int]
    hidden_layers_activations: List[str]

@dataclass
class ActorConfig: 
    policy_net_config: PolicyNetConfig
    clamp_log_prob: float

@dataclass
class CriticTrainConfig: 
    learning_rate: float
    num_epochs: int
    steps_per_epoch: int
    pct_start: float
    batch_size: int
    num_epochs: int
    max_grad_norm: float
