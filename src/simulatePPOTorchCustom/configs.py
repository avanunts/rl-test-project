from typing import List, Dict, Tuple, Any
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
    pct_start: float
    batch_size: int
    max_grad_norm: float
    loss_fn_name: str
    advantage_bin_distribution_num_bins: int

@dataclass
class PPOLossConfig: 
    ppo_eps: float
    entropy_coeff: float

@dataclass
class LocScaleRegularizationConfig: 
    loc_soft_min: float
    loc_soft_max: float
    scale_soft_max: float

@dataclass
class ActorTrainConfig: 
    learning_rate: float
    pct_start: float
    batch_size: int
    max_grad_norm: float
    ppo_loss_config: PPOLossConfig
    loc_scale_regularization_config: LocScaleRegularizationConfig    

@dataclass
class ActorConfigUpdateOnMinEpisodeLength: 
    min_episode_length: int
    new_ppo_eps: float
    new_actor_lr: float 
    entropy_coeff_multiplier: float

@dataclass
class ActorTrainConfigSchedule: 
    max_entropy_to_objective_loss_ratio: float
    updates_on_episode_length: List[ActorConfigUpdateOnMinEpisodeLength]


@dataclass
class AgentConfig: 
    value_net_config: ValueNetConfig
    actor_config: ActorConfig
    critic_train_config: CriticTrainConfig
    actor_train_config: ActorTrainConfig
    val_ratio: float
    gamma: float
    critic_lr_multiplicators: List[float]
    actor_train_config_schedule: ActorTrainConfigSchedule
    

@dataclass
class SimulatorConfig: 
    agent_config: AgentConfig
    env_max_episode_steps: int
    target_num_frames_for_train_val: int 