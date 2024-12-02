from dataclasses import dataclass


@dataclass
class LearningParams:
    gamma: float
    clip_epsilon: float
    entropy_coef: float
    batch_size: int
    num_epochs: int
    lr: float
    cosine_lr_min_ratio: float
    max_grad_norm: float
    loss_critic_type: str
    ciric_loss_coef: float = 1.0
    samples_mc_entropy: int = 10


@dataclass
class SimulatorParams:
    env_name: str
    device: str
    policy_layers_dims: list[int]
    policy_activation_func_type: str
    value_layers_dims: list[int]
    value_activation_func_type: str
    max_frames_per_episode: int
    num_episodes: int
    learning_params: LearningParams
    random_seed_torch: int = 42
    random_seed_gym: int = 777
