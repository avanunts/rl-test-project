from typing import Tuple
import numpy as np
import torch
import gym 

from src.simulatePPOTorchCustom.configs import ActorConfig, PolicyNetConfig, ValueNetConfig, LocScaleRegularizationConfig

def stub_policy_net_conifg() -> PolicyNetConfig: 
    return PolicyNetConfig(
        observation_space_dim=4, 
        hidden_layers_dims=[4, 4, 4, 4],
        hidden_layers_activations=['relu', 'relu', 'relu', 'relu'], 
        action_space_low=-3.0, 
        action_space_high=3.0, 
        scale_min=1.0e-2
    )

def stub_value_net_config() -> ValueNetConfig: 
    return ValueNetConfig(
        observation_space_dim=4, 
        hidden_layers_dims=[4, 4, 4],
        hidden_layers_activations=['relu', 'relu', 'relu']
    )

def stub_actor_config() -> ActorConfig: 
    policy_net_config = stub_policy_net_conifg()

    return ActorConfig(policy_net_config=policy_net_config, clamp_log_prob=-10)


def stub_observations() -> torch.Tensor: 
    rng = np.random.default_rng(seed=42)
    batch_size = 1_000
    observation_dim = 4
    loc = 3.0 
    scale = 10.0
    obs_np = (scale * rng.standard_normal(size=(batch_size, observation_dim)) + loc).astype(np.float32)
    return torch.Tensor(obs_np)

def stub_env(max_episode_steps: int=100_000) -> gym.Env: 
    return gym.make(
        'InvertedPendulum-v4',
        max_episode_steps=max_episode_steps,
        autoreset=False,
    )

class StubAgent: 
    def __init__(self):
        rng = np.random.default_rng(seed=42) 
        self.current_position = 0
        self.actions = rng.random(size=100) - 0.5

    def sample_action(self, observation: np.ndarray) -> Tuple[float, float]: 
        action = self.actions[self.current_position]
        log_prob = 0.0 
        self.current_position = (self.current_position + 1) % len(self.actions)
        return action, log_prob

