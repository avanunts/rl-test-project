from typing import List, Dict, Optional

import numpy as np
import torch
from tensordict import TensorDict 

def build_dataset(
        episodes_data: List[Dict[str, np.ndarray]], 
        final_observations: List[Optional[np.ndarray]],
        gamma: float
) -> TensorDict: 
    episodes_tensor_dicts = [
        build_episode_tensor_dict(episode_data, final_observation, gamma) 
        for episode_data, final_observation in zip(episodes_data, final_observations)
    ]
    
    return torch.cat(episodes_tensor_dicts)

def build_episode_tensor_dict(
    episode_data: Dict[str, np.ndarray],
    final_observation: Optional[np.ndarray],
    gamma: float
) -> TensorDict: 
    episode_length = len(episode_data['observation'])
    tail_reward = np.zeros(episode_length)
    for i in reversed(range(episode_length)): 
        if i == episode_length - 1: 
            tail_reward[i] = episode_data['reward'][i]
        else: 
            tail_reward[i] = episode_data['reward'][i] + gamma * tail_reward[i + 1]

    if final_observation is None: 
        final_obs_coeff = np.zeros(episode_length)
        final_observation_data = np.zeros((episode_length, 4))
    else: 
        final_observation_data = np.repeat(final_observation.reshape(1, -1), episode_length, axis=0)
        final_obs_coeff = gamma ** np.arange(episode_length, 0, -1)
    
    return TensorDict({
        'observation': episode_data['observation'].astype(np.float32), 
        'episode_sample_log_prob': episode_data['episode_sample_log_prob'].astype(np.float32),
        'episode_action': episode_data['episode_action'].astype(np.float32),
        'tail_reward': tail_reward.astype(np.float32),
        'final_observation_coeff': final_obs_coeff.astype(np.float32),
        'final_observation': final_observation_data.astype(np.float32),
        }, 
        batch_size=episode_length
    )

