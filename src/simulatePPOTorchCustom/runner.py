from typing import Tuple, Any, List, Optional
import gym
import numpy as np 
import torch 
from torch import Tensor
from tensordict.tensordict import TensorDict
from tensordict.nn import TensorDictModule

def run_episodes_sequence(
    actor: TensorDictModule,
    critic: TensorDictModule,
    env: gym.Env, 
    num_episodes: int,
    gym_seeds: Optional[List[int]] = None,
) -> Tuple[List[TensorDict], List[float]]: 
    trajectories = []
    for num_episode in range(num_episodes): 
        if gym_seeds is not None: 
            init_gym_obs, _ = env.reset(seed=gym_seeds[num_episode])    
        else: 
            init_gym_obs, _ = env.reset(seed=num_episode)
        trajectory, final_value = run_episode(actor=actor, critic=critic, env=env, init_gym_obs=init_gym_obs)
        append_tail_reward(trajectory, final_value)
        trajectories.append(trajectory)
    return torch.cat(trajectories, dim=0)
    

def run_episode(
    actor: TensorDictModule,
    critic: TensorDictModule,
    env: gym.Env, 
    init_gym_obs: np.ndarray
) -> Tuple[TensorDict, float]: 
    
    all_frames_data = []
    obs = convert_gym_obs_to_torch(init_gym_obs)
    with torch.no_grad():
        done = False
        while not done: 
            frame_data = TensorDict({'observation': obs}, batch_size=1)
            actor(frame_data)
            critic(frame_data)
            gym_obs, reward, terminated, truncated, info = env.step(frame_data['action'].flatten().numpy())
            obs = convert_gym_obs_to_torch(gym_obs)
            frame_data['reward'] = torch.Tensor([reward])
            all_frames_data.append(frame_data)
            done = terminated or truncated
        if truncated:
            final_obs = info['final_observation']
            final_value = critic(torch.Tensor([final_obs]))
        else:
            final_value = torch.Tensor([[0.0]])
    all_frames_concat = torch.cat(all_frames_data)
    key_map_rename = {
        'loc': 'episode_loc', 
        'scale': 'episode_scale', 
        'action': 'episode_action', 
        'sample_log_prob': 'episode_sample_log_prob', 
        'state_value': 'episode_state_value'
    }
    for key, value in key_map_rename.items(): 
        all_frames_concat.rename_key_(old_key=key, new_key=value)
    return all_frames_concat, final_value
        
def convert_gym_obs_to_torch(gym_obs: np.ndarray) -> torch.Tensor: 
    return torch.Tensor(gym_obs.reshape(1, -1))

def append_tail_reward(trajectory: TensorDict, final_value: Tensor): 
    reward = trajectory['reward']
    reverse_reward = torch.flip(torch.flatten(reward), dims=[0]) + final_value.flatten()
    tail_reward = torch.flip(
        torch.cumsum(reverse_reward, dim=0), 
        dims=[0]
    ).reshape(reward.shape)
    trajectory['tail_reward'] = tail_reward