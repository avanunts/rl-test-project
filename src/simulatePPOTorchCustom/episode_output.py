from typing import Tuple, Dict, Optional

import numpy as np 

class EpisodeOutput: 
    def __init__(self,):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.final_observation = None
    
    def add_frame(self, observation: np.ndarray, action: float, log_prob: float, reward: float): 
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
    
    def add_final_observation(self, final_observation: np.ndarray): 
        self.final_observation = final_observation

    def get_episode_dict(self) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]: 
        return {
            'observation': np.vstack(self.observations), 
            'episode_action': np.array(self.actions),
            'episode_sample_log_prob': np.array(self.log_probs),
            'reward': np.array(self.rewards)
        }, self.final_observation

    def length(self) -> int: 
        return len(self.observations)