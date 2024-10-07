from typing import List, Optional
from dataclasses import dataclass
import gym
import numpy as np 

from simulatePPOTorchCustom.episode_output import EpisodeOutput
from simulatePPOTorchCustom.abstract_agent import AbstractAgent

class Runner: 
    def __init__(self, env: gym.Env) -> None:
        self.env = env
        self.env_seed = 0
    
    def run_episode(
        self,
        agent: AbstractAgent
    ) -> EpisodeOutput: 
        
        output = EpisodeOutput()
        observation, _ = self.env.reset(seed=self.env_seed)
        self.env_seed += 1
        done = False
        while not done: 
            action, log_prob = agent.sample_action(observation=observation)
            next_observation, reward, terminated, truncated, _ = self.env.step(np.array([action]))
            output.add_frame(
                observation=observation, 
                action=action,
                log_prob=log_prob,
                reward=reward
            )
            observation = next_observation
            done = terminated or truncated
        if truncated:
            output.add_final_observation(final_observation=observation)
        return output

    def run_episode_sequence(
        self, 
        agent: AbstractAgent,
        target_num_frames: Optional[int], 
        num_episodes: Optional[int],
    ) -> List[EpisodeOutput]: 
        if num_episodes is not None and target_num_frames is not None: 
            raise ValueError(f'only one of two arguments target_num_frames or num_episodes must be not None, got both.')
        
        if num_episodes is not None: 
            return [self.run_episode(agent=agent) for _ in range(num_episodes)]
        
        num_frames = 0
        results = []
        while num_frames < target_num_frames: 
            results.append(self.run_episode(agent=agent))
            num_frames += len(results[-1].observations)
        return results
