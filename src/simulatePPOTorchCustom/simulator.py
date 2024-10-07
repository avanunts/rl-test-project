from typing import List

import gym 
import numpy as np 

from dataclasses import dataclass
from simulatePPOTorchCustom.configs import SimulatorConfig
from simulatePPOTorchCustom.exceptions import NansInLossException
from simulatePPOTorchCustom.runner import Runner
from simulatePPOTorchCustom.agent import Agent, DatasetMetrics
from simulatePPOTorchCustom.episode_output import EpisodeOutput
from simulatePPOTorchCustom.train_actor import ActorTrainLog
from simulatePPOTorchCustom.train_critic import CriticTrainLog

@dataclass
class SimulationEpochResults: 
    dataset_metrics: DatasetMetrics
    critic_train_log: CriticTrainLog
    actor_train_log: ActorTrainLog

class Simulator: 
    def __init__(self, simulator_config: SimulatorConfig): 
        self.simulator_config = simulator_config
        env = gym.make(
            'InvertedPendulum-v4',
            max_episode_steps=simulator_config.env_max_episode_steps,
            autoreset=False,
        )
        self.runner = Runner(env=env)
        self.agent = Agent(agent_config=simulator_config.agent_config)

        self.results = []

    def simulate(self, num_epochs: int): 
        for _ in range(num_epochs): 
            episodes_outputs = self.runner.run_episode_sequence(
                agent=self.agent, 
                target_num_frames=self.simulator_config.target_num_frames_for_train_val,
                num_episodes=None,
            )

            print(f'Num episodes: {len(episodes_outputs)}, total frames: {sum([x.length() for x in episodes_outputs])}')

            try: 
                dataset_metrics, critic_train_log, actor_train_log = self.agent.train_actor_critic(episodes_outputs)
            except NansInLossException as e: 
                print('Stopped train actor because of the raised exception')
                print(f'Exception message: {e}')
                return 

            self.results.append(SimulationEpochResults(
                dataset_metrics=dataset_metrics,
                critic_train_log=critic_train_log,
                actor_train_log=actor_train_log
            ))

    def get_results(self) -> List[SimulationEpochResults]: 
        return self.results
