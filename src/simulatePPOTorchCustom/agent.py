import copy
from typing import Tuple, List, Dict, Any

from dataclasses import dataclass
import numpy as np 
import torch 
from tensordict import TensorDict

from simulatePPOTorchCustom.configs import AgentConfig
from simulatePPOTorchCustom.actor import get_actor
from simulatePPOTorchCustom.critic import get_critic
from simulatePPOTorchCustom.episode_output import EpisodeOutput
from simulatePPOTorchCustom.data_utils import build_dataset
from simulatePPOTorchCustom.train_critic import CriticTrainLog, train_critic, eval_advantage
from simulatePPOTorchCustom.abstract_agent import AbstractAgent
from simulatePPOTorchCustom.train_actor import ActorTrainLog, train_actor

@dataclass
class DatasetMetrics: 
    average_num_frames_per_episode: float
    max_num_frames_per_episode: float
    min_num_frames_per_episode: float
    tail_reward_distribution_bins: Tuple[np.ndarray, np.ndarray]

class Agent(AbstractAgent): 
    def __init__(self, agent_config: AgentConfig) -> None:
        self.config = agent_config
        self.actor = get_actor(config=agent_config.actor_config)
        self.critic = get_critic(value_net_config=agent_config.value_net_config)

    def sample_action(self, observation: np.ndarray) -> Tuple[float, float]: 
        tensor_observation = TensorDict({'observation': observation.astype(np.float32).reshape(1, -1)}, batch_size=1)
        with torch.no_grad(): 
            self.actor(tensor_observation)
        return tensor_observation['action'][0].float(), tensor_observation['sample_log_prob'][0].float()
    
    def update_actor_train_config(self, simulations_results: DatasetMetrics):
        min_num_frames_thresholds = [x.min_episode_length for x in self.config.actor_train_config_schedule.updates_on_episode_length]
        i = -1 
        while len(min_num_frames_thresholds) > i + 1 and simulations_results.min_num_frames_per_episode > min_num_frames_thresholds[i + 1]: 
            i += 1
        if i == -1: 
            return 
        update_params = self.config.actor_train_config_schedule.updates_on_episode_length[i]
        if self.config.actor_train_config.learning_rate > update_params.new_actor_lr:
            self.config.actor_train_config.learning_rate = update_params.new_actor_lr
            self.config.actor_train_config.ppo_loss_config.ppo_eps = update_params.new_ppo_eps
            self.config.actor_train_config.ppo_loss_config.entropy_coeff *= update_params.entropy_coeff_multiplier
            print('Updated actor train config')
            print(f'''
                  lr: {self.config.actor_train_config.learning_rate}, 
                  ppo_eps: {self.config.actor_train_config.ppo_loss_config.ppo_eps}, 
                  entropy_coeff: {self.config.actor_train_config.ppo_loss_config.entropy_coeff}
                  ''')
            
    def update_critic_train_config(self, simulation_results: DatasetMetrics): 
        avg_num_frames_thresholds = [x.average_episode_length for x in self.config.critic_train_config_schedule.updates_on_episode_length]
        i = -1
        while len(avg_num_frames_thresholds) > i + 1 and simulation_results.average_num_frames_per_episode > avg_num_frames_thresholds[i + 1]: 
            i += 1
        if i == -1: 
            return 
        update_params = self.config.critic_train_config_schedule.updates_on_episode_length[i]
        if self.config.gamma < update_params.new_gamma: 
            self.config.gamma = update_params.new_gamma
            self.config.critic_train_config.learning_rate = update_params.new_critic_lr
            print('Updated critic train config')
            print(f'''
                    lr: {self.config.critic_train_config.learning_rate}
                    gamma: {self.config.gamma}
                  ''')

    def update_entropy_coeff(self, mean_abs_objective_loss: float, mean_abs_entropy_loss: float): 
        entropy_to_objective_loss_ratio = mean_abs_entropy_loss / mean_abs_objective_loss
        if  entropy_to_objective_loss_ratio > self.config.actor_train_config_schedule.max_entropy_to_objective_loss_ratio: 
            print('Update entropy coeff')
            print(f'Old coeff: {self.config.actor_train_config.ppo_loss_config.entropy_coeff}')
            self.config.actor_train_config.ppo_loss_config.entropy_coeff = (
                self.config.actor_train_config_schedule.max_entropy_to_objective_loss_ratio / entropy_to_objective_loss_ratio * 0.1
            ) * self.config.actor_train_config.ppo_loss_config.entropy_coeff
            print(f'New coeff: {self.config.actor_train_config.ppo_loss_config.entropy_coeff}')

    def train_actor_critic(
        self,
        episodes_outputs:  List[EpisodeOutput]
    ) -> Tuple[DatasetMetrics, CriticTrainLog, ActorTrainLog]: 
        episodes_data, final_observations = [], []
        for output in episodes_outputs: 
            episode_data, final_observation = output.get_episode_dict()
            episodes_data.append(episode_data), final_observations.append(final_observation)
        
        dataset = build_dataset(
            episodes_data=episodes_data,
            final_observations=final_observations,
            gamma=self.config.gamma
        )
        dataset_size = dataset.shape[0]
        print(f'Dataset size: {dataset_size}')

        average_num_frames_per_episode = sum([x.length() for x in episodes_outputs]) / len(episodes_outputs)
        max_num_frames_per_episode = max([x.length() for x in episodes_outputs])
        min_num_frames_per_episode = min([x.length() for x in episodes_outputs])
        tail_reward_distribution_bins = eval_tail_reward_distribution(dataset)
        dataset_metrics = DatasetMetrics(
            average_num_frames_per_episode=average_num_frames_per_episode,
            max_num_frames_per_episode=max_num_frames_per_episode,
            min_num_frames_per_episode=min_num_frames_per_episode,
            tail_reward_distribution_bins=tail_reward_distribution_bins
        )

        self.update_actor_train_config(simulations_results=dataset_metrics)
        self.update_critic_train_config(simulation_results=dataset_metrics)

        train_val_threshold = int(dataset_size * (1 - self.config.val_ratio))
        train_dataset = dataset[:train_val_threshold]
        val_dataset = dataset[train_val_threshold:]
        print(f'Train datset size: {train_dataset.shape[0]}')
        print(f'Val datset size: {val_dataset.shape[0]}')

        critic_train_log = train_critic(
            critic=self.critic,
            train_critic_trajectories=train_dataset,
            val_critic_trajectories=val_dataset,
            critic_train_config=self.config.critic_train_config
        )
        train_dataset.del_(key='final_observation_state_value').del_('observation_state_value')
        val_dataset.del_(key='final_observation_state_value').del_('observation_state_value')
        
        eval_advantage(critic=self.critic, trajectories=train_dataset)
        eval_advantage(critic=self.critic, trajectories=val_dataset)

        train_dataset, val_dataset = train_dataset.detach(), val_dataset.detach()

        actor_train_log = train_actor(
            actor=self.actor,
            train_trajectories=train_dataset,
            val_trajectories=val_dataset,
            actor_train_config=self.config.actor_train_config,
        )

        mean_abs_objective_loss = sum([
            abs(x.mean_batch_objective_loss)
            for x in actor_train_log.loss_history
        ]) / len(actor_train_log.loss_history)

        mean_abs_entropy_loss = sum([
            abs(x.mean_batch_entropy_loss)
            for x in actor_train_log.loss_history
        ]) / len(actor_train_log.loss_history)

        self.update_entropy_coeff(
            mean_abs_entropy_loss=mean_abs_entropy_loss,
            mean_abs_objective_loss=mean_abs_objective_loss,
        )

        return dataset_metrics, critic_train_log, actor_train_log

def eval_tail_reward_distribution(dataset: TensorDict) -> Tuple[np.ndarray, np.ndarray]: 
    data = dataset['tail_reward'].numpy()
    unique_values, counts = np.unique(data, return_counts=True)
    return unique_values, counts

