from logging import config
from typing import List, Tuple
from dataclasses import dataclass
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from simulatePPOTorchCustom.configs import CriticTrainConfig
from simulatePPOTorchCustom.utlis import get_grad_norm

@dataclass
class CriticStepLogEntry: 
    grad_norm: float
    grad_norm_after_clip: float
    mean_loss: float
    lr: float

@dataclass
class CriticEvaluationMetricsProfile: 
    train_l1: float
    train_l2: float
    val_l1: float
    val_l2: float
    train_advantage_bin_distribution: Tuple[np.ndarray, np.ndarray]
    val_advantage_bin_distribution: Tuple[np.ndarray, np.ndarray]

@dataclass 
class CriticTrainLog: 
    loss_history: List[CriticStepLogEntry]
    eval_results: CriticEvaluationMetricsProfile

def train_critic(
    critic: TensorDictModule, 
    train_critic_trajectories: TensorDict, 
    val_critic_trajectories: TensorDict,
    critic_train_config: CriticTrainConfig,
) -> CriticTrainLog: 

    train_dataloader = get_critic_dataloader(
        critic_dataset=train_critic_trajectories, 
        batch_size=critic_train_config.batch_size
    )

    loss_module = get_critic_loss_module(critic_train_config.loss_fn_name)
    optimizer = torch.optim.Adam(critic.parameters(), critic_train_config.learning_rate)

    steps = int(train_critic_trajectories.shape[0] / critic_train_config.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=critic_train_config.learning_rate,
        epochs=1,
        steps_per_epoch=steps,
        pct_start=critic_train_config.pct_start,
    )

    loss_history = []

    for _, batch_data in enumerate(train_dataloader):
        if batch_data.shape[0] < critic_train_config.batch_size: 
            continue
        critic(batch_data) # add observation state value
        batch_data.rename_key_(old_key='state_value', new_key='observation_state_value')
        batch_data['final_observation_state_value'] = critic(batch_data['final_observation']) # add final observation state value
        loss = loss_module.forward(
            batch_data['observation_state_value'] - batch_data['final_observation_coeff'] * batch_data['final_observation_state_value'],
            batch_data['tail_reward']
        ).mean()
        loss.backward()
        grad_norm = get_grad_norm(critic.parameters())
        torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_train_config.max_grad_norm)
        grad_norm_after_clip = get_grad_norm(critic.parameters())
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        mean_loss = loss.detach().numpy()
        lr = optimizer.param_groups[0]["lr"]
        loss_history.append(CriticStepLogEntry(
            grad_norm=grad_norm,
            grad_norm_after_clip=grad_norm_after_clip,
            mean_loss=mean_loss, 
            lr=lr
        ))
    
    critic_eval_metrics_profile = eval_critic(
        critic=critic,
        train_trajectories=train_critic_trajectories, 
        val_trajectories=val_critic_trajectories, 
        advantage_bin_distribution_num_bins=critic_train_config.advantage_bin_distribution_num_bins
    )
    return CriticTrainLog(
        loss_history=loss_history,
        eval_results=critic_eval_metrics_profile,
    )

def get_critic_loss_module(loss_fn_name: str) -> nn.Module: 
    if loss_fn_name == 'l1': 
        return nn.L1Loss()
    elif loss_fn_name == 'l2': 
        return nn.MSELoss()
    else: 
        raise NotImplementedError(f'loss_fn_name={loss_fn_name} is not implemented for critic')

def critic_dataloader(observations: Tensor, tail_rewards: Tensor, batch_size: int) -> DataLoader: 
    dataset = TensorDataset(observations, tail_rewards)
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=False,
    )

def get_critic_dataloader(critic_dataset: TensorDict, batch_size: int) -> DataLoader: 
    return DataLoader(
        dataset=critic_dataset,
        batch_size=batch_size,
        collate_fn=lambda x: x, 
        shuffle=False,
    )


def eval_critic(
        critic: TensorDictModule, 
        train_trajectories: TensorDict, 
        val_trajectories: TensorDict, 
        advantage_bin_distribution_num_bins: int,
    ) -> CriticEvaluationMetricsProfile: 
    eval_advantage(critic=critic, trajectories=train_trajectories)
    eval_advantage(critic=critic, trajectories=val_trajectories)
    val_l1 = torch.abs(val_trajectories['advantage']).mean().float()
    train_l1 = torch.abs(train_trajectories['advantage']).mean().float()
    val_l2 = (val_trajectories['advantage'] * val_trajectories['advantage']).mean().float()
    train_l2 = (train_trajectories['advantage'] * train_trajectories['advantage']).mean().float()

    train_advantage_bin_counts = np.histogram(train_trajectories['advantage'].numpy(), bins=advantage_bin_distribution_num_bins)
    val_advantage_bin_counts = np.histogram(val_trajectories['advantage'].numpy(), bins=advantage_bin_distribution_num_bins)

    return CriticEvaluationMetricsProfile(
        train_l1=train_l1,
        train_l2=train_l2,
        val_l1=val_l1,
        val_l2=val_l2,
        train_advantage_bin_distribution=train_advantage_bin_counts, 
        val_advantage_bin_distribution=val_advantage_bin_counts,   
    )


def eval_advantage(critic: TensorDictModule, trajectories: TensorDict): 
    with torch.no_grad(): 
        critic(trajectories) # add observation state value
        trajectories.rename_key_(old_key='state_value', new_key='observation_state_value')
        trajectories['final_observation_state_value'] = critic(trajectories['final_observation'])

    trajectories['advantage'] = (
        - trajectories['observation_state_value'] +
        trajectories['final_observation_coeff'] * trajectories['final_observation_state_value'] +
        trajectories['tail_reward']
    )