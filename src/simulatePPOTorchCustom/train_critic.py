from typing import List
from dataclasses import dataclass
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
class StepLogEntry: 
    num_epoch: int
    num_batch: int
    grad_norm: float
    grad_norm_after_clip: float
    mean_loss: float
    lr: float
    batch_size: int

def train_critic(
    critic: TensorDictModule, 
    trajectories: TensorDict, 
    critic_train_config: CriticTrainConfig,
): 

    observations = trajectories['observation']
    tail_rewards = trajectories['tail_reward']
    dataloader = critic_dataloader(
        observations=observations, 
        tail_rewards=tail_rewards, 
        batch_size=critic_train_config.batch_size
    )
    loss_module = nn.MSELoss()
    optimizer = torch.optim.Adam(critic.parameters(), critic_train_config.learning_rate)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=critic_train_config.learning_rate,
        epochs=critic_train_config.num_epochs,
        steps_per_epoch=critic_train_config.steps_per_epoch,
        pct_start=critic_train_config.pct_start,
    )

    loss_history = []

    for num_epoch in tqdm(range(critic_train_config.num_epochs)):
        for num_batch, (observations_batch, tail_rewards_batch) in enumerate(dataloader):
            state_values_batch = critic.forward(observations_batch) 
            loss = loss_module.forward(state_values_batch[:, 0], tail_rewards_batch)
            loss.backward()
            grad_norm = get_grad_norm(critic.parameters())
            torch.nn.utils.clip_grad_norm_(critic.parameters(), critic_train_config.max_grad_norm)
            grad_norm_after_clip = get_grad_norm(critic.parameters())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            mean_loss = loss.mean().detach().numpy()
            lr = optimizer.param_groups[0]["lr"]
            loss_history.append(StepLogEntry(
                num_epoch=num_epoch, 
                num_batch=num_batch, 
                grad_norm=grad_norm,
                grad_norm_after_clip=grad_norm_after_clip,
                mean_loss=mean_loss, 
                lr=lr,
                batch_size=observations_batch.shape[0]
            ))
    return loss_history

def critic_dataloader(observations: Tensor, tail_rewards: Tensor, batch_size: int) -> DataLoader: 
    dataset = TensorDataset(observations, tail_rewards)
    return DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
