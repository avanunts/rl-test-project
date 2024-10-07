
import math
from typing import List

from dataclasses import dataclass
from simulatePPOTorchCustom.configs import ActorTrainConfig, PPOLossConfig, LocScaleRegularizationConfig
from simulatePPOTorchCustom.exceptions import NansInLossException
from simulatePPOTorchCustom.utlis import get_grad_norm
from tensordict.nn import TensorDictModule,  ProbabilisticTensorDictModule
from tensordict import TensorDict
from torch.utils.data import DataLoader

from torchviz import make_dot

import torch
from torch import Tensor, nn

class LocScaleRegularizationModule(nn.Module): 
    def __init__(self, loc_scale_regularization_config: LocScaleRegularizationConfig): 
        super().__init__()
        self.config = loc_scale_regularization_config
    
    def forward(self, loc, scale): 
        loc_max_regularization = torch.where(
            loc > self.config.loc_soft_max, 
            loc - self.config.loc_soft_max,
            0.0
        )
        loc_min_regulariztion = torch.where(
            loc < self.config.loc_soft_min, 
            self.config.loc_soft_min - loc, 
            0.0
        )
        loc_regularization = loc_max_regularization + loc_min_regulariztion
        scale_max_regularization = torch.where(
            scale > self.config.scale_soft_max, 
            scale - self.config.scale_soft_max, 
            0.0
        )
        return loc_regularization + scale_max_regularization


class ClippedPPOLossModule(nn.Module):
    def __init__(self, ppo_loss_config: PPOLossConfig):
        super().__init__()
        self.config = ppo_loss_config

    def forward(self, sample_log_prob, episode_sample_log_prob, advantage, entropy):
        log_weight = (sample_log_prob - episode_sample_log_prob)
        log_weight_clamped = log_weight.clamp(
            min=-self.config.ppo_eps, 
            max=self.config.ppo_eps
        )
        gain = log_weight.exp() * advantage
        gain_clamped = log_weight_clamped.exp() * advantage
        loss_objective = -torch.minimum(gain, gain_clamped)
        loss_entropy = -self.config.entropy_coeff * entropy

        return loss_objective, loss_entropy

class LocScaleRegularizationDictModule(TensorDictModule): 
    def __init__(self, loc_scale_regulariation_config: LocScaleRegularizationConfig): 
        super().__init__(
            in_keys=['loc', 'scale'], 
            out_keys=['loc_scale_regulariztion'], 
            module=LocScaleRegularizationModule(loc_scale_regularization_config=loc_scale_regulariation_config)
        )   
class ClippedPPOLossDictModule(TensorDictModule): 
    def __init__(self, ppo_loss_config: PPOLossConfig): 
        super().__init__(
            in_keys=['sample_log_prob', 'episode_sample_log_prob', 'advantage', 'entropy'],
            out_keys=['loss_objective', 'loss_entropy'],
            module=ClippedPPOLossModule(ppo_loss_config=ppo_loss_config)
        )
    
@dataclass
class ActorStepLogEntry: 
    grad_norm: float
    grad_norm_after_clip: float
    mean_batch_objective_loss: float
    mean_batch_entropy_loss: float
    mean_batch_regularization_loss: float
    lr: float

@dataclass
class ActorEvaluationMetricsProfile: 
    train_gain_before_learning: float
    val_gain_before_learning: float
    train_gain_after_learning: float
    val_gain_after_learning: float
    train_optimal_gain: float
    val_optimal_gain: float
    train_mean_abs_log_prob_change: float
    val_mean_abs_log_prob_change: float
    train_mean_entropy_after_learning: float
    val_mean_entropy_after_learning: float


@dataclass 
class ActorTrainLog: 
    loss_history: List[ActorStepLogEntry]
    eval_results: ActorEvaluationMetricsProfile

def train_actor(
    actor: TensorDictModule, 
    train_trajectories: TensorDict, 
    val_trajectories: TensorDict,
    actor_train_config: ActorTrainConfig,
): 

    actor_dataloader = get_actor_dataloader(
        data=train_trajectories,
        batch_size=actor_train_config.batch_size,
    )

    loss_module = ClippedPPOLossDictModule(actor_train_config.ppo_loss_config)
    regularization_loss_module = LocScaleRegularizationDictModule(actor_train_config.loc_scale_regularization_config)
    optimizer = torch.optim.Adam(actor.parameters(), actor_train_config.learning_rate)
    steps = int(train_trajectories.shape[0] / actor_train_config.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer, 
        max_lr=actor_train_config.learning_rate,
        epochs=1,
        steps_per_epoch=steps,
        pct_start=actor_train_config.pct_start,
    )

    loss_history = []

    for _, input_batch in enumerate(actor_dataloader):
        if input_batch.shape[0] < actor_train_config.batch_size: 
            continue

        dist = actor.get_dist(input_batch)
        input_batch['sample_log_prob'] = dist.log_prob(input_batch['episode_action'])
        input_batch['entropy'] = dist.entropy()
        loss_module(input_batch)
        regularization_loss_module(input_batch)
        if torch.any(torch.isnan(input_batch['loss_objective'])): 
            index = torch.isnan(input_batch['loss_objective'])
            for key in input_batch.keys(): 
                torch.save(input_batch[key][index], f'{key}.pt')
            raise NansInLossException('Found nans in loss objective')
        if torch.any(torch.isnan(input_batch['loss_entropy'])): 
            index = torch.isnan(input_batch['loss_entropy'])
            for key in input_batch.keys(): 
                torch.save(input_batch[key][index], f'{key}.pt')
            raise NansInLossException('Found nans in loss entropy')
        mean_batch_objective_loss = input_batch['loss_objective'].mean()
        mean_batch_entropy_loss = input_batch['loss_entropy'].mean()
        mean_batch_regularization_loss = input_batch['loc_scale_regulariztion'].mean()
        mean_batch_loss = mean_batch_objective_loss + mean_batch_entropy_loss + mean_batch_regularization_loss
        mean_batch_loss.backward()
        grad_norm = get_grad_norm(actor.parameters())
        torch.nn.utils.clip_grad_norm_(actor.parameters(), actor_train_config.max_grad_norm)
        grad_norm_after_clip = get_grad_norm(actor.parameters())

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        loss_history.append(ActorStepLogEntry(
            grad_norm=grad_norm,
            grad_norm_after_clip=grad_norm_after_clip,
            mean_batch_entropy_loss=mean_batch_entropy_loss.detach(), 
            mean_batch_objective_loss=mean_batch_objective_loss.detach(),
            mean_batch_regularization_loss=mean_batch_regularization_loss.detach(),
            lr=lr
        ))
    
    actor_eval_metrics_profile = eval_actor(
        actor=actor,
        train_trajectories=train_trajectories,
        val_trajectories=val_trajectories,
        loss_module=loss_module,
        config_log_prob_clamp=actor_train_config.ppo_loss_config.ppo_eps
    )


    return ActorTrainLog(
        loss_history=loss_history,
        eval_results=actor_eval_metrics_profile,
    )
    
    
def get_actor_dataloader(
    data: TensorDict,
    batch_size: int,
): 
    return DataLoader(
        dataset=data,
        batch_size=batch_size,
        collate_fn=lambda x: x
    )

def eval_actor(
    actor: ProbabilisticTensorDictModule, 
    train_trajectories: TensorDict, 
    val_trajectories: TensorDict, 
    loss_module: TensorDictModule, 
    config_log_prob_clamp: float,
) -> ActorEvaluationMetricsProfile:
    with torch.no_grad(): 
        train_gain_before_learning = train_trajectories['advantage'].mean().numpy()
        val_gain_before_learning = val_trajectories['advantage'].mean().numpy()
        train_optimal_gain = torch.where(
            train_trajectories['advantage'] > 0, 
            train_trajectories['advantage'] * math.exp(config_log_prob_clamp), 
            train_trajectories['advantage'] * math.exp(-config_log_prob_clamp)
        ).mean().numpy()
        val_optimal_gain = torch.where(
            val_trajectories['advantage'] > 0, 
            val_trajectories['advantage'] * math.exp(config_log_prob_clamp), 
            val_trajectories['advantage'] * math.exp(-config_log_prob_clamp)
        ).mean().numpy()
        train_dists = actor.get_dist(train_trajectories)
        train_trajectories['sample_log_prob'] = train_dists.log_prob(train_trajectories['episode_action'])
        train_trajectories['entropy'] = train_dists.entropy()
        loss_module(train_trajectories)
        mean_train_gain_after_learning = train_trajectories['loss_objective'].mean().numpy()
        train_mean_abs_log_prob_change = torch.abs(train_trajectories['sample_log_prob'] - train_trajectories['episode_sample_log_prob']).mean().numpy()
        train_mean_entropy_after_learning = train_trajectories['entropy'].mean().numpy()
        

        val_dists = actor.get_dist(val_trajectories)
        val_trajectories['sample_log_prob'] = val_dists.log_prob(val_trajectories['episode_action'])
        val_trajectories['entropy'] = val_dists.entropy()
        loss_module(val_trajectories)
        mean_val_gain_after_learning = val_trajectories['loss_objective'].mean().numpy()
        val_mean_abs_log_prob_change = torch.abs(val_trajectories['sample_log_prob'] - val_trajectories['episode_sample_log_prob']).mean().numpy()
        val_mean_entropy_after_learning = val_trajectories['entropy'].mean().numpy()


    return ActorEvaluationMetricsProfile(
        train_gain_after_learning=mean_train_gain_after_learning,
        val_gain_after_learning=mean_val_gain_after_learning,
        train_gain_before_learning=train_gain_before_learning,
        val_gain_before_learning=val_gain_before_learning,
        train_optimal_gain=train_optimal_gain,
        val_optimal_gain=val_optimal_gain,
        train_mean_abs_log_prob_change=train_mean_abs_log_prob_change,
        val_mean_abs_log_prob_change=val_mean_abs_log_prob_change,
        train_mean_entropy_after_learning=train_mean_entropy_after_learning,
        val_mean_entropy_after_learning=val_mean_entropy_after_learning
    )
        