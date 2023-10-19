from collections import defaultdict
from tqdm import tqdm

import torch
from tensordict.tensordict import TensorDict
from torch import Tensor
from torch.utils.data import DataLoader


from utils.simulator_params import SimulatorParams
from utils.nets import get_value_net, get_policy_net, extract_models_params
from .models import get_actor
from .envs import get_env, EnvParams
from .loss import ClippedPPOLoss, LossParams
from .helpers import *
from .normalizers import get_normalization_terms


def extract_env_params(params: SimulatorParams) -> EnvParams:
    return EnvParams(params.env_name, params.max_frames_per_episode)


def extract_loss_params(params: SimulatorParams) -> LossParams:
    return LossParams(
        params.learning_params.gamma,
        params.learning_params.gamma,
        params.learning_params.entropy_coef,
        params.learning_params.loss_critic_type,
        params.learning_params.ciric_loss_coef,
        params.learning_params.samples_mc_entropy
    )


class Simulator:
    def __init__(self, params: SimulatorParams):
        self.params = params
        self.env_seed = params.random_seed_gym
        torch.manual_seed(params.random_seed_torch)
        self.env = get_env(extract_env_params(params))
        self.obs_mean, self.obs_std = get_normalization_terms(params.env_name)

        policy_params, value_params = extract_models_params(params, self.env.action_space.shape[0])
        policy_net = get_policy_net(policy_params)
        device = torch.device(params.device)
        self.actor = get_actor(policy_net, self.env.action_space, device)
        self.critic = get_value_net(value_params)
        self.init_lazy_layers()

        loss_params = extract_loss_params(params)
        self.loss_module = ClippedPPOLoss(self.actor, self.critic, loss_params)
        self.optim = torch.optim.Adam(self.loss_module.parameters(), params.learning_params.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, params.num_episodes, params.learning_params.cosine_lr_min_ratio * params.learning_params.lr
        )
        self.data_collector = []

    def get_seed(self):
        self.env_seed += 1
        return self.env_seed - 1

    def init_lazy_layers(self):
        obs, _ = self.env_reset_normalized()
        with torch.no_grad():
            self.actor(torch.Tensor([obs]))
            self.critic(torch.Tensor([obs]))

    def train_episode(self, loop_log):
        stacked_td = torch.stack(self.data_collector, dim=0).flatten()
        self.data_collector = []
        dataloader = DataLoader(
            stacked_td,
            batch_size=self.params.learning_params.batch_size,
            shuffle=True,
            collate_fn=lambda x: x
        )
        lr = self.optim.param_groups[0]["lr"]
        for _ in tqdm(range(self.params.learning_params.num_epochs)):
            for data in dataloader:
                loss_vals = self.loss_module(data)
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.params.learning_params.max_grad_norm)
                self.optim.step()
                self.optim.zero_grad()
        self.scheduler.step()
        loop_log['lr'].append(lr)

    def run_episode(self):
        self.data_collector = []
        with torch.no_grad():
            episode_log = defaultdict(list)
            obs, _ = self.env_reset_normalized()
            done = False
            info = None
            while not done:
                td = TensorDict({'observation': torch.Tensor([obs])}, batch_size=1).to(device=self.params.device)
                self.actor(td)
                self.critic(td)
                obs, reward, terminated, truncated, info = self.env_step_normalized(td)
                td['reward'] = Tensor([[reward]])

                episode_log['reward'].append(reward)
                episode_log['info'].append(info)
                done = terminated or truncated
                self.data_collector.append(td)
            if truncated:
                final_obs = info['final_observation']
                # double input to overcome error in the BatchNorm1d layer
                final_value = self.critic(torch.Tensor([final_obs]).to(device=self.params.device))
            else:
                final_value = torch.Tensor([[0.0]]).to(device=self.params.device)
            tail_reward_suffix = final_value
            gamma = self.params.learning_params.gamma
            for td in reversed(self.data_collector):
                tail_reward = td['reward'] + gamma * tail_reward_suffix
                td['tail_reward'] = tail_reward
                tail_reward_suffix = tail_reward
        return episode_log

    def run_episodes(self):
        loop_log = defaultdict(list)
        for i in range(self.params.num_episodes):
            episode_log = self.run_episode()
            self.train_episode(loop_log)
            step_count = len(episode_log['reward'])
            loop_log['avg_reward'].append(sum(episode_log['reward']) / step_count)
            loop_log['step_count'].append(step_count)
            print_stats_for_episode(loop_log)

        return loop_log

    def env_reset_normalized(self):
        obs, _ = self.env.reset(seed=self.get_seed())
        return self.normalize_obs(obs), None

    def env_step_normalized(self, td: TensorDict):
        obs, reward, terminated, truncated, info = self.env.step(td['action'].flatten().numpy())
        obs = self.normalize_obs(obs)
        return obs, reward, terminated, truncated, info

    def normalize_obs(self, obs):
        return (obs - self.obs_mean) / self.obs_std




