from dataclasses import dataclass

import math

import torch
from torch.nn import functional as F
from torch.distributions import Distribution
from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential, TensorDictModuleBase
from tensordict.tensordict import TensorDict


@dataclass
class LossParams:
    gamma: float
    clip_epsilon: float
    entropy_coef: float
    loss_critic_type: str
    critic_loss_coef: float
    samples_mc_entropy: int


class ClippedPPOLoss(TensorDictModuleBase):
    def __init__(self, actor: ProbabilisticTensorDictSequential, critic: TensorDictModule, params: LossParams):
        super().__init__()
        self.in_keys = ['observation', 'action', 'tail_reward', 'sample_log_prob']
        self.out_keys = ['loss_objective', 'loss_entropy', 'loss_critic']
        self.actor = actor
        self.critic = critic
        self.params = params

    def _clip_bounds(self):
        return (
            math.log1p(-self.params.clip_epsilon),
            math.log1p(self.params.clip_epsilon),
        )

    def distance_loss(self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        loss_function: str,
    ) -> torch.Tensor:
        match loss_function:
            case "l2":
                value_loss = F.mse_loss(v1, v2, reduction="none")
            case "l1":
                value_loss = F.l1_loss(v1, v2, reduction="none")
            case "smooth_l1":
                value_loss = F.smooth_l1_loss(v1, v2, reduction="none")
            case other:
                raise NotImplementedError(f"Unknown loss {other}")
        return value_loss.mean() * self.params.critic_loss_coef

    def get_entropy_bonus(self, dist: Distribution) -> torch.Tensor:
        try:
            entropy = dist.entropy()
        except NotImplementedError:
            x = dist.rsample((self.params.samples_mc_entropy,))
            entropy = -dist.log_prob(x)
        return entropy.unsqueeze(-1)

    def forward(self, td: TensorDict):
        self.critic(td)
        with torch.no_grad():
            advantage = td['tail_reward'] - td['state_value']

        dist = self.actor.get_dist(td)
        log_prob = dist.log_prob(td['action'])
        log_weight = (log_prob - td['sample_log_prob']).unsqueeze(-1)
        gain1 = log_weight.exp() * advantage
        log_weight_clip = log_weight.clamp(*self._clip_bounds())
        gain2 = log_weight_clip.exp() * advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1)[0]
        gain_loss = -gain.mean()  # add minus, because we will minimize

        entropy = self.get_entropy_bonus(dist)
        entropy_loss = -self.params.entropy_coef * entropy.mean()
        critic_loss = self.distance_loss(td['state_value'], td['tail_reward'], self.params.loss_critic_type)
        td_out = TensorDict({
            "loss_objective": gain_loss,
            "loss_critic": critic_loss,
            "loss_entropy": entropy_loss},
            []
        )
        return td_out







