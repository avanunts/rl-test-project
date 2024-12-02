from typing import List
from dataclasses import dataclass
from collections import OrderedDict

from torch import nn
from tensordict.nn.distributions import NormalParamExtractor

from utils.nets import match_activation_fn

import torch
from tensordict.nn import (
    TensorDictModule, ProbabilisticTensorDictModule,
    InteractionType, ProbabilisticTensorDictSequential
)



from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.normal import Normal
from torch.distributions.transforms import TanhTransform, AffineTransform

from simulatePPOTorchCustom.configs import ActorConfig, PolicyNetConfig


def get_actor(config: ActorConfig): 
    policy_net = TensorDictModule(PolicyNet(config.policy_net_config), in_keys=['observation'], out_keys=['loc', 'scale'])

    prob_module = get_prob_module(
        tanh_min=config.policy_net_config.action_space_low,
        tanh_max=config.policy_net_config.action_space_high,
        clamp_log_prob=config.clamp_log_prob,
    )

    return ProbabilisticTensorDictSequential(policy_net, prob_module)



class PolicyNet(torch.nn.Module): 
    def __init__(self, policy_net_config: PolicyNetConfig): 
        super().__init__()
        self.scale_min = policy_net_config.scale_min
        in_dims = [policy_net_config.observation_space_dim] + policy_net_config.hidden_layers_dims[:-1]
        out_dims = policy_net_config.hidden_layers_dims
        activation_fn_names = policy_net_config.hidden_layers_activations

        hidden_layers = []
        for i, (in_dim, out_dim, activation_fn_name) in enumerate(zip(in_dims, out_dims, activation_fn_names)): 
            hidden_layers.extend([
                (f'dense_{i}', nn.Linear(in_features=in_dim, out_features=out_dim)), 
                (f'activation_{i}', match_activation_fn(activation_fn_name)())
            ])

        self.hidden_layers_module = nn.Sequential(OrderedDict(hidden_layers))

        self.loc = nn.Linear(in_features=policy_net_config.hidden_layers_dims[-1], out_features=1)
        self.scale = nn.Linear(in_features=policy_net_config.hidden_layers_dims[-1], out_features=1)
        
    def forward(self, x): 
        hidden_layers = self.hidden_layers_module(x)
        loc = self.loc(hidden_layers)[:, 0]
        scale = torch.abs(self.scale(hidden_layers))[:, 0] # in order to have positive scale in the result
        return loc, scale
    

def get_prob_module(**tanh_normal_affine_1d_kwargs): 
    return ProbabilisticTensorDictModule(
        in_keys=['loc', 'scale'],
        out_keys=['action'],
        default_interaction_type=InteractionType.RANDOM,
        distribution_class=TanhNormalAffine1d,
        distribution_kwargs=tanh_normal_affine_1d_kwargs,
        return_log_prob=True
    )

class TanhNormalAffine1d(TransformedDistribution):
    def __init__(self, loc, scale, tanh_min, tanh_max, clamp_log_prob, samples_mc_entropy=20):
        self.samples_mc_entropy = samples_mc_entropy
        super().__init__(
            Normal(loc, scale),
            [
                TanhTransform(),
                AffineTransform(
                    loc=(tanh_min + tanh_max) / 2,
                    scale=(tanh_max - tanh_min) / 2,
                )
            ]
        )

    def entropy(self): 
        x = self.rsample((self.samples_mc_entropy,))
        log_prob = self.log_prob(x)
        return -log_prob.mean(dim=0)