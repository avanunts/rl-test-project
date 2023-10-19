import torch
from tensordict.nn import (
    TensorDictModule, ProbabilisticTensorDictModule,
    InteractionType, ProbabilisticTensorDictSequential
)

from gymnasium.spaces import Box
from torch import FloatTensor
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.normal import Normal
from torch.distributions.transforms import TanhTransform, AffineTransform


class TanhNormal(TransformedDistribution):
    def __init__(self, loc, scale, tanh_min, tanh_max, device):
        super().__init__(
            Normal(loc, scale),
            [
                TanhTransform(),
                AffineTransform(
                    FloatTensor((tanh_min + tanh_max) / 2).to(device),
                    FloatTensor((tanh_max - tanh_min) / 2).to(device)
                )
            ]
        )


def get_actor(policy_net: TensorDictModule, action_space: Box, device: torch.device) -> ProbabilisticTensorDictSequential:
    prob_module = ProbabilisticTensorDictModule(
        in_keys=['loc', 'scale'],
        out_keys=['action'],
        default_interaction_type=InteractionType.RANDOM,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "tanh_min": action_space.low,
            "tanh_max": action_space.high,
            "device": device,
        },
        return_log_prob=True
    )
    return ProbabilisticTensorDictSequential(
     policy_net, prob_module
 )
