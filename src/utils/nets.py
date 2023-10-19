from dataclasses import dataclass
from collections import OrderedDict

from torch import nn
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import TensorDictModule

from .simulator_params import SimulatorParams


@dataclass
class NetParams:
    layers_dims: list[int]
    activation_func_type: str
    device: str


@dataclass
class PolicyParams(NetParams):
    action_space_dim: int


def extract_models_params(params: SimulatorParams, action_space_dim: int) -> (PolicyParams, NetParams):
    return (
        PolicyParams(
            params.policy_layers_dims,
            params.policy_activation_func_type,
            params.device,
            action_space_dim,
        ),
        NetParams(
            params.value_layers_dims,
            params.value_activation_func_type,
            params.device,
        )
    )


def match_activation_fn(activation_fn_type: str):
    match activation_fn_type:
        case 'tanh':
            return nn.Tanh
        case 'relu':
            return nn.ReLU
        case other:
            raise Exception("Unimplemented activation_fn_type {}".format(other))


def stack_dense_layers(params: NetParams) -> list[(str, nn.Module)]:
    activation_fn = match_activation_fn(params.activation_func_type)

    layers: list[(str, nn.Module)] = []
    for i, layer_dim in enumerate(params.layers_dims):
        layers.extend([
            ('dense_{}'.format(i), nn.LazyLinear(layer_dim, device=params.device)),
            ('non_linear_{}'.format(i), activation_fn())
        ])
    return layers


def get_policy_net(params: PolicyParams) -> TensorDictModule:
    layers = stack_dense_layers(params)
    layers.extend([
        ('dense_{}'.format(len(params.layers_dims)), nn.LazyLinear(2 * params.action_space_dim, device=params.device)),
        ('normal_param_extractor', NormalParamExtractor())
    ])
    return TensorDictModule(
        nn.Sequential(OrderedDict(layers)), in_keys=["observation"], out_keys=["loc", "scale"]
    )


def get_value_net(params: NetParams) -> TensorDictModule:
    layers = stack_dense_layers(params)
    layers.append(
        ('dense_{}'.format(len(params.layers_dims)), nn.LazyLinear(1, device=params.device))
    )
    return TensorDictModule(
        nn.Sequential(OrderedDict(layers)), in_keys=['observation'], out_keys=['state_value']
    )
