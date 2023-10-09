from dataclasses import dataclass
from torch import nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.data.tensor_specs import TensorSpec


@dataclass
class PolicyParams:
    layers_dims: list[int]
    activation_func_type: str
    device: str
    action_space_dims: int


@dataclass
class ValueParams:
    layers_dims: list[int]
    activation_func_type: str
    device: str


def match_activation_fn(activation_fn_type: str):
    match activation_fn_type:
        case 'tanh':
            return nn.Tanh
        case 'relu':
            return nn.ReLU
        case other:
            raise Exception("Unimplemented activation_fn_type {}".format(other))


def stack_layers(layers_dims: list[int], device: str, activation_func_type: str):
    activation_fn = match_activation_fn(activation_func_type)

    layers = []
    for layer_dim in layers_dims:
        layers.extend([nn.LazyLinear(layer_dim, device=device), activation_fn()])
    return layers


def get_policy_net(params: PolicyParams):
    layers = stack_layers(params.layers_dims, params.device, params.activation_func_type)
    layers.extend([nn.LazyLinear(2 * params.action_space_dims, device=params.device), NormalParamExtractor()])
    return nn.Sequential(*layers)


def get_value_net(params: ValueParams):
    layers = stack_layers(params.layers_dims, params.device, params.activation_func_type)
    layers.append(nn.LazyLinear(1, device=params.device))
    return nn.Sequential(*layers)


def get_actor(policy_params: PolicyParams, action_spec: TensorSpec):
    policy_net = get_policy_net(policy_params)
    policy_net_wrapped = TensorDictModule(
        policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    return ProbabilisticActor(
        module=policy_net_wrapped,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.minimum,  # this will raise an error if the spec is not bounded
            "max": action_spec.space.maximum,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )


def get_critic(params: ValueParams):
    value_net = get_value_net(params)
    return ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )
