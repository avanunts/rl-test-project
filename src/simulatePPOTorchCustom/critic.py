from collections import OrderedDict
from turtle import forward

import torch
from torch import nn
from tensordict.nn import TensorDictModule
from simulatePPOTorchCustom.configs import ValueNetConfig

from utils.nets import match_activation_fn


class ValueNet(nn.Module): 
    def __init__(self, value_net_config: ValueNetConfig): 
        super().__init__()
        in_dims = [value_net_config.observation_space_dim] + value_net_config.hidden_layers_dims[:-1]
        out_dims = value_net_config.hidden_layers_dims
        activation_fn_names = value_net_config.hidden_layers_activations

        hidden_layers = []
        for i, (in_dim, out_dim, activation_fn_name) in enumerate(zip(in_dims, out_dims, activation_fn_names)): 
            hidden_layers.extend([
                (f'dense_{i}', nn.Linear(in_features=in_dim, out_features=out_dim)), 
                (f'activation_{i}', match_activation_fn(activation_fn_name)())
            ])

        last_layer = ('pre_value', nn.Linear(in_features=out_dims[-1], out_features=1))

        self.module = nn.Sequential(OrderedDict(hidden_layers + [last_layer]))

    def forward(self, x): 
        return self.module.forward(x)[:, 0]

def get_critic(value_net_config: ValueNetConfig): 
    value_net = ValueNet(value_net_config=value_net_config)
    return TensorDictModule(value_net, in_keys=['observation'], out_keys=['state_value'])

        