
from webbrowser import get
import numpy as np 
import torch 
from tensordict import TensorDict

from src.simulatePPOTorchCustom.actor import PolicyNet, get_prob_module, get_actor

from tests.stubs import stub_policy_net_conifg, stub_actor_config, stub_observations
from tests.distribution_utils import get_tanh_standard_normal_mean_and_std


from torch.distributions.transforms import TanhTransform, AffineTransform
from torch import FloatTensor

def test_policy_net(): 
    policy_net_config = stub_policy_net_conifg()
    policy_net = PolicyNet(policy_net_config=policy_net_config)

    observations = stub_observations()
    batch_size = observations.shape[0]

    with torch.no_grad(): 
        loc, scale = policy_net(observations)
    
    assert len(loc.shape) == 2
    assert loc.shape[0] == batch_size 
    assert loc.shape[1] == 1

    assert len(scale.shape) == 2
    assert scale.shape[0] == batch_size
    assert scale.shape[1] == 1
    assert torch.all(scale > 0)

def test_prob_module(): 
    tanh_min = 0.0 
    tanh_max = 10.0
    clamp_log_prob = -5
    prob_module = get_prob_module(
        tanh_min=tanh_min,
        tanh_max=tanh_max,
        clamp_log_prob=clamp_log_prob,
    )

    batch_size=1_000_000
    loc = torch.zeros(size=(batch_size, 1))
    scale = torch.zeros(size=(batch_size, 1)) + 1.0
    data = TensorDict({'loc': loc, 'scale': scale}, batch_size=batch_size)
    
    prob_module(data)
    data_keys = data.keys()
    
    assert 'loc' in data_keys
    assert 'scale' in data_keys
    assert 'action' in data_keys
    assert 'sample_log_prob' in data_keys
    assert len(data_keys) == 4

    assert torch.all(data['action'] > tanh_min)
    assert torch.all(data['action'] < tanh_max)
    assert data['action'].shape[1] == 1
    assert data['sample_log_prob'].shape[1] == 1
    
    action_mean = torch.mean(data['action'])
    action_std = torch.std(data['action'])

    tanh_standard_normal_mean, tanh_standard_normal_std = get_tanh_standard_normal_mean_and_std()

    must_be_action_mean = 5 * tanh_standard_normal_mean + 5
    must_be_action_std = 5 * tanh_standard_normal_std

    assert torch.abs(action_mean - must_be_action_mean) < 1.0e-2
    assert torch.abs(action_std - must_be_action_std) < 1.0e-2


def test_actor(): 
    actor_config = stub_actor_config()

    actor = get_actor(actor_config)

    observations = stub_observations()
    data = TensorDict({'observation': observations}, batch_size=observations.shape[0])

    with torch.no_grad(): 
        actor.forward(data) 
    
    data_keys = data.keys()

    assert 'observation' in data_keys
    assert 'loc' in data_keys
    assert 'scale' in data_keys
    assert 'action' in data_keys
    assert 'sample_log_prob' in data_keys
    assert len(data_keys) == 5