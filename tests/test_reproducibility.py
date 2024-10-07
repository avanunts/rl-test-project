import gym
import torch
from tensordict import TensorDict

from src.simulatePPOTorchCustom.critic import get_critic
from src.simulatePPOTorchCustom.actor import get_actor

from tests.stubs import stub_value_net_config, stub_actor_config, stub_observations
from tests.utils import must_coincide

def test_reproducibility(): 
    actor_config = stub_actor_config()
    value_net_config = stub_value_net_config()

    torch.manual_seed(0)
    actor = get_actor(config=actor_config)
    critic = get_critic(value_net_config=value_net_config)

    observations = stub_observations()[:5] # deterministic already
    data = TensorDict({'observation': observations}, batch_size=observations.shape[0])
    with torch.no_grad(): 
        actor(data)
        critic(data)

    action = data['action']
    # expected action is first got from the output; 
    # this is a bad solution, but i'm too lazy to implement a good one; 
    expected_action = torch.Tensor([
         1.2983,
        -0.6063,
        -0.2787,
         0.6396,
        -0.1010
    ])
    
    must_coincide(action, expected_action)

    state_value = data['state_value']
    expected_state_value = torch.Tensor([
        0.1743,
        -0.1127,
        0.1587,
        0.0227,
        0.2209
    ])

    must_coincide(state_value, expected_state_value)

