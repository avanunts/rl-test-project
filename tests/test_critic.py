import torch 
from tensordict import TensorDict

from src.simulatePPOTorchCustom.critic import get_critic
from tests.stubs import stub_observations, stub_value_net_config

def test_critic(): 
    value_net_config = stub_value_net_config()

    critic = get_critic(value_net_config=value_net_config)

    observations = stub_observations()
    data = TensorDict({'observation': observations}, batch_size=observations.shape[0])

    with torch.no_grad(): 
        critic(data)

    data_keys = data.keys()
    assert len(data_keys) == 2
    assert 'observation' in data_keys
    assert 'state_value' in data_keys
    assert torch.all(data['state_value'] > 0)