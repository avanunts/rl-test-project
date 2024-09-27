import gym
import torch
from tensordict import TensorDict

from src.simulatePPOTorchCustom.critic import get_critic
from src.simulatePPOTorchCustom.actor import get_actor
from src.simulatePPOTorchCustom.runner import run_episode

from tests.stubs import stub_value_net_config, stub_actor_config, stub_observations, stub_env
from tests.utils import must_coincide

def test_runner(): 
    actor_config = stub_actor_config()
    value_net_config = stub_value_net_config()

    torch.manual_seed(0)
    actor = get_actor(config=actor_config)
    critic = get_critic(value_net_config=value_net_config)

    env = stub_env()
    init_obs, _ = env.reset(seed=42)

    all_frames_data, final_value = run_episode(actor=actor, critic=critic, env=env, init_gym_obs=init_obs)

    all_frames_data_keys = all_frames_data.keys()
    assert len(all_frames_data_keys) == 7
    assert 'observation' in all_frames_data_keys
    assert 'episode_loc' in all_frames_data_keys
    assert 'episode_scale' in all_frames_data_keys
    assert 'episode_sample_log_prob' in all_frames_data_keys
    assert 'episode_action' in all_frames_data_keys
    assert 'episode_state_value' in all_frames_data_keys
    assert 'reward' in all_frames_data_keys

    expected_observations = torch.Tensor([
        [ 0.00548, -0.00122, 0.00717, 0.00395],
        [ 0.01440, -0.02101, 0.43777, -0.98310],
        [ 0.04194, -0.08286, 0.93831, -2.10468],
        [ 0.07745, -0.16179, 0.83872, -1.86084]
    ])
    must_coincide(all_frames_data['observation'], expected_observations)

    expected_final_value = torch.Tensor([[0.0]])
    must_coincide(final_value, expected_final_value)

def test_runner_if_truncate(): 
    actor_config = stub_actor_config()
    value_net_config = stub_value_net_config()

    torch.manual_seed(0)
    actor = get_actor(config=actor_config)
    critic = get_critic(value_net_config=value_net_config)

    env = stub_env(max_episode_steps=2)
    init_obs, _ = env.reset(seed=42)

    _, final_value = run_episode(actor=actor, critic=critic, env=env, init_gym_obs=init_obs)

    expected_final_value = torch.Tensor([[0.1110]])
    must_coincide(final_value, expected_final_value)

def test_reproducibility_actor_critic(): 
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
        [ 1.2983],
        [-0.6063],
        [-0.2787],
        [ 0.6396],
        [-0.1010]
    ])
    
    must_coincide(action, expected_action)

    state_value = data['state_value']
    expected_state_value = torch.Tensor([
        [0.1743],
        [0.1127],
        [0.1587],
        [0.0227],
        [0.2209]
    ])

    must_coincide(state_value, expected_state_value)

