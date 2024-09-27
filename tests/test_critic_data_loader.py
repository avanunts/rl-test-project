import torch

from src.simulatePPOTorchCustom.train_critic import critic_dataloader
from src.simulatePPOTorchCustom.actor import get_actor
from src.simulatePPOTorchCustom.critic import get_critic
from src.simulatePPOTorchCustom.runner import run_episodes_sequence

from tests.stubs import stub_actor_config, stub_value_net_config, stub_env

def test_critic_data_loader(): 
    actor_config = stub_actor_config()
    value_net_config = stub_value_net_config()

    torch.manual_seed(0)
    actor = get_actor(config=actor_config)
    critic = get_critic(value_net_config=value_net_config)

    env = stub_env()
    num_episodes = 1_000
    stacked_trajectories = run_episodes_sequence(
        actor=actor, 
        critic=critic,
        env=env,
        num_episodes=num_episodes
    )
    observations = stacked_trajectories['observation']
    tail_reward = stacked_trajectories['tail_reward']

    batch_size = 512
    data_loader = critic_dataloader(observations, tail_reward, batch_size)

    num_iterations = 5
    current_iter = 0
    for observation_batch, tail_reward_batch in data_loader: 
        if current_iter > num_iterations: 
            break
        assert len(observation_batch.shape) == 2
        assert observation_batch.shape[0] == batch_size
        assert observation_batch.shape[1] == 4
        assert len(tail_reward_batch.shape) == 1
        assert tail_reward_batch.shape[0] == batch_size
        current_iter += 1

    

