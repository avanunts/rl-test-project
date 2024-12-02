import torch

from src.simulatePPOTorchCustom.train_critic import get_critic_dataloader
from src.simulatePPOTorchCustom.data_utils import build_dataset
from simulatePPOTorchCustom.runner import Runner

from tests.stubs import stub_env, StubAgent
from tests.utils import must_coincide

def test_build_critic_dataset(): 

    env = stub_env(max_episode_steps=10)
    runner = Runner(env=env)
    agent = StubAgent()

    episodes_outputs = runner.run_episode_sequence(
        agent=agent,
        target_num_frames=None, 
        num_episodes=20
    )

    episodes_data, final_observations = [], []
    for output in episodes_outputs: 
        episode_data, final_observation = output.get_episode_dict()
        episodes_data.append(episode_data), final_observations.append(final_observation)

    gamma = 0.999
    critic_dataset = build_dataset(episodes_data=episodes_data, final_observations=final_observations, gamma=gamma)
    
    assert len(critic_dataset.keys()) == 6
    assert 'observation' in critic_dataset.keys()
    assert 'episode_sample_log_prob' in critic_dataset.keys()
    assert 'episode_action' in critic_dataset.keys()
    assert 'tail_reward' in critic_dataset.keys()
    assert 'final_observation' in critic_dataset.keys()
    assert 'final_observation_coeff' in critic_dataset.keys()

    episode_start_index = 0
    for episode_data, final_observation in zip(episodes_data, final_observations): 
        episode_end_index = episode_start_index + len(episode_data['observation'])
        episode_tail_reward = critic_dataset['tail_reward'][episode_start_index: episode_end_index]
        tail_reward_is_ok(episode_tail_reward, gamma)
        if final_observation is None: 
            coeffs_in_episode = critic_dataset[episode_start_index: episode_end_index]['final_observation_coeff']
            coeffs_must_be = torch.zeros(size=(episode_end_index - episode_start_index, ))
            must_coincide(coeffs_in_episode, coeffs_must_be) 

            must_coincide(
                critic_dataset[episode_start_index: episode_end_index]['final_observation'], 
                torch.zeros(size=(episode_end_index - episode_start_index, 4))
            )

        else: 
            last_coeff = critic_dataset[episode_start_index: episode_end_index]['final_observation_coeff'][-1].float()
            must_be_last_coeff = gamma
            assert abs(last_coeff - must_be_last_coeff) < 1.0e-7

            first_coeff = critic_dataset[episode_start_index: episode_end_index]['final_observation_coeff'][0].float()
            must_be_first_coeff = gamma ** (episode_end_index - episode_start_index)
            assert abs(first_coeff - must_be_first_coeff) < 1.0e-7

            for index in range(episode_start_index, episode_end_index): 
                final_observation_from_dataset = critic_dataset[index]['final_observation']
                must_coincide(torch.Tensor(final_observation), final_observation_from_dataset)

        
        episode_start_index = episode_end_index

    data_loader = get_critic_dataloader(critic_dataset=critic_dataset, batch_size=16)
    num_batch = 0
    for data_batch in data_loader: 
        if num_batch > 3: 
            break
        num_batch += 1
        assert data_batch.shape[0] == 16
        assert len(data_batch.keys()) == 6
        assert 'observation' in data_batch.keys()
        assert 'episode_sample_log_prob' in data_batch.keys()
        assert 'episode_action' in data_batch.keys()
        assert 'tail_reward' in data_batch.keys()
        assert 'final_observation' in data_batch.keys()
        assert 'final_observation_coeff' in data_batch.keys()


def tail_reward_is_ok(tail_reward: torch.Tensor, gamma: float): 
    for i in range(tail_reward.shape[0] - 1): 
        assert abs(tail_reward[i].float() - gamma * tail_reward[i + 1].float() - 1) < 1.0e-5
    

