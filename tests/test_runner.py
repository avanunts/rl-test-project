import numpy as np 

from simulatePPOTorchCustom.runner import Runner

from tests.stubs import stub_env, StubAgent

def test_runner(): 

    env = stub_env()
    runner = Runner(env=env)
    agent = StubAgent()

    # Test runner

    episode_output = runner.run_episode(agent=agent)
    data_dict, final_observation = episode_output.get_episode_dict()
    expected_observations = np.array([
        [ 0.00273923, -0.00460427, -0.00918053, -0.00966945],
        [ 0.00420045, -0.00925795,  0.0820031 , -0.22097192],
        [ 0.0070585 , -0.01704209,  0.06112823, -0.17049396],
        [ 0.01189845, -0.02950312,  0.18066873, -0.45107116],
        [ 0.02043229, -0.05062361,  0.24617554, -0.60691219],
        [ 0.02760058, -0.06901214,  0.11306493, -0.32031055],
        [ 0.0353857 , -0.0902442 ,  0.2758999 , -0.73940025],
        [ 0.04823738, -0.12486788,  0.36686593, -0.99471432],
        [ 0.06491749, -0.17049059,  0.46735779, -1.29052785]
    ])
    
    assert data_dict['observation'].shape == expected_observations.shape
    assert np.all(np.abs(data_dict['observation'] - expected_observations) < 1.0e-3)

    expected_actions = np.array([
        0.27395605, -0.06112156,  0.35859792,  0.19736803, -0.40582265, 
        0.47562235,  0.2611397 ,  0.28606431, -0.37188637
    ])

    assert data_dict['episode_action'].shape == expected_actions.shape
    assert np.all(np.abs(data_dict['episode_action'] - expected_actions) < 1.0e-3)

    assert final_observation is None

    # Test run_episode_sequence for 10 episodes
    episodes_outputs = runner.run_episode_sequence(
        agent=agent,
        target_num_frames=None, 
        num_episodes=10
    )

    assert len(episodes_outputs) == 10
    expected_episodes_lengths = [16, 10, 10, 18, 22, 10, 22, 17, 10, 14]
    for i, expected_episode_length in enumerate(expected_episodes_lengths): 
        assert len(episodes_outputs[i].observations) == expected_episode_length

    # Test run_episode_sequence for 1_000 frames

    episodes_outputs = runner.run_episode_sequence(
        agent=agent, 
        target_num_frames=1_000,
        num_episodes=None,
    )

    assert len(episodes_outputs) == 70 
    assert sum([len(x.observations) for x in episodes_outputs]) == 1_000 

    # Test case truncate 

    env = stub_env(max_episode_steps=3)
    runner = Runner(env=env)

    episode_output = runner.run_episode(agent=agent)
    assert len(episode_output.observations) == 3
    expected_final_observation = np.array([-0.01670276,  0.03533848, -0.19706525,  0.40294135])
    assert np.all(np.abs(episode_output.final_observation - expected_final_observation) < 1.0e-3)
