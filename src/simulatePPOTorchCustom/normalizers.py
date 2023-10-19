import numpy as np

import gym


def get_normalization_terms(env_name: str):
    match env_name:
        case 'InvertedPendulum-v4':
            return get_normalizer_for_inv_pendulum()
        case other:
            raise NotImplementedError('{} env normalization is not supported yet'.format(other))


def get_normalizer_for_inv_pendulum():
    env = gym.make(
        'InvertedPendulum-v4',
        max_episode_steps=100,
        autoreset=True,
    )
    env.reset(seed=0)
    rollout = np.array([env.step(np.array([0.0]))[0] for _ in range(1000)])
    mean = rollout.mean(axis=0)
    std = rollout.std(axis=0)
    mean[0] = 0.0  # this is x-coordinate of the cart
    std[0] = 1.0
    mean[2] = 0.0  # this is velocity of the cart
    std[1] = 1.0
    return mean, std




