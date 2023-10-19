from dataclasses import dataclass

import gym


SUPPORTED_ENVS = {'InvertedPendulum-v4', 'InvertedDoublePendulum-v4'}


@dataclass
class EnvParams:
    env_name: str
    max_episode_steps: int


def get_env(params: EnvParams) -> gym.Env:
    if params.env_name not in SUPPORTED_ENVS:
        raise Exception('Env with name {} is not supported yet.'.format(params.env_name))
    return gym.make(
        params.env_name,
        max_episode_steps=params.max_episode_steps,
        autoreset=True,
    )