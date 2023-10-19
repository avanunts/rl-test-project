from logging import Logger
from collections import defaultdict

import numpy as np

from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule


def print_normalization_terms(mean: np.array, std: np.array):
    print('''
    Observation mean: {}
    Observation std:  {}
'''.format(mean, std)
          )


def log_actor_layers_devices(actor: ProbabilisticTensorDictSequential, logger: Logger):
    for name, param in actor[0].named_parameters():
        logger.debug('Params named {} are stored on device {}'.format(name, param.device))


def log_critic_layers_devices(critic: TensorDictModule, logger: Logger):
    for name, param in critic.named_parameters():
        logger.debug('Params named {} are stored on device {}'.format(name, param.device))


def print_stats_for_episode(loop_log: defaultdict):
    print(
'''
    Episode num:        {}
    Episode step_count: {}
    Episode lr:         {}
    Episode avg reward: {}
'''.format(len(loop_log['avg_reward']), loop_log['step_count'][-1], loop_log['lr'][-1], loop_log['avg_reward'][-1])
          )