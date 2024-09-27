from logging import Logger
from collections import defaultdict
from io import TextIOBase
from functools import reduce

import numpy as np
import pandas as pd

import torch

from tensordict.nn import ProbabilisticTensorDictSequential, TensorDictModule
from tensordict.tensordict import TensorDict

from simulatePPOTorchCustom.loss import ClippedPPOLoss



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


def to_pd(td: TensorDict):
    return pd.DataFrame({
            x: td[x].detach().numpy().tolist() for x in td.keys()
    })


def debug_write_input(debug_file: TextIOBase, td: TensorDict, num_iter: int):
    debug_file.write('Ran episode {}\n\n'.format(num_iter))
    debug_file.write(to_pd(td).to_markdown())
    debug_file.write("\n\n")


def debug_write_grads(debug_file: TextIOBase, loss_module: ClippedPPOLoss):
    total_grad_norm = 0.0
    debug_file.write('Gradients: \n\n')
    for name, param in loss_module.named_parameters():
        debug_file.write(name + "\n")
        grad = param.grad.detach()
        debug_file.write(str(grad) + "\n")
        grad_norm = torch.linalg.vector_norm(grad)
        debug_file.write(str(grad_norm) + "\n")
        total_grad_norm = grad_norm + total_grad_norm
    debug_file.write("Total grad norm: {}".format(total_grad_norm))


def debug_write_ppo_losses(debug_file: TextIOBase, ppo_loss_log: list[list[dict]]):
    debug_file.write('PPO objectives: \n\n')
    for i, losses in enumerate(ppo_loss_log):
        debug_file.write('Num epoch: {}\n'.format(i))
        avg_losses = {
                    'ppo_loss': 0.0,
                    'critic_loss': 0.0,
                    'entropy_loss': 0.0,
                    'count': 0,
                }
        for loss in losses:
            avg_losses['ppo_loss'] = (avg_losses['ppo_loss'] * avg_losses['count'] + loss['ppo_loss'] * loss['batch_size']) / (avg_losses['count'] + loss['batch_size'])
            avg_losses['critic_loss'] = (avg_losses['critic_loss'] * avg_losses['count'] + loss['critic_loss'] * loss['batch_size']) / (avg_losses['count'] + loss['batch_size'])
            avg_losses['entropy_loss'] = (avg_losses['entropy_loss'] * avg_losses['count'] + loss['entropy_loss'] * loss['batch_size']) / (avg_losses['count'] + loss['batch_size'])
            avg_losses['count'] = avg_losses['count'] + loss['batch_size']
        debug_file.write('Epoch loss: {}\n'.format(avg_losses))
