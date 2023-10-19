from collections import defaultdict
from tqdm import tqdm

import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

from . import envs, models, helpers
from ..utils.simulator_params import SimulatorParams
from ..utils.nets import get_policy_net, get_value_net, extract_models_params

LAMBDA = 1.0


def extract_env_params(params: SimulatorParams) -> envs.EnvParams:
    return envs.EnvParams(params.env_name, params.device, params.max_frames_per_episode)


def simulate(params: SimulatorParams):
    env_params = extract_env_params(params)
    env = envs.get_env(env_params)
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)  # init loc and scale for states normalization
    print('Normalization loc:   ' + str(env.transform[0].loc))
    print('Normalization scale: ' + str(env.transform[0].scale))
    check_env_specs(env)

    policy_params, value_params = extract_models_params(params, env.action_spec.shape[0])
    policy_net = get_policy_net(policy_params)
    actor = models.get_actor(policy_net, env.action_spec)
    critic = get_value_net(value_params)

    # init lazy layers
    actor(env.reset())
    critic(env.reset())

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=params.max_frames_per_episode,
        total_frames=params.max_frames_per_episode * params.num_episodes,
        split_trajs=False,
        device=params.device,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(params.max_frames_per_episode),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=params.learning_params.gamma, lmbda=LAMBDA, value_network=critic, average_gae=True
    )

    loss_module = ClipPPOLoss(
        actor=actor,
        critic=critic,
        clip_epsilon=params.learning_params.clip_epsilon,
        entropy_bonus=bool(params.learning_params.entropy_coef),
        entropy_coef=params.learning_params.entropy_coef,
        value_target_key=advantage_module.value_target_key,
        critic_coef=params.learning_params.ciric_loss_coef,
        gamma=params.learning_params.gamma,
        loss_critic_type=params.learning_params.loss_critic_type,
    )

    optim = torch.optim.Adam(loss_module.parameters(), params.learning_params.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, params.num_episodes, params.learning_params.cosine_lr_min_ratio * params.learning_params.lr
    )

    env.reset()

    logs = defaultdict(list)

    for i, tensordict_data in enumerate(collector):
        print('======================== i = {} =========================='.format(i))
        step_count = tensordict_data["step_count"].max().item()
        for _ in tqdm(range(params.learning_params.num_epochs)):
            with torch.no_grad():
                advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            num_batches = step_count // params.learning_params.batch_size
            if step_count % params.learning_params.batch_size > 0:
                num_batches += 1
            for _ in range(num_batches):
                subdata = replay_buffer.sample(params.learning_params.batch_size)
                loss_vals = loss_module(subdata.to(params.device))
                loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                )

                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), params.learning_params.max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        logs["lr"].append(optim.param_groups[0]["lr"])
        helpers.print_last_episode_logs(logs)
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                eval_rollout = env.rollout(1000, actor)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                del eval_rollout

        scheduler.step()
    return actor, critic, env, logs
