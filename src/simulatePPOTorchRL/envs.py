from dataclasses import dataclass
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)


SUPPORTED_ENVS = {'InvertedPendulum-v4', 'InvertedDoublePendulum-v4'}
FRAME_SKIP = 1


@dataclass
class EnvParams:
    env_name: str
    device: str
    max_frames_per_episode: int


def get_env(params: EnvParams):
    if params.env_name not in SUPPORTED_ENVS:
        raise Exception('Env {} is not supported'.format(params.env_name))
    base_env = GymEnv(params.env_name, device=params.device, frame_skip=FRAME_SKIP)
    return TransformedEnv(
        base_env,
        Compose(
            # normalize observations for neural nets input
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(
                in_keys=["observation"],
            ),
            StepCounter(max_steps=params.max_frames_per_episode),
        ),
    )
