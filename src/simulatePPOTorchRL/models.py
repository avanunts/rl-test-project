from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.data.tensor_specs import TensorSpec


def get_actor(policy_net: nn.Module, action_spec: TensorSpec):
    policy_net_wrapped = TensorDictModule(
        policy_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )

    return ProbabilisticActor(
        module=policy_net_wrapped,
        spec=action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": action_spec.space.minimum,  # this will raise an error if the spec is not bounded
            "max": action_spec.space.maximum,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )
