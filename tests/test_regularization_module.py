import numpy as np 
from tensordict import TensorDict

from simulatePPOTorchCustom.configs import LocScaleRegularizationConfig
from simulatePPOTorchCustom.train_actor import LocScaleRegularizationDictModule


def test_regularization_module(): 

    config = LocScaleRegularizationConfig(
        loc_soft_max=1.0, 
        loc_soft_min=-1.0, 
        scale_soft_max=0.5, 
    )

    loc_scale_regularization_module = LocScaleRegularizationDictModule(loc_scale_regulariation_config=config)

    input = TensorDict({
        'loc': np.array([-0.8909]), 
        'scale': np.array([1.9757]),
        }, 
        batch_size=1
    )

    loc_scale_regularization_module(input)

    print(input['loc_scale_regulariztion'])
    print(input)