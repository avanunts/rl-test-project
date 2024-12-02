from typing import Tuple
import numpy as np 

def get_tanh_standard_normal_mean_and_std() -> Tuple[float, float]: 
    rng = np.random.default_rng(seed=42)
    standard_normal_samples = rng.standard_normal(size=1_000_000)
    tanh_standard_normal_samples = np.tanh(standard_normal_samples)
    mean = tanh_standard_normal_samples.mean()
    std = tanh_standard_normal_samples.std()
    return mean, std
