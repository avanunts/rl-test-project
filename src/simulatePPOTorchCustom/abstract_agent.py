from typing import Tuple

import numpy as np 

class AbstractAgent: 
    def __init__(self) -> None:
        pass

    def sample_action(self, observation: np.ndarray) -> Tuple[float, float]:
        pass
