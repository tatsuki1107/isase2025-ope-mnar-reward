from typing import Optional

import numpy as np
from sklearn.utils import check_random_state


def sample_slate_fast_with_replacement(action_dist: np.ndarray, random_state: Optional[int] = None) -> np.ndarray:
    random_ = check_random_state(random_state)
    uniform_rvs = random_.uniform(size=(action_dist.shape[0], action_dist.shape[2]))[:, np.newaxis]
    cum_action_dist = action_dist.cumsum(axis=1)
    flg = cum_action_dist > uniform_rvs
    sampled_action_at_k = flg.argmax(axis=1)

    return sampled_action_at_k
