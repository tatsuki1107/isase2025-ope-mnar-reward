import numpy as np
from obp.utils import softmax
from sklearn.utils import check_random_state


def target_policy(
    expected_reward: np.ndarray,
    is_optimal: bool = True,
    eps: float = 0.3,
) -> np.ndarray:
    base_pol = np.zeros_like(expected_reward)
    a = np.argmax(expected_reward, axis=1) if is_optimal else np.argmin(expected_reward, axis=1)
    base_pol[
        np.arange(expected_reward.shape[0])[:, None],
        a,
        np.arange(expected_reward.shape[2])[None, :],
    ] = 1
    pol = (1.0 - eps) * base_pol
    pol += eps / expected_reward.shape[1]

    return pol


def logging_policy(
    q_func: np.ndarray,
    beta: float,
    sigma: float = 1.0,
    lam: float = 0.0,
    random_state: int = 12345,
) -> np.ndarray:
    random_ = check_random_state(random_state)
    noise = random_.normal(scale=sigma, size=q_func.shape)
    pi = softmax(beta * (lam * q_func + (1.0 - lam) * noise))
    return pi
