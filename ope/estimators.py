from dataclasses import dataclass

import numpy as np
from obp.ope import BaseOffPolicyEstimator


@dataclass
class MarginalizedInversePropensityScore(BaseOffPolicyEstimator):
    estimator_name: str

    def __post_init__(self) -> None:
        if self.estimator_name != "MIPS":
            raise ValueError("estimator_name must be 'MIPS'.")

    def _estimate_round_rewards(self, reward: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return (weight * reward).sum(1)

    def estimate_policy_value(self, reward: np.ndarray, weight: np.ndarray) -> np.float64:
        return self._estimate_round_rewards(reward=reward, weight=weight).mean()

    def estimate_interval(self):
        pass


@dataclass
class SampledDirectMethod(BaseOffPolicyEstimator):
    estimator_name: str

    def __post_init__(self) -> None:
        if self.estimator_name != "FM-IPS (DM)":
            raise NotImplementedError("estimator_name must be 'FM-IPS (DM)'.")

    def _estimate_round_rewards(self, q_hat_factual: np.ndarray) -> np.ndarray:
        return q_hat_factual.sum(1)

    def estimate_policy_value(self, q_hat_factual: np.ndarray) -> np.float64:
        return self._estimate_round_rewards(q_hat_factual=q_hat_factual).mean()

    def estimate_interval(self):
        pass


@dataclass
class MIPSwithRewardObservationIPS(MarginalizedInversePropensityScore):
    def __post_init__(self) -> None:
        if self.estimator_name not in {"MIPS (w/true ROIPS)", "MIPS (w/heuristic ROIPS)"}:
            raise ValueError("estimator_name must be 'MIPS (w/true ROIPS)' or 'MIPS (w/heuristic ROIPS)'.")

    def _estimate_round_rewards(self, reward: np.ndarray, weight: np.ndarray, theta: np.ndarray) -> np.ndarray:
        return (weight * reward / theta).sum(1)

    def estimate_policy_value(self, reward: np.ndarray, weight: np.ndarray, theta: np.ndarray) -> np.float64:
        return self._estimate_round_rewards(reward=reward, weight=weight, theta=theta).mean()
