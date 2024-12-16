from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state

from db import get_rankings_given_keys
from ope import MarginalizedInversePropensityScore as MIPS
from utils import sample_slate_fast_with_replacement


def calc_importance_weights(data: dict, pi_e_c: np.ndarray) -> np.ndarray:
    rounds = np.arange(data["n_rounds"])
    position = np.arange(data["len_list"])[None, :]
    w_x_c = pi_e_c[rounds[:, None], data["action_context"], position] / data["pscore_c"]

    return w_x_c


@dataclass
class OffPolicyEvaluation:
    bandit_feedback: dict
    ope_estimators: list[MIPS]
    random_state: int = 12345

    def __post_init__(self) -> None:
        self.estimator_names = {estimator.estimator_name for estimator in self.ope_estimators}

    def _sample_rankings_by_target_policy(self, cluster_dist: np.ndarray) -> np.ndarray:
        clusters = sample_slate_fast_with_replacement(cluster_dist)
        random_ = check_random_state(self.random_state)

        action_set_given_cluster = self.bandit_feedback["action_set_given_cluster"]
        top_action = np.array(
            [random_.choice(list(action_set_given_cluster[cluster_])) for cluster_ in clusters[:, 0]]
        )[:, None]
        keys_ = np.concatenate([top_action, clusters[:, 1:]], axis=1)
        rankings = get_rankings_given_keys(
            keys=keys_,
            n_actions=self.bandit_feedback["n_actions"],
            n_clusters=self.bandit_feedback["n_clusters"],
        )

        return rankings

    def _create_estimator_inputs(
        self,
        cluster_dist: np.ndarray,
        rankings_sampled_by_pi_e: Optional[np.ndarray] = None,
        q_hat_dict: Optional[dict[str, np.ndarray]] = None,
    ) -> dict:
        input_data = dict()
        for estimator in self.ope_estimators:
            if "(DM)" in estimator.estimator_name:
                q_hat = q_hat_dict[estimator.estimator_name]
                q_hat_factual = q_hat[np.arange(len(q_hat))[:, None], rankings_sampled_by_pi_e]
                input_data[estimator.estimator_name] = {
                    "q_hat_factual": q_hat_factual,
                }
            else:
                weight_ = calc_importance_weights(
                    data=self.bandit_feedback,
                    pi_e_c=cluster_dist,
                )

                input_data[estimator.estimator_name] = {
                    "weight": weight_,
                    "reward": self.bandit_feedback["reward"],
                }
                if estimator.estimator_name == "MIPS (w/true ROIPS)":
                    input_data[estimator.estimator_name]["theta"] = self.bandit_feedback["true_theta_o_k_x"]

                elif estimator.estimator_name == "MIPS (w/heuristic ROIPS)":
                    input_data[estimator.estimator_name]["theta"] = (
                        self.bandit_feedback["heuristic_theta_o_k_x"] ** 0.95
                    )

        return input_data

    def estimate_policy_values(
        self, cluster_dist: np.ndarray, q_hat_dict: Optional[dict[str, np.ndarray]] = None
    ) -> dict:
        if any("(DM)" in name for name in self.estimator_names):
            rankings_sampled_by_pi_e = self._sample_rankings_by_target_policy(cluster_dist=cluster_dist)

        else:
            rankings_sampled_by_pi_e = None

        input_data = self._create_estimator_inputs(
            cluster_dist=cluster_dist, rankings_sampled_by_pi_e=rankings_sampled_by_pi_e, q_hat_dict=q_hat_dict
        )

        estimated_policy_values = dict()
        for estimator in self.ope_estimators:
            estimated_policy_value = estimator.estimate_policy_value(**input_data[estimator.estimator_name])
            estimated_policy_values[estimator.estimator_name] = estimated_policy_value

        return estimated_policy_values
