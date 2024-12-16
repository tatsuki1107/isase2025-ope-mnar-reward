from dataclasses import dataclass

import numpy as np
from obp.dataset import BaseBanditDataset
from obp.dataset import logistic_reward_function
from obp.utils import sample_action_fast
from sklearn.utils import check_random_state

from db import get_rankings_given_keys
from policy import logging_policy
from policy import target_policy
from utils import sample_slate_fast_with_replacement


def gen_decomposed_expected_reward(g_x_c: np.ndarray, h_x_a: np.ndarray, lam: float) -> np.ndarray:
    return (1 - lam) * g_x_c + lam * (h_x_a + (h_x_a**2) + (h_x_a**3))


def linear_observation_distribution(
    context: np.ndarray, all_observation: np.ndarray, alpha: float, random_state: int
) -> np.ndarray:
    len_list = all_observation.shape[0]
    dim_context = context.shape[1]

    random_ = check_random_state(random_state)
    theta_coef = random_.normal(size=(len_list, dim_context))

    bar_o = all_observation.sum(1)
    logits = np.abs(context @ theta_coef) * (alpha ** (len_list - bar_o))
    theta_o_x = logits / logits.sum(1)[:, None]

    return theta_o_x


@dataclass
class SyntheticDataset(BaseBanditDataset):
    n_actions: int
    n_clusters: int
    dim_context: int
    len_list: int
    reward_noise: float
    beta: float
    alpha: float
    random_state: int
    eps: float
    lam: float = 0.0
    delta: float = 1.0
    examination: str = "tri"

    def __post_init__(self) -> None:
        self.random_ = check_random_state(self.random_state)

        # define the cluster of the action.
        self.n_actions_given_cluster = self.n_actions // self.n_clusters
        self.actions_given_cluster = np.arange(self.n_actions).reshape(self.n_clusters, self.n_actions_given_cluster)
        self.action_set_given_cluster = {
            c: set(action_set_) for c, action_set_ in enumerate(self.actions_given_cluster)
        }
        self.e_a = np.repeat(np.arange(self.n_clusters), self.n_actions_given_cluster)

        self.onehot_action = np.eye(self.n_actions)
        self.onehot_cluster = np.eye(self.n_clusters)

        # set of missing-not-at-random rewards
        if self.examination == "tri":
            self.all_observation = np.tri(self.len_list, dtype=int)

        else:
            raise NotImplementedError

    def obtain_batch_bandit_feedback(self, n_rounds: int, is_online: bool = False) -> dict:
        # x ~ p(x)
        context = self.random_.normal(size=(n_rounds, self.dim_context))

        # r
        g_x_c = logistic_reward_function(
            context=context,
            action_context=self.onehot_cluster,
            random_state=self.random_state,
        )
        tiled_g_x_c = np.tile(g_x_c[:, :, None], reps=self.len_list)
        h_x_a = logistic_reward_function(
            context=context,
            action_context=self.onehot_action,
            random_state=self.random_state,
        )

        # c ~ \pi(\cdot|x)
        pi_b_c = target_policy(tiled_g_x_c, eps=self.eps) if is_online else logging_policy(tiled_g_x_c, beta=self.beta)
        clusters = sample_slate_fast_with_replacement(pi_b_c)
        rounds = np.arange(n_rounds)[:, None]
        pscores_c = pi_b_c[rounds, clusters, np.arange(self.len_list)[None, :]]

        # a ~ \pi(a|x) = \sum_{c} \pi(a|x,c)\pi(c|x)
        top_action = np.array(
            [self.random_.choice(list(self.action_set_given_cluster[cluster])) for cluster in clusters[:, 0]]
        )[:, None]
        keys_ = np.concatenate([top_action, clusters[:, 1:]], axis=1)

        rankings = get_rankings_given_keys(keys=keys_, n_actions=self.n_actions, n_clusters=self.n_clusters)

        # o ~ \theta(\cdot|x)
        theta_o_x = linear_observation_distribution(
            context=context,
            all_observation=self.all_observation,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        observation_idx = sample_action_fast(theta_o_x)
        observations = self.all_observation[observation_idx]

        true_theta_o_k_x = (theta_o_x[:, :, None] * self.all_observation).sum(1)
        heuristic_theta_o_k_x = np.tile(observations.mean(0), reps=(n_rounds, 1))

        # conjunct effect decomposition
        g_x_c_factual = g_x_c[rounds, clusters]
        h_x_a_factual = h_x_a[rounds, rankings]
        q_x_a_factual = gen_decomposed_expected_reward(g_x_c=g_x_c_factual, h_x_a=h_x_a_factual, lam=self.lam)

        # r_i ~ p(r_i|x_i, a_i, e_i)
        reward = self.random_.normal(q_x_a_factual, scale=self.reward_noise)
        # missing-not-at-random rewards
        reward = reward * observations

        return dict(
            n_rounds=n_rounds,
            len_list=self.len_list,
            n_actions=self.n_actions,
            n_clusters=self.n_clusters,
            context=context,
            action=rankings,
            action_context=clusters,
            e_a=self.e_a,
            true_theta_o_k_x=true_theta_o_k_x,
            heuristic_theta_o_k_x=heuristic_theta_o_k_x,
            observation=observations.astype(bool),
            reward=reward,
            pscore_c=pscores_c,
            evaluation_policy_logit=tiled_g_x_c,
            expected_reward_factual=q_x_a_factual,
            pi_b_c=pi_b_c,
            action_set_given_cluster=self.action_set_given_cluster,
        )
