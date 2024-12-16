import logging
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
import pandas as pd
from tqdm import tqdm

from dataset import SyntheticDataset
from ope import MIPSwithRewardObservationIPS as MIPSwithROIPS
from ope import MarginalizedInversePropensityScore as MIPS
from ope import OffPolicyEvaluation
from ope import PointwiseRecommender
from ope import SampledDirectMethod as DM
from policy import target_policy
from utils import aggregate_simulation_results
from utils import visualize_mean_squared_error


TQDM_FORMAT = "{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

cs = ConfigStore.instance()

logger = logging.getLogger(__name__)

ope_estimators = [
    DM(estimator_name="FM-IPS (DM)"),
    MIPS(estimator_name="MIPS"),
    MIPSwithROIPS(estimator_name="MIPS (w/true ROIPS)"),
    MIPSwithROIPS(estimator_name="MIPS (w/heuristic ROIPS)"),
]


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg) -> None:
    logger.info("start the experiment.")
    logger.info(cfg)

    result_path = Path(HydraConfig.get().run.dir)
    result_path.mkdir(parents=True, exist_ok=True)

    result_df_list = []
    for alpha in cfg.variation.alpha_list:
        dataset = SyntheticDataset(
            n_actions=cfg.n_actions,
            dim_context=cfg.dim_context,
            n_clusters=cfg.n_clusters,
            beta=cfg.beta,
            alpha=alpha,
            len_list=cfg.len_list,
            random_state=cfg.random_state,
            reward_noise=cfg.reward_noise,
            eps=cfg.eps,
        )

        # calculate ground truth policy value (on policy)
        test_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.test_size, is_online=True)
        policy_value = test_data["expected_reward_factual"].sum(1).mean()

        message = f"alpha: {alpha}"
        tqdm_ = tqdm(range(cfg.n_val_seeds), desc=message, bar_format=TQDM_FORMAT)
        result_list = []
        for seed in tqdm_:
            val_data = dataset.obtain_batch_bandit_feedback(n_rounds=cfg.val_size)

            # target policy
            cluster_dist = target_policy(val_data["evaluation_policy_logit"], eps=cfg.eps)
            # off policy evaluation
            ope = OffPolicyEvaluation(
                bandit_feedback=val_data,
                ope_estimators=ope_estimators,
                random_state=cfg.random_state,
            )

            if seed % 10 == 0:
                fm_ips = PointwiseRecommender(
                    dim_context=cfg.dim_context,
                    n_actions=cfg.n_actions,
                    len_list=cfg.len_list,
                    n_clusters=cfg.n_clusters,
                    q_hat_name="factorization_machine",
                    max_iter=cfg.max_iter,
                )
                fm_ips.fit(val_data)
                q_hat_ips = fm_ips.predict()

            else:
                input_data = fm_ips.create_test_data(
                    test_context=val_data["context"],
                    test_e_a=val_data["e_a"],
                )
                q_hat_ips = fm_ips.predict(input_data)

            q_hat_dict = {
                "FM-IPS (DM)": q_hat_ips,
            }

            estimated_policy_values = ope.estimate_policy_values(cluster_dist=cluster_dist, q_hat_dict=q_hat_dict)
            result_list.append(estimated_policy_values)

        logger.info(tqdm_)
        # calculate MSE
        result_df = aggregate_simulation_results(
            simulation_result_list=result_list, policy_value=policy_value, x_value=alpha
        )
        result_df_list.append(result_df)

    result_df = pd.concat(result_df_list).reset_index(level=0)
    result_df.to_csv(result_path / "result.csv")

    visualize_mean_squared_error(
        result_df=result_df,
        xlabel="levels of bias in reward observations",
        img_path=result_path,
        xscale="linear",
    )

    logger.info("finish experiment...")


if __name__ == "__main__":
    main()
