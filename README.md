# Off-Policy Evaluation for Recommendations with Missing-Not-At-Random Rewards
This repository contains the experimental code for the paper titled “Off-Policy Evaluation for Recommendations with Missing-Not-At-Random Rewards,” which we are contributing to [ISASE 2025](https://www.isase-ke.org/isase2025/), an international conference organized by the Japan Society of Kansei Engineering.  

Preprint: https://arxiv.org/abs/2502.08993  

## Abstract
Unbiased recommender learning (URL) and off-policy evaluation/learning (OPE/L) techniques are effective in addressing the data bias caused by display position and logging policies, thereby consistently improving the performance of recommendations. However, when both bias exits in the logged data, these estimators may suffer from significant bias. In this study, we first analyze the position bias of the OPE estimator when rewards are missing not at random. To mitigate both biases, we propose a novel estimator that leverages two probabilities of logging policies and reward observations as propensity scores. Our experiments demonstrate that the proposed estimator achieves superior performance compared to other estimators, even as the levels of bias in reward observations increases.

## Main Package Version
We use [Poetry](https://python-poetry.org/) to manage versions of Python and its libraries. The primary versions are as follows (see `pyproject.toml` for details).
```
[tool.poetry.dependencies]
python = ">=3.9,<3.10"
obp = "^0.5.5"
numpy = "^1.22.5"
scikit-learn = "1.1.3"
seaborn = "^0.11.2"
matplotlib = "3.7.3"
hydra-core = "^1.3.2"
```

## Synthetic Experiments
Our experimental script operates within a Docker container. Please ensure that [Docker Desktop](https://docs.docker.com/desktop/) is installed first. Then, build a Docker image using the `docker compose build` command.

```
docker compose build
```
### Define Ranking Action Spaces
Ensure that the file `db/ranking_action_spaces.db` exists. If necessary, you can define the ranking action spaces to be used for our experiment using the following command:

```
docker compose run --rm isase2025-ope db/insert.py
```

### Run Experimental Script

・How does our estimator perform with varying the levels of bias in reward observations?
```
docker compose run --rm isase2025-ope main_alpha.py variation=alpha
```
A CSV file containing the experimental results and plot images will be saved in the `logs/main_alpha` directory.
