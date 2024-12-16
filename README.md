# Off-Policy Evaluation for Recommendations from Missing-Not-At-Random rewards
This repository contains the experimental code for the paper titled “Off-Policy Evaluation for Recommendations from Missing-Not-At-Random Rewards,” which we intend to submit to [ISASE 2025](https://www.isase-ke.org/isase2025/), an international conference organized by the Japan Society of Kansei Engineering.

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

## Run Experiments
Our experimental script operates within a Docker container. Please ensure that [Docker Desktop](https://docs.docker.com/desktop/) is installed first. Then, build a Docker image using the `docker compose build` command.

```
docker compose build
```

・How does our estimator perform with varying the levels of bias in reward observations?
```
docker compose run --rm isase2025-ope main_alpha.py variation=alpha
```
A CSV file containing the experimental results and plot images will be saved in the `logs/main_alpha` directory.
