from abc import ABCMeta
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


@dataclass
class BatchDataset(Dataset):
    context: torch.Tensor
    observation: torch.Tensor
    reward: torch.Tensor
    pscore: torch.Tensor

    def __post_init__(self):
        assert self.context.shape[0] == self.reward.shape[0] == self.pscore.shape[0] == self.observation.shape[0]

    def __getitem__(self, index):
        return (
            self.context[index],
            self.observation[index],
            self.reward[index],
            self.pscore[index],
        )

    def __len__(self):
        return self.context.shape[0]


class FactorizationMachine(nn.Module):
    def __init__(self, dim_input, dim_latent):
        super().__init__()
        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.randn(dim_input))
        self.v = nn.Parameter(torch.randn(dim_input, dim_latent))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_sparse:
            raise ValueError("x must be a sparse tensor")

        linear_terms = self.w0 + torch.sparse.mm(x, self.w.unsqueeze(1)).squeeze(1)
        # v: (input_dim, factor_dim)
        # x * v: (batch_size, factor_dim)
        xv = torch.sparse.mm(x, self.v)  # (batch_size, factor_dim)
        xv_square = xv.pow(2)

        v_square = self.v.pow(2)
        x_v_square = torch.sparse.mm(x, v_square)  # (batch_size, factor_dim)

        # Sum over the factor dimensions
        interaction_term = 0.5 * (torch.sum(xv_square, dim=1) - torch.sum(x_v_square, dim=1))  # Shape: (batch_size,)

        output = linear_terms + interaction_term
        return output


@dataclass
class BaseRecommender(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, **kwargs):
        pass


@dataclass
class PointwiseRecommender(BaseRecommender):
    dim_context: int
    n_actions: int
    len_list: int
    n_clusters: int
    method: str = "ips"  # "ips" or "naive"
    dim_latent: int = 5
    max_iter: int = 5
    batch_size: int = 100
    q_hat_name: str = "factorization_machine"
    learning_rate_init: float = 0.03
    gamma: float = 0.98
    alpha: float = 0.8
    solver: str = "adam"

    def __post_init__(self) -> None:
        self.train_loss = []
        self.test_value = []

        if self.q_hat_name == "factorization_machine":
            self.dim_input = self.dim_context + self.n_actions + self.n_clusters
            self.reg_model = FactorizationMachine(self.dim_input, self.dim_latent)

        else:
            raise NotImplementedError

    def fit(self, train_data: dict) -> None:
        train_data_loader = self._create_train_data(train_data)
        self.batch_test_input_data = self.create_test_data(
            test_context=train_data["context"],
            test_e_a=train_data["e_a"],
        )

        scheduler, optimizer = self._init_scheduler()
        for _ in range(self.max_iter):
            loss_epochs = 0.0
            self.reg_model.train()
            for input_data_, observation_, reward_, pscore_ in train_data_loader:
                optimizer.zero_grad()
                q_hat = self.reg_model(input_data_)
                loss = (((reward_[observation_] - q_hat[observation_]) ** 2) / pscore_[observation_]).mean()
                loss.backward()
                optimizer.step()
                loss_epochs += loss.item()

            self.train_loss.append(loss_epochs / len(train_data_loader))
            scheduler.step()

        # print(self.train_loss)

    def predict(self, batch_input_data: Optional[torch.Tensor] = None) -> np.ndarray:
        if batch_input_data is None:
            batch_input_data = self.batch_test_input_data

        self.reg_model.eval()
        with torch.no_grad():
            q_hat = self.reg_model(batch_input_data).numpy()
        q_hat = q_hat.reshape(-1, self.n_actions)

        return q_hat

    def _init_scheduler(self) -> tuple[ExponentialLR, optim.Optimizer]:
        if self.solver == "adagrad":
            optimizer = optim.Adagrad(
                self.reg_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        elif self.solver == "adam":
            optimizer = optim.AdamW(
                self.reg_model.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        else:
            raise NotImplementedError("`solver` must be one of 'adam' or 'adagrad'")

        scheduler = ExponentialLR(optimizer, gamma=self.gamma)

        return scheduler, optimizer

    def _create_train_data(self, train_data: dict) -> DataLoader:
        observation = train_data["observation"].flatten()
        context = train_data["context"]
        action = train_data["action"].flatten()
        reward = train_data["reward"].flatten()
        pscore = train_data["heuristic_theta_o_k_x"].flatten()
        if self.method == "naive":
            pscore = np.ones_like(pscore)
        cluster = train_data["action_context"].flatten()

        n_trains = context.shape[0] * self.len_list
        col = np.repeat(np.arange(n_trains), repeats=self.dim_context + 2)

        row_context = np.tile(np.arange(self.dim_context), reps=(n_trains, 1))
        row_action = action[:, None] + self.dim_context
        row_cluster = cluster[:, None] + self.dim_context + self.n_actions
        row = np.concatenate([row_context, row_action, row_cluster], axis=1).flatten()

        indices = np.array([col, row])
        indices = torch.from_numpy(indices).long()

        values = np.ones((n_trains, self.dim_context + 2))
        values[:, : self.dim_context] = np.repeat(context, self.len_list, axis=0)
        values = torch.from_numpy(values.flatten()).float()

        input_data = torch.sparse_coo_tensor(indices, values, [n_trains, self.dim_input])

        dataset = BatchDataset(
            input_data,
            torch.from_numpy(observation).bool(),
            torch.from_numpy(reward).float(),
            torch.from_numpy(pscore).float(),
        )

        data_loader = DataLoader(dataset, batch_size=self.batch_size)

        return data_loader

    def create_test_data(self, test_context: np.ndarray, test_e_a: np.ndarray) -> torch.Tensor:
        n_rounds = test_context.shape[0]
        values = np.ones((n_rounds, self.dim_context + 2))
        values[:, : self.dim_context] = test_context
        values = np.repeat(values.flatten(), self.n_actions)

        col = np.repeat(np.arange(n_rounds * self.n_actions), repeats=self.dim_context + 2)

        row = np.zeros((n_rounds * self.n_actions, self.dim_context + 2), dtype=int)
        row_context = np.tile(np.arange(self.dim_context), reps=(n_rounds * self.n_actions, 1))
        row[:, : self.dim_context] = row_context

        row_actions = np.arange(self.dim_context, self.n_actions + self.dim_context)[:, None]
        row_clusters = test_e_a[:, None] + self.dim_context + self.n_actions
        row_a_c = np.concatenate([row_actions, row_clusters], axis=1)
        row[:, self.dim_context :] = np.tile(row_a_c, reps=(n_rounds, 1))
        row = row.flatten()

        indices = np.array([col, row])
        indices = torch.from_numpy(indices).long()
        values = torch.from_numpy(values).float()

        test_input_data = torch.sparse_coo_tensor(
            indices, values, [n_rounds * self.n_actions, self.dim_context + self.n_actions + self.n_clusters]
        )

        return test_input_data
