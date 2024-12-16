from copy import deepcopy
import itertools
import json
import sqlite3

import hydra
import numpy as np
from omegaconf import DictConfig


def setup_database(conn: sqlite3.Connection, table_name: str) -> None:
    cursor = conn.cursor()
    cursor.execute("PRAGMA synchronous = OFF;")
    cursor.execute("PRAGMA journal_mode = MEMORY;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            key1 INTEGER,
            key2 INTEGER,
            key3 INTEGER,
            key4 INTEGER,
            key5 INTEGER,
            value TEXT,
            PRIMARY KEY (key1, key2, key3, key4, key5)
        )
    """)
    conn.commit()


def generate_input_data(
    cfg: DictConfig, clusters: list[tuple[int, ...]], action_set_given_cluster: dict[int, set], top_action: int
) -> list[tuple[int, int, int, int, int, str]]:
    input_data = []
    above_action = top_action

    for cluster in clusters:
        action_set = deepcopy(action_set_given_cluster)
        ranking_action = {0: top_action}
        for pos_ in range(cfg.len_list - 1):
            random_ = np.random.RandomState(above_action)
            cluster_pos_ = cluster[pos_]
            action = random_.choice(list(action_set[cluster_pos_]))
            ranking_action[pos_ + 1] = int(action)
            action_set[cluster_pos_].remove(action)
            above_action = action

        key = [top_action] + list(cluster)
        value = json.dumps(ranking_action)
        input_data.append((*key, value))
        above_action = top_action

    return input_data


def insert_data(cursor: sqlite3.Cursor, table_name: str, input_data: list[tuple[int, int, int, int, int, str]]) -> None:
    try:
        cursor.executemany(
            f"""
            INSERT INTO {table_name} (key1, key2, key3, key4, key5, value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            input_data,
        )
    except sqlite3.IntegrityError as e:
        raise e
    except Exception as e:
        raise e


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.len_list != 5:
        raise NotImplementedError("len_list must be 5")

    actions_given_cluster = np.arange(cfg.n_actions).reshape(cfg.n_clusters, cfg.n_actions // cfg.n_clusters).tolist()
    action_set_given_cluster = {c: set(action_set_) for c, action_set_ in enumerate(actions_given_cluster)}
    clusters = list(itertools.product(range(cfg.n_clusters), repeat=cfg.len_list - 1))
    table_name = f"n_actions_{cfg.n_actions}_n_clusters_{cfg.n_clusters}_len_list_5"

    try:
        with sqlite3.connect("./db/ranking_action_spaces.db") as conn:
            setup_database(conn, table_name)
            cursor = conn.cursor()

            conn.execute("BEGIN TRANSACTION;")

            for top_action in range(cfg.n_actions):
                input_data = generate_input_data(cfg, clusters, action_set_given_cluster, top_action)
                try:
                    insert_data(cursor, table_name, input_data)
                except Exception as e:
                    conn.rollback()
                    raise e

            conn.commit()

    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
