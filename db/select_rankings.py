import json
import sqlite3

import numpy as np


def _insert_temp_keys(cursor, keys):
    cursor.execute("""
        CREATE TEMPORARY TABLE temp_keys (
            key1 INTEGER,
            key2 INTEGER,
            key3 INTEGER,
            key4 INTEGER,
            key5 INTEGER
        );
    """)
    cursor.executemany(
        """
        INSERT INTO temp_keys (key1, key2, key3, key4, key5)
        VALUES (?, ?, ?, ?, ?);
    """,
        keys,
    )

    cursor.execute("CREATE INDEX idx_temp_keys ON temp_keys (key1, key2, key3, key4, key5);")


def get_rankings_given_keys(keys: np.ndarray, n_actions: int, n_clusters: int) -> np.ndarray:
    conn = sqlite3.connect("./db/ranking_action_spaces.db")
    cursor = conn.cursor()

    n_rounds, len_list = keys.shape
    table_name = f"n_actions_{n_actions}_n_clusters_{n_clusters}_len_list_{len_list}"

    query = f"""
        SELECT r.value
        FROM {table_name} r
        INNER JOIN temp_keys t
        ON r.key1 = t.key1
        AND r.key2 = t.key2
        AND r.key3 = t.key3
        AND r.key4 = t.key4
        AND r.key5 = t.key5;
    """

    keys = keys.tolist()

    try:
        cursor.execute("BEGIN TRANSACTION;")

        _insert_temp_keys(cursor, keys)

        cursor.execute(query)
        rows = cursor.fetchall()

    except sqlite3.Error as e:
        raise e
    finally:
        cursor.execute("DROP TABLE IF EXISTS temp_keys;")
        cursor.close()

    def extract_values(row):
        d = json.loads(row[0])
        return [d[str(i)] for i in range(len(d))]

    results = np.array(list(map(extract_values, rows)))

    return results


# test case
def main():
    n_actions, n_clusters = 500, 5

    n_rounds = 3
    top_actions = np.random.choice(n_actions, size=n_rounds)[:, None]
    clusters = np.random.choice(n_clusters, size=(n_rounds, 4))
    keys = np.concatenate([top_actions, clusters], axis=1)

    results = get_rankings_given_keys(keys, n_actions, n_clusters)
    print(results)
    return results


if __name__ == "__main__":
    main()
