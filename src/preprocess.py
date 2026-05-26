"""CLI replica of CAP4773_fin_1.ipynb preprocessing.

Reads dataset_godknowswhat.csv, builds the 9-tweet-window matrix, and writes:
    filtered_input_matrix.csv  -- 28 columns (9 tweets x 3 features + 1 target)
    user_ids.csv               -- parallel user_id per row

Mirrors the notebook cells exactly so results match (including the existing
split_group slicing quirk -- fix in a follow-up if/when desired).
"""
from __future__ import annotations

import pandas as pd

from paths import DATA_INTERIM

INPUT_CSV = DATA_INTERIM / "dataset_godknowswhat.csv"
MATRIX_OUT = DATA_INTERIM / "filtered_input_matrix.csv"
USERS_OUT = DATA_INTERIM / "user_ids.csv"

WINDOW = 10  # 9 history tweets + 1 target
EXPECTED_ROW_LEN = 3 * (WINDOW - 1) + 1  # 28


def overnight_tweet(hour: int) -> int:
    if hour in range(5) or hour == 23:
        return 1
    return 0


def split_group(group: pd.DataFrame) -> list[pd.DataFrame]:
    num_tweets = len(group)
    num_batches = num_tweets // WINDOW
    batches: list[pd.DataFrame] = []
    for i in range(num_batches):
        batches.append(group.iloc[i * WINDOW : (i + 1) * WINDOW])
    remaining_rows = num_tweets % WINDOW
    if remaining_rows > 0:
        last_batch = group.iloc[-num_batches * WINDOW :]
        for _ in range(WINDOW - remaining_rows):
            last_batch = pd.concat(
                [last_batch, last_batch.iloc[[-1]]], ignore_index=True
            )
        batches.append(last_batch)
    return batches


def generate_row(batch: pd.DataFrame) -> list:
    row: list = []
    for _, row_data in batch.iloc[:-1].iterrows():
        row.extend(row_data[1:].tolist())  # skip user_id
    row.append(batch.iloc[-1]["emotion_encoded"])
    return row


def main() -> None:
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows")

    df["post_created"] = pd.to_datetime(df["post_created"])
    df["hour_of_day"] = df["post_created"].dt.hour
    df["overnight"] = df["hour_of_day"].apply(overnight_tweet)

    df = df.groupby("user_id").filter(lambda x: 20 <= len(x) <= 1000)
    print(f"After 20-1000 tweet/user filter: {len(df):,} rows, "
          f"{df['user_id'].nunique()} users")

    df["weekend"] = df["post_created"].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    df = df.sort_values(by="post_created")
    df = df[["user_id", "emotion_encoded", "weekend", "overnight"]]

    new_rows: list[list] = []
    user_ids: list = []
    for user_id, group in df.groupby("user_id"):
        for batch in split_group(group):
            row = generate_row(batch)
            if len(row) == EXPECTED_ROW_LEN:
                new_rows.append(row)
                user_ids.append(user_id)

    new_df = pd.DataFrame(new_rows, columns=range(EXPECTED_ROW_LEN))
    user_df = pd.DataFrame({"user_id": user_ids})

    new_df.to_csv(MATRIX_OUT, index=False)
    user_df.to_csv(USERS_OUT, index=False)
    print(f"Wrote {len(new_df):,} windows to {MATRIX_OUT.name}")
    print(f"Wrote {len(user_df):,} user_ids to {USERS_OUT.name}")

    target_dist = new_df.iloc[:, -1].astype(int).value_counts().sort_index()
    print("\nTarget distribution (10th tweet emotion):")
    print(target_dist.to_string())


if __name__ == "__main__":
    main()
