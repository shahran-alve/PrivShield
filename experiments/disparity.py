"""Who absorbs the threshold's refusals (Section IX).

For each value of race and of age band, the script asks the outcome breakdown
of that group within every (partition, gender) combination, using the first
six partition values. Patient-k refuses a breakdown when it is empty or when
its smallest outcome cell has fewer than k people. The output is the share of
these questions refused per group, next to the group's share of the data.

Output: results/disparity.csv

Run:  python experiments/disparity.py
"""
import numpy as np
import pandas as pd

from common import OUT, load_adult, load_nhanes, result_path

K = 10


def refused(data, filters):
    mask = np.ones(len(data), dtype=bool)
    for col, val in filters.items():
        mask &= (data[col].astype(str) == str(val)).values
    counts = data[mask][OUT].value_counts()
    return len(counts) == 0 or int(counts.min()) < K


def run(name, data):
    rows = []
    for attr in ("race", "age_band"):
        for val in sorted(data[attr].dropna().unique()):
            questions = [{attr: str(val), "partition": str(p), "gender": str(g)}
                         for p in sorted(data["partition"].unique())[:6]
                         for g in sorted(data["gender"].unique())]
            n_refused = sum(refused(data, f) for f in questions)
            rows.append(dict(dataset=name, attribute=attr, value=str(val),
                             share=round(float((data[attr] == val).mean()), 4),
                             questions=len(questions), refused=n_refused,
                             refused_rate=round(n_refused / len(questions), 3)))
    return rows


if __name__ == "__main__":
    rows = run("Adult", load_adult()) + run("NHANES", load_nhanes())
    df = pd.DataFrame(rows)
    df.to_csv(result_path("disparity.csv"), index=False)
    print(df.to_string(index=False))
