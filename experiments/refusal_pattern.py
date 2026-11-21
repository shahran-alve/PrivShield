"""Does the pattern of threshold refusals predict a target's outcome? (Section VI)

For each at-risk target the attacker observes which of the following queries
the threshold answers: for every outcome y, the count of the target's
(age_band, race, gender) group with outcome y, and the count of the target's
own cell with outcome y. That gives one allowed/refused bit per query.

The attacker is evaluated with 5-fold cross-validation. It learns the majority
outcome for each pattern on the training folds and falls back to the overall
majority outcome for a pattern it has not seen. Advantage is its held-out
accuracy minus the accuracy of always guessing the majority outcome, floored
at zero because the attacker can always ignore the pattern.

The patterns come from the threshold alone. Refusals issued by the rank
auditor are not part of this test.

Output: results/refusal_pattern.csv

Run:  python experiments/refusal_pattern.py
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from common import load_adult, load_hospital, load_nhanes, prep, result_path, target_of

K = 10
FOLDS = 5


def evaluate(name, data, seed=0):
    loc, parts, outs, pool = prep(data, K)
    rows = []
    for _, t in pool.iterrows():
        bk, tp, ty = target_of(t)
        cell = loc.get(bk, {})
        pattern = []
        for y in outs:
            pattern.append(int(sum(cell.get((p, y), 0) for p in parts) >= K))
            pattern.append(int(cell.get((tp, y), 0) >= K))
        rows.append((tuple(pattern), ty))
    order = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(order)
    fold = order % FOLDS
    prior_acc, advantage = [], []
    for f in range(FOLDS):
        train = [rows[i] for i in range(len(rows)) if fold[i] != f]
        test = [rows[i] for i in range(len(rows)) if fold[i] == f]
        prior = pd.Series([y for _, y in train]).value_counts().idxmax()
        by_pattern = defaultdict(list)
        for s, y in train:
            by_pattern[s].append(y)
        majority = {s: pd.Series(ys).value_counts().idxmax() for s, ys in by_pattern.items()}
        p_acc = float(np.mean([y == prior for _, y in test]))
        a_acc = float(np.mean([majority.get(s, prior) == y for s, y in test]))
        prior_acc.append(p_acc)
        advantage.append(max(0.0, a_acc - p_acc))
    return dict(dataset=name, targets=len(rows), folds=FOLDS,
                distinct_patterns=len({s for s, _ in rows}),
                prior_accuracy=round(float(np.mean(prior_acc)), 3),
                advantage_mean=round(float(np.mean(advantage)), 4),
                advantage_max=round(float(np.max(advantage)), 4))


if __name__ == "__main__":
    df = pd.DataFrame([evaluate(n, l()) for n, l in
                       [("Hospital", load_hospital), ("Adult", load_adult), ("NHANES", load_nhanes)]])
    df.to_csv(result_path("refusal_pattern.csv"), index=False)
    print(df.to_string(index=False))
