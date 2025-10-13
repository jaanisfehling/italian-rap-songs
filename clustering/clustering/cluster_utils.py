import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def one_hot_encode_feature(
    df: pd.DataFrame, feature_to_encode: str, weight: float = 1
) -> pd.DataFrame:
    dummies = pd.get_dummies(
        df[feature_to_encode], dtype="int32", prefix=feature_to_encode
    )
    dummies *= weight
    result_df = pd.concat([df, dummies], axis=1)
    return result_df.drop(columns=feature_to_encode)


def fix_weights(
    weights: Optional[Dict[str, float]], columns: List[str] | pd.Index
) -> Dict[str, float]:
    if weights is None or not isinstance(weights, dict):
        weights = {}
    for col in columns:
        if col not in weights or not isinstance(weights[col], (int, float)):
            weights[col] = 1.0
        if not 0.0 <= weights[col]:
            weights[col] = 0.0
    return weights


def cluster_overview(
    original_df: pd.DataFrame,
    cluster_labels: np.ndarray,
    num_cols: List[str],
    cat_cols: List[str],
    bool_cols: List[str],
    feature_importances: Dict[str, float],
) -> pd.DataFrame:
    def num_agg(series):
        res = f"~{series.mean():.1f}"
        if res == "~-0.0":
            return "~0.0"
        return res

    def cat_agg(series):
        vc = series.value_counts(normalize=True)
        if len(vc) >= 1:
            return f"{vc.index[0]} ({vc.iloc[0]:.0%})"
        else:
            return "NA (100%)"

    def bool_agg(series):
        vc = series.value_counts(normalize=True)
        if len(vc) == 1:
            return f"{vc.iloc[0]:.0%} {vc.index[0]}"
        elif len(vc) == 2:
            return f"{vc.iloc[0]:.0%} {vc.index[0]}/{vc.iloc[1]:.0%} {vc.index[1]}"
        else:
            return "100% NA"

    df = original_df.copy(deep=True)
    df["Cluster ID"] = cluster_labels
    df = pd.concat(
        [
            (
                df[num_cols + ["Cluster ID"]].groupby("Cluster ID").agg(num_agg)
                if num_cols
                else pd.DataFrame()
            ),
            (
                df[cat_cols + ["Cluster ID"]].groupby("Cluster ID").agg(cat_agg)
                if cat_cols
                else pd.DataFrame()
            ),
            (
                df[bool_cols + ["Cluster ID"]].groupby("Cluster ID").agg(bool_agg)
                if bool_cols
                else pd.DataFrame()
            ),
        ],
        axis=1,
    )

    # After grouping, Cluster ID becomes index, this will add it as a column instead
    df = df.reset_index()

    counts = np.bincount(cluster_labels)
    df["Members"] = df.apply(lambda row: counts[int(row["Cluster ID"])], axis=1)

    df["Cluster ID"] = df["Cluster ID"].apply(lambda x: f"Cluster {x}")

    # Order features by importance
    df = df[["Cluster ID", "Members"] + list(feature_importances.keys())]

    return df


def feature_importances(
    df: pd.DataFrame, labels: np.ndarray, efficient: bool, weights: Dict[str, float]
) -> Dict[str, float]:
    start = time.time()
    cols = [col for col, weight in weights.items() if weight > 0.0]
    df = df[cols]
    classifier = RandomForestClassifier(n_estimators=50 if efficient else 100)
    classifier.fit(df, labels)
    logging.info(
        "Feature importances execution time: %s s", round(time.time() - start, 3)
    )
    return dict(
        sorted(
            zip(cols, classifier.feature_importances_),
            key=lambda it: it[1],
            reverse=True,
        )
    )
