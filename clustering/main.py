import argparse
import json
import os
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder

from clustering.benchmark_utils import print_best
from clustering.evaluations import MetricEvaluator
from clustering.mixed_type import methods as mixed_type_methods
from clustering.preprocess import *


def run(methods: List[Callable], metrics: List[str], iterations: int):
    with open("data/tabular/info.json") as f:
        infos = json.loads(f.read())

    evaluator = MetricEvaluator()

    # Initialize results storage
    results = {metric: {} for metric in metrics}
    best_results = {
        metric: {
            filename.replace(".csv", ""): (
                "None",
                float("-inf") if metric != "sse" else float("inf"),
            )
            for filename in infos.keys()
        }
        for metric in metrics
    }

    for method in methods:
        method_name = method.__name__
        print(f"Running {method_name}")

        # Initialize method results
        for metric in metrics:
            results[metric][method_name] = {}

        for filename, info in infos.items():
            print(f"Loading {filename}")
            df = pd.read_csv(
                os.path.join("data/tabular/", filename),
                sep=None,
                engine="python",
            )

            # Check if target exists and is not null
            has_target = "target" in info and info["target"] is not None
            y_true = None

            if has_target:
                y_true = LabelEncoder().fit_transform(df[info["target"]])
                df = df.drop(columns=info["drop"] + [info["target"]])
            else:
                df = df.drop(columns=info["drop"])
                print(
                    f"No target specified for {filename}, using unsupervised metrics only..."
                )

            df = clean(df)
            num_cols, cat_cols = classify_cols(df)
            df = imputate_na(df, num_cols, cat_cols)
            df = scale_cols(df, num_cols, cat_cols)

            # Store metric results for iterations
            metric_results = {metric: [] for metric in metrics}

            for i in range(iterations):
                print(f"Test iteration {i+1}/{iterations}", end="\r", flush=True)
                df_copied = df.copy(deep=True)

                n_clusters = len(np.unique(y_true)) if has_target else None
                if n_clusters is None:
                    # For datasets without labels, we need to estimate number of clusters
                    # This is a simple heuristic - you might want to make this configurable
                    n_clusters = min(10, max(2, len(df) // 50))

                y_pred = method(df_copied, n_clusters, num_cols, cat_cols)

                # Convert DataFrame to numpy array for metrics that need it
                X = df_copied.values

                # Evaluate all requested metrics
                iteration_results = evaluator.evaluate_metrics(
                    metrics, y_true, y_pred, X
                )

                for metric, value in iteration_results.items():
                    metric_results[metric].append(value)

            # Process results for this dataset
            df_name = filename.replace(".csv", "")

            for metric in metrics:
                values = [v for v in metric_results[metric] if v is not None]

                if values:  # If we have valid values
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    results[metric][method_name][
                        df_name
                    ] = f"{mean_val:.3f} ± {std_val:.2f}"

                    # Update best results (lower is better for SSE, higher for others)
                    is_better = (
                        mean_val < best_results[metric][df_name][1]
                        if metric == "sse"
                        else mean_val > best_results[metric][df_name][1]
                    )

                    if is_better:
                        best_results[metric][df_name] = (method_name, mean_val)
                else:
                    results[metric][method_name][df_name] = "N/A"

    # Print results
    print("\n\n***RESULTS***")

    for metric in metrics:
        if any(results[metric].values()):  # Only show if we have results
            print(f"\n{metric.upper()}:")
            df_results = pd.DataFrame(
                results[metric].values(), index=results[metric].keys()
            )
            with pd.option_context("display.max_rows", None):
                print(df_results)
            print("")
            print_best(best_results[metric])


parser = argparse.ArgumentParser(
    description="Run clustering benchmarks of different categories.",
)
parser.add_argument("category")
parser.add_argument(
    "-m",
    "--metrics",
    default=["nmi", "sse"],
    choices=["nmi", "accuracy", "bic", "silhouette", "sse"],
    nargs="*",
)
parser.add_argument(
    "-i",
    "--iterations",
    type=int,
    default=10,
)
args = parser.parse_args()

match args.category:
    case "mixed_type":
        run(mixed_type_methods, args.metrics, args.iterations)
    case other:
        print(f"Invalid benchmark: {other}")
        exit(1)
