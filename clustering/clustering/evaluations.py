from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score


def cluster_accuracy(labels_true, labels_pred):
    # We need to map the labels to our cluster labels
    # This is a linear assignment problem on a bipartite graph
    k = max(len(np.unique(labels_true)), len(np.unique(labels_pred)))
    cost_matrix = np.zeros((k, k))
    for i in range(labels_true.shape[0]):
        cost_matrix[labels_true[i], labels_pred[i]] += 1
    inverted_cost_matrix = cost_matrix.max() - cost_matrix
    row_ind, col_ind = linear_sum_assignment(inverted_cost_matrix)
    return cost_matrix[row_ind, col_ind].sum() / labels_pred.size


def bic(x: np.ndarray, kmeans: KMeans) -> float:
    # number of clusters
    k = kmeans.n_clusters
    # size of the clusters
    M = np.bincount(kmeans.labels_)
    # size of data set
    n, d = x.shape
    if d == 0:
        logging.error("d = 0")
        return 0.0
    if n - k == 0:
        logging.error("n - k = 0")
        return 0.0
    # compute variance for all clusters beforehand
    cl_var = (1.0 / (d * (n - k))) * np.sum(
        [
            np.sum(
                distance.cdist(
                    x[np.nonzero(kmeans.labels_ == i)],
                    [kmeans.cluster_centers_[i]],
                    "euclidean",
                )
                ** 2
            )
            for i in range(k)
        ]
    )
    return np.sum(
        [
            M[i] * np.log(M[i])
            - M[i] * np.log(n)
            - ((M[i] * d) / 2) * np.log(2 * np.pi * cl_var)
            - ((M[i] - 1) * d / 2)
            for i in range(k)
        ]
    ) - (k / 2) * np.log(n)


def sum_of_squared_errors(X: np.ndarray, labels: np.ndarray) -> float:
    """Calculate Sum of Squared Errors for clustering."""
    sse = 0
    for cluster_id in np.unique(labels):
        cluster_points = X[labels == cluster_id]
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
            sse += np.sum((cluster_points - centroid) ** 2)
    return sse


class MetricEvaluator:
    """Handles evaluation of different clustering metrics."""

    def __init__(self):
        # Metrics that require true labels
        self.labeled_metrics = {
            "nmi": self._evaluate_nmi,
            "accuracy": self._evaluate_accuracy,
        }

        # Metrics that don't require true labels
        self.unlabeled_metrics = {
            "bic": self._evaluate_bic,
            "silhouette": self._evaluate_silhouette,
            "sse": self._evaluate_sse,
        }

    def _evaluate_nmi(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        return normalized_mutual_info_score(y_true, y_pred)

    def _evaluate_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        return cluster_accuracy(y_true, y_pred)

    def _evaluate_bic(
        self, y_true: Optional[np.ndarray], y_pred: np.ndarray, X: np.ndarray, **kwargs
    ) -> float:
        return 0.0

    def _evaluate_silhouette(
        self, y_true: Optional[np.ndarray], y_pred: np.ndarray, X: np.ndarray, **kwargs
    ) -> float:
        if len(np.unique(y_pred)) < 2:
            return -1.0  # Invalid clustering for silhouette score
        return silhouette_score(X, y_pred)

    def _evaluate_sse(
        self, y_true: Optional[np.ndarray], y_pred: np.ndarray, X: np.ndarray, **kwargs
    ) -> float:
        return sum_of_squared_errors(X, y_pred)

    def evaluate_metrics(
        self,
        metrics: List[str],
        y_true: Optional[np.ndarray],
        y_pred: np.ndarray,
        X: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate specified metrics and return results."""
        results = {}

        # Evaluate labeled metrics only if y_true is available
        if y_true is not None:
            for metric in metrics:
                if metric in self.labeled_metrics:
                    results[metric] = self.labeled_metrics[metric](y_true, y_pred, X)
        else:
            # Set labeled metrics to None if no true labels
            for metric in metrics:
                if metric in self.labeled_metrics:
                    results[metric] = None

        # Evaluate unlabeled metrics
        for metric in metrics:
            if metric in self.unlabeled_metrics:
                results[metric] = self.unlabeled_metrics[metric](y_true, y_pred, X)

        return results

    def get_available_metrics(self) -> List[str]:
        """Return list of all available metrics."""
        return list(self.labeled_metrics.keys()) + list(self.unlabeled_metrics.keys())
