from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
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


def bic(
    X: np.ndarray, labels: np.ndarray, centroids: Optional[np.ndarray] = None
) -> float:
    """
    Calculate Bayesian Information Criterion (BIC) for clustering.

    Args:
        X: Data matrix of shape (n_samples, n_features)
        labels: Cluster labels for each data point
        centroids: Optional precomputed centroids. If None, will be computed.

    Returns:
        BIC score (higher is better, but typically negative)
    """
    n, d = X.shape
    k = len(np.unique(labels))

    # Input validation
    if d == 0:
        logging.error("Number of features is 0")
        return float("-inf")
    if n <= k:
        logging.error(
            f"Number of samples ({n}) must be greater than number of clusters ({k})"
        )
        return float("-inf")
    if k <= 0:
        logging.error(f"Number of clusters must be positive, got {k}")
        return float("-inf")

    # Compute centroids if not provided
    if centroids is None:
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    # Size of each cluster
    M = np.bincount(labels, minlength=k)

    # Remove empty clusters
    non_empty_clusters = M > 0
    k_actual = np.sum(non_empty_clusters)

    if k_actual == 0:
        return float("-inf")

    # Compute within-cluster sum of squares for variance estimation
    wcss = 0
    for i in range(k):
        if M[i] > 0:
            cluster_points = X[labels == i]
            wcss += np.sum((cluster_points - centroids[i]) ** 2)

    # Pooled within-cluster variance (spherical assumption)
    if n - k_actual <= 0:
        logging.error(f"Insufficient degrees of freedom: n-k = {n - k_actual}")
        return float("-inf")

    sigma_squared = wcss / (d * (n - k_actual))

    if sigma_squared <= 0:
        logging.error("Variance is non-positive")
        return float("-inf")

    # Compute log-likelihood
    log_likelihood = 0
    for i in range(k):
        if M[i] > 0:
            # Log probability for points in cluster i
            # Assuming spherical Gaussian with variance sigma_squared
            log_likelihood += M[i] * np.log(M[i] / n)  # mixing proportion
            log_likelihood -= (M[i] * d / 2) * np.log(
                2 * np.pi * sigma_squared
            )  # normalization

            # Distance term (already included in wcss, so we subtract it)
            cluster_points = X[labels == i]
            distances_squared = np.sum((cluster_points - centroids[i]) ** 2, axis=1)
            log_likelihood -= np.sum(distances_squared) / (2 * sigma_squared)

    # Number of free parameters:
    # - k centroids, each with d coordinates: k * d
    # - k-1 mixing proportions (sum to 1): k - 1
    # - 1 variance parameter: 1
    # Total: k * d + (k - 1) + 1 = k * (d + 1)
    num_params = k_actual * (d + 1)

    # BIC = -2 * log_likelihood + num_params * log(n)
    bic_score = -2 * log_likelihood + num_params * np.log(n)

    return (
        -bic_score
    )  # Return negative so higher is better (consistent with other metrics)


def bic_simplified(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Simplified BIC calculation based on within-cluster sum of squares.
    This version is more commonly used and computationally simpler.
    """
    n, d = X.shape
    k = len(np.unique(labels))

    if k <= 0 or n <= k or d == 0:
        return float("-inf")

    # Compute centroids
    centroids = np.array(
        [X[labels == i].mean(axis=0) for i in range(k) if np.sum(labels == i) > 0]
    )
    k_actual = len(centroids)

    # Within-cluster sum of squares
    wcss = 0
    for i in range(k):
        if np.sum(labels == i) > 0:
            cluster_points = X[labels == i]
            wcss += np.sum((cluster_points - centroids[i]) ** 2)

    # BIC = n * log(WCSS/n) + k * log(n) * d
    if wcss <= 0:
        return float("-inf")

    bic_score = n * np.log(wcss / n) + k_actual * np.log(n) * d

    return -bic_score  # Return negative so higher is better


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
        return bic_simplified(X, y_pred)

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
