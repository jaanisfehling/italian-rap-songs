import math
from typing import Callable, List

import gower
import kmedoids
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from stepmix import StepMix
from torch.utils.data import DataLoader

from .cluster_utils import one_hot_encode_feature
from .implementations.all_to_all_autoencoder import AllToAllAutoencoder
from .implementations.cat_to_cat_autoencoder import CatToCatAutoencoder
from .implementations.kamila import KAMILA
from .implementations.kprototypes import kprototypes
from .implementations.torch_utils import PytorchMixedTypeDataset, build_autoencoder


def random(df, k, num_cols, cat_cols):
    return np.random.randint(0, k, size=len(df))


def kmeans_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return KMeans(n_clusters=k, init="random").fit(df.values).labels_


def kmeans_no_cat(df, k, num_cols, cat_cols):
    n = len(df)
    df = df.drop(columns=cat_cols)
    if df.empty:
        return np.zeros(n)
    return KMeans(n_clusters=k, init="random").fit(df.values).labels_


def kmeans_pp_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return KMeans(n_clusters=k, init="k-means++").fit(df.values).labels_


def kmeans_pp_no_cat(df, k, num_cols, cat_cols):
    n = len(df)
    df = df.drop(columns=cat_cols)
    if df.empty:
        return np.zeros(n)
    return KMeans(n_clusters=k, init="k-means++").fit(df.values).labels_


def kmedoids_build_alternate_gower(df, k, num_cols, cat_cols):
    dm = gower.gower_matrix(
        df.astype("float64"), cat_features=[c in cat_cols for c in df.columns]
    )
    return (
        KMedoids(n_clusters=k, metric="precomputed", method="alternate", init="build")
        .fit(dm)
        .labels_
    )


def kmedoids_build_pam_gower(df, k, num_cols, cat_cols):
    dm = gower.gower_matrix(
        df.astype("float64"), cat_features=[c in cat_cols for c in df.columns]
    )
    return (
        KMedoids(n_clusters=k, metric="precomputed", method="pam", init="build")
        .fit(dm)
        .labels_
    )


def kmedoids_pp_alternate_gower(df, k, num_cols, cat_cols):
    dm = gower.gower_matrix(
        df.astype("float64"), cat_features=[c in cat_cols for c in df.columns]
    )
    return (
        KMedoids(
            n_clusters=k, metric="precomputed", method="alternate", init="k-medoids++"
        )
        .fit(dm)
        .labels_
    )


def kmedoids_pp_pam_gower(df, k, num_cols, cat_cols):
    dm = gower.gower_matrix(
        df.astype("float64"), cat_features=[c in cat_cols for c in df.columns]
    )
    return (
        KMedoids(n_clusters=k, metric="precomputed", method="pam", init="k-medoids++")
        .fit(dm)
        .labels_
    )


def kmedoids_build_fasterpam_gower(df, k, num_cols, cat_cols):
    dm = gower.gower_matrix(
        df.astype("float64"), cat_features=[c in cat_cols for c in df.columns]
    )
    return (
        kmedoids.KMedoids(
            n_clusters=k, metric="precomputed", method="fasterpam", init="build"
        )
        .fit(dm)
        .labels_
    )


def agglomerative_avglink_gower(df, k, num_cols, cat_cols):
    dm = gower.gower_matrix(
        df.astype("float64"), cat_features=[c in cat_cols for c in df.columns]
    )
    return (
        AgglomerativeClustering(
            n_clusters=k, memory=".cache/", metric="precomputed", linkage="average"
        )
        .fit(dm)
        .labels_
    )


def kprototypes_huang(df, k, num_cols, cat_cols):
    return kprototypes(df, k, num_cols, cat_cols, "Huang")


def kprototypes_cao(df, k, num_cols, cat_cols):
    return kprototypes(df, k, num_cols, cat_cols, "Cao")


def kmeans_pp_ae_all(df, k, num_cols, cat_cols):
    embedding_sizes = [
        (df[col].nunique(), min(50, math.ceil(df[col].nunique() / 2)))
        for col in df[cat_cols]
    ]
    input_dim = sum(d for _, d in embedding_sizes) + len(num_cols)
    encoder, decoder = build_autoencoder(
        input_dim, input_dim, max(1, round(math.log2(input_dim)))
    )

    ae = AllToAllAutoencoder(encoder, decoder, embedding_sizes, device="cpu")
    dataset = PytorchMixedTypeDataset(df, cat_cols, num_cols)
    dataloader = DataLoader(dataset, batch_size=32)
    ae.fit(dataloader, n_epochs=100, lr=0.001)

    num = torch.tensor(df[num_cols].values, dtype=torch.float).detach().to("cpu")
    cat = torch.tensor(df[cat_cols].values, dtype=torch.int).detach().to("cpu")
    features = ae.encode(cat, num).detach().cpu().numpy()
    return KMeans(n_clusters=k).fit(features).labels_


def kmeans_pp_ae_cat(df, k, num_cols, cat_cols):
    embedding_sizes = [
        (df[col].nunique(), min(50, math.ceil(df[col].nunique() / 2)))
        for col in df[cat_cols]
    ]
    input_dim = sum(d for _, d in embedding_sizes)
    encoder, decoder = build_autoencoder(
        input_dim, input_dim, max(1, round(math.log2(input_dim)))
    )

    ae = CatToCatAutoencoder(encoder, decoder, embedding_sizes, device="cpu")
    dataset = PytorchMixedTypeDataset(df, cat_cols, num_cols)
    dataloader = DataLoader(dataset, batch_size=32)
    ae.fit(dataloader, n_epochs=100, lr=0.001)

    num = torch.tensor(df[num_cols].values, dtype=torch.float).detach().to("cpu")
    cat = torch.tensor(df[cat_cols].values, dtype=torch.int).detach().to("cpu")
    scaled_cat = StandardScaler().fit_transform(
        ae.encode(cat, num).detach().cpu().numpy()
    )
    features = np.concatenate((scaled_cat, num), axis=1)
    return KMeans(n_clusters=k).fit(features).labels_


def gaussian_mixture_full_kmeans_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return GaussianMixture(
        n_components=k, covariance_type="full", init_params="kmeans"
    ).fit_predict(df)


def gaussian_mixture_full_kmeans_pp_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return GaussianMixture(
        n_components=k, covariance_type="full", init_params="k-means++"
    ).fit_predict(df)


def gaussian_mixture_tied_kmeans_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return GaussianMixture(
        n_components=k, covariance_type="tied", init_params="kmeans"
    ).fit_predict(df)


def gaussian_mixture_diag_kmeans_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return GaussianMixture(
        n_components=k, covariance_type="diag", init_params="kmeans"
    ).fit_predict(df)


def gaussian_mixture_spherical_kmeans_onehot_cat(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return GaussianMixture(
        n_components=k, covariance_type="spherical", init_params="kmeans"
    ).fit_predict(df)


def step_mix_kmeans_gaussian_diag_multinoulli(df, k, num_cols, cat_cols):
    descriptor = {}
    if len(num_cols) > 0:
        descriptor["num_model"] = {
            "model": "gaussian_diag",
            "n_columns": len(num_cols),
        }
    if len(cat_cols) > 0:
        descriptor["cat_model"] = {
            "model": "multinoulli",
            "n_columns": len(cat_cols),
        }
    df = df[num_cols + cat_cols]  # reorder columns
    return (
        StepMix(
            n_components=k, measurement=descriptor, progress_bar=0, init_params="kmeans"
        )
        .fit(df)
        .predict(df)
    )


def step_mix_kmeans_gaussian_unit_multinoulli(df, k, num_cols, cat_cols):
    descriptor = {}
    if len(num_cols) > 0:
        descriptor["num_model"] = {
            "model": "gaussian_unit",
            "n_columns": len(num_cols),
        }
    if len(cat_cols) > 0:
        descriptor["cat_model"] = {
            "model": "multinoulli",
            "n_columns": len(cat_cols),
        }
    df = df[num_cols + cat_cols]  # reorder columns
    return (
        StepMix(
            n_components=k, measurement=descriptor, progress_bar=0, init_params="kmeans"
        )
        .fit(df)
        .predict(df)
    )


def step_mix_kmeans_gaussian_spherical_multinoulli(df, k, num_cols, cat_cols):
    descriptor = {}
    if len(num_cols) > 0:
        descriptor["num_model"] = {
            "model": "gaussian_spherical",
            "n_columns": len(num_cols),
        }
    if len(cat_cols) > 0:
        descriptor["cat_model"] = {
            "model": "multinoulli",
            "n_columns": len(cat_cols),
        }
    df = df[num_cols + cat_cols]  # reorder columns
    return (
        StepMix(
            n_components=k, measurement=descriptor, progress_bar=0, init_params="kmeans"
        )
        .fit(df)
        .predict(df)
    )


def kamila(df, k, num_cols, cat_cols):
    return KAMILA(n_clusters=k).fit_predict(df[num_cols].values, df[cat_cols].values)


def kamila_one_hot(df, k, num_cols, cat_cols):
    for col in cat_cols:
        df = one_hot_encode_feature(df, col)
    return KAMILA(n_clusters=k).fit_predict(
        df[num_cols].values, df.drop(columns=num_cols).values
    )


methods: List[Callable[[pd.DataFrame, int, List[str], List[str]], np.ndarray]] = [
    # random,
    # kmeans_onehot_cat,
    # kmeans_no_cat,
    kmeans_pp_onehot_cat,
    # kmeans_pp_no_cat,
    # kmedoids_build_alternate_gower,
    # kmedoids_build_pam_gower,
    # kmedoids_pp_alternate_gower,
    # kmedoids_pp_pam_gower,
    # kmedoids_build_fasterpam_gower,
    # agglomerative_avglink_gower,
    # kprototypes_huang,
    # kprototypes_cao,
    # kmeans_pp_ae_all,
    # kmeans_pp_ae_cat,
    # gaussian_mixture_full_kmeans_onehot_cat,
    # gaussian_mixture_full_kmeans_pp_onehot_cat,
    # gaussian_mixture_tied_kmeans_onehot_cat,
    # gaussian_mixture_diag_kmeans_onehot_cat,
    # gaussian_mixture_spherical_kmeans_onehot_cat,
    # step_mix_kmeans_gaussian_diag_multinoulli,
    # step_mix_kmeans_gaussian_unit_multinoulli,
    # step_mix_kmeans_gaussian_spherical_multinoulli,
    # kamila,
    # kamila_one_hot,
]
