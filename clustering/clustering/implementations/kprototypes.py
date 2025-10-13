from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans


def kprototypes(df, k, num_cols, cat_cols, init):
    # KPrototypes does not accept a dataset without numeric columns
    if not num_cols:
        c = KModes(
            n_clusters=k,
            init=init,
            n_jobs=-1,
        ).fit(df.values)
    elif not cat_cols:
        c = KMeans(n_clusters=k).fit(df.values)
    else:
        cat_cols_indices = [df.columns.get_loc(col) for col in cat_cols]
        num_cols_indices = [df.columns.get_loc(col) for col in num_cols]
        try:
            c = KPrototypes(init=init, n_clusters=k, n_jobs=-1).fit(df.values, categorical=cat_cols_indices)
        # The initilization can go wrong for big k values, but we can initialize randomly ourselves
        # The init="random" argument will not work in this case
        except ValueError:
            init_centroids = [
                df.sample(frac=1).values[:, num_cols_indices][:k],
                df.sample(frac=1).values[:, cat_cols_indices][:k],
            ]
            c = KPrototypes(
                init=init_centroids,
                n_clusters=k,
                n_jobs=-1,
            ).fit(df.values, categorical=cat_cols_indices)
    return c.labels_
