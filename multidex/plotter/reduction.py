import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline


def transform_df(df, skl_pipeline):
    vectors = df.T.to_dict("list")
    vectarray = np.array(tuple(vectors.values()))
    transform = skl_pipeline.fit_transform(vectarray)
    transform = pd.DataFrame(transform)

    transform.columns = [
        "PC" + str(column + 1) for column in transform.columns
    ]
    return transform


def reduction_pipeline(norm, scale, reduce):
    return Pipeline(
        steps=[("norm", norm), ("scale", scale), ("reduce", reduce)]
    )


def default_multidex_pipeline():
    return reduction_pipeline(
        Normalizer(), StandardScaler(), PCA(n_components=8)
    )


def explained_variance_ratios(array):
    variances = array.var(axis=0)
    total = sum(variances)
    return variances / total


def transform_and_explain_variance(df, skl_pipeline):
    transform = transform_df(df, skl_pipeline)
    return transform, explained_variance_ratios(transform)
