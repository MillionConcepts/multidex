import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline


def transform_df(df, skl_pipeline):
    fit_pipeline = skl_pipeline.fit(df)
    transform = pd.DataFrame(fit_pipeline.transform(df))
    if "reduce" in skl_pipeline.named_steps.keys():
        transform.columns = [
            "PC" + str(column + 1) for column in transform.columns
        ]
        eigenvectors = fit_pipeline.named_steps['reduce'].components_
    else:
        transform.columns = df.columns
        eigenvectors = None
    return transform, eigenvectors


def reduction_pipeline(norm, scale, reduce):
    return Pipeline(
        steps=[("norm", norm), ("scale", scale), ("reduce", reduce)]
    )


def default_multidex_pipeline():
    return reduction_pipeline(
        Normalizer(), StandardScaler(), PCA(n_components=6)
    )


def explained_variance_ratios(array):
    variances = array.var(axis=0)
    total = sum(variances)
    return variances / total


def transform_and_explain(df, skl_pipeline):
    transform, eigenvectors = transform_df(df, skl_pipeline)
    ratios = explained_variance_ratios(transform)
    ratios.name = "explained_variance"
    return transform, eigenvectors, ratios


def fit_line(xvar: np.ndarray, yvar: np.ndarray):
    """simple least-squares linear regression"""
    regression = linregress(xvar, yvar)
    intercept = regression.intercept
    slope = regression.slope
    scaled = intercept + slope * xvar
    return regression, scaled


