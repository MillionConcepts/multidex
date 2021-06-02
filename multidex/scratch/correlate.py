import os
import re

import django
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler

from fit import correlation_matrix
from marslab.imgops.imgutils import normalize_range
from marslab.compat.xcam import DERIVED_CAM_DICT

from marslab.imgops.pltutils import attach_axis


os.chdir("..")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()
from plotter.reduction import default_multidex_pipeline, \
    explained_variance_ratios

def make_orange_teal_cmap():
    teal = (98, 252, 232)
    orange = (255, 151, 41)
    half_len = 256
    vals = np.ones((half_len * 2, 4))
    vals[0:half_len, 0] = np.linspace(orange[0] / half_len, 0, half_len)
    vals[0:half_len, 1] = np.linspace(orange[1] / half_len, 0, half_len)
    vals[0:half_len, 2] = np.linspace(orange[2] / half_len, 0, half_len)
    vals[half_len:, 0] = np.linspace(0, teal[0] / half_len, half_len)
    vals[half_len:, 1] = np.linspace(0, teal[1] / half_len, half_len)
    vals[half_len:, 2] = np.linspace(0, teal[2] / half_len, half_len)
    return ListedColormap(vals, name="orange_teal")


register_cmap(cmap=make_orange_teal_cmap())


def s_from_midnight(instant):
    if instant is None:
        return None
    return instant.hour * 3600 + instant.minute * 60 + instant.second


def explode_binary(df, column, remove_nan=True):
    if remove_nan is True:
        unique_values = df[column].dropna().unique()
    else:
        unique_values = df[column].unique()
    exploded = pd.DataFrame(columns=unique_values, index=df.index).fillna(0)
    for value in unique_values:
        exploded.loc[df[column] == value, value] = 1
    return exploded


def reduce_and_correlate(
        pca_data, corr_data, pipeline = default_multidex_pipeline()
):
    vectors = pca_data.T.to_dict("list")
    vectarray = np.array(tuple(vectors.values()))
    transform = pipeline.fit_transform(vectarray)
    transform = pd.DataFrame(transform)
    transform.columns = [
        "P" + str(column + 1) for column in transform.columns
    ]
    explained_variance = np.round(
        explained_variance_ratios(transform) * 100, 2
    )
    corr_frame = pd.concat([corr_data, transform], axis=1)
    correlations = correlation_matrix(corr_frame)
    return transform, correlations, explained_variance


def plot_mdex_pca(
    corpus,
    explode_field=None,
    pca_fields="filters",
    corr_fields="filters",
    **plot_kwargs
):
    if (explode_field is not None) and (explode_field in corpus.columns):
        exploded = explode_binary(corpus, explode_field)
        search = pd.concat([corpus.copy(), exploded], axis=1)
    else:
        exploded = None
        search = corpus.copy()
    if exploded is not None:
        corr_fields += list(exploded.columns)
    pca_data, corr_data = preprocess_for_corrs(
        corr_fields, pca_fields, search
    )
    transform, correlations, explained_variance = reduce_and_correlate(pca_data, corr_data)

    title = ""
    return (
        correlations,
        transform,
        plot_dimensionality_matrices(
            correlations,
            transform,
            corr_fields,
            explained_variance,
            title,
            **plot_kwargs
        ),
    )


def mutually_filter_fields(
    pca_data, corr_data, pca_fields, corr_fields
):
    data_fields = set(pca_fields).union(corr_fields)
    corr_data = corr_data.dropna(
        subset=data_fields, axis=0
    ).reset_index(drop=True)
    pca_data = pca_data.dropna(
        subset=data_fields, axis=0
    )
    pca_data = pca_data[pca_fields]
    corr_data = corr_data[corr_fields]
    return pca_data, corr_data


def title_corrs(instrument, norm_values, r_star, scale_to, search_fields):
    if search_fields is None:
        search_fields = []
    title = "{}; {}; scaled to {} with R* {}; normalized {}".format(
        instrument,
        ", ".join([str(query) for query in search_fields]),
        str(scale_to),
        str(r_star),
        str(norm_values),
    )
    print(title)
    return title


def translate_fields_for_corr_graphs(filters, fields):
    if fields == "filters":
        fields = filters
    elif fields == "narrowband":
        fields = [
            filt for filt in filters if not re.match(r"[LR]0[RGB]", filt)
        ]
    return fields


def plot_dimensionality_matrices(
    correlations,
    transform,
    corr_fields,
    explained_variance=None,
    which="both",
    **plot_kwargs
):
    correlations = correlations.copy()

    if "fontsize" in plot_kwargs:
        plt.rcParams["font.size"] = plot_kwargs["fontsize"]
    if "corr_cmap" in plot_kwargs:
        corr_cmap = plot_kwargs.get("corr_cmap")
    else:
        corr_cmap = "Greys_r"
    if "norm" in plot_kwargs:
        norm = plot_kwargs["norm"]
    else:
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    if "absolute_correlations" in plot_kwargs:
        if plot_kwargs["absolute_correlations"] is True:
            correlations = correlations.abs()
    # if things were dropped b/c normalized away, don't die
    corr_fields = [f for f in corr_fields if f in correlations.columns]
    feature_quadrant = correlations.loc[corr_fields, corr_fields].copy()
    if transform is not None:
        param_quadrant = correlations.loc[
            transform.columns, corr_fields
        ].copy()
        param_quadrant = param_quadrant.rename(
            index={"incidence_angle": "\u03b8i"}
        )
        param_quadrant = param_quadrant.rename(
            columns={"incidence_angle": "\u03b8i"}
        )
    else:
        param_quadrant = None

    # todo: ugly
    feature_quadrant = feature_quadrant.rename(
        index={"incidence_angle": "\u03b8i"}
    )
    feature_quadrant = feature_quadrant.rename(
        columns={"incidence_angle": "\u03b8i"}
    )
    figs = {}
    for plot_type, quadrant in zip(
        ("features", "parameters"),
        (feature_quadrant, param_quadrant),
    ):
        if quadrant is None:
            continue
        if not ((which == "both") or (which == plot_type)):
            continue
        fig, ax = plt.subplots()
        corrchart = ax.imshow(quadrant, norm=norm, cmap=corr_cmap)
        ax.set_yticks(np.arange(len(quadrant.index)))
        ax.set_yticklabels([ix for ix in quadrant.index])
        ax.set_xticks(np.arange(len(quadrant.columns)))
        ax.set_xticklabels([ix for ix in quadrant.columns])
        # plt.title(plot_type + "\n" + title)
        cax = attach_axis(ax, "right", "8%")
        plt.colorbar(corrchart, cax=cax)
        if (explained_variance is not None) and (plot_type == "parameters"):
            for ix, ev in enumerate(explained_variance):
                ax.annotate(
                    str(ev),
                    (0, ix),
                    # xytext=(0, 0),
                    color="white",
                    # textcoords="offset points",
                )
        figs[plot_type] = fig
    return figs
