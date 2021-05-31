import os
import re

import django
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import register_cmap
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

from fit import correlation_matrix
from marslab.imgops.imgutils import normalize_range
from marslab.compat.xcam import DERIVED_CAM_DICT

from marslab.imgops.pltutils import attach_axis, set_colorbar_font

os.chdir("..")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "multidex.settings")
os.environ["DJANGO_ALLOW_ASYNC_UNSAFE"] = "true"

django.setup()
import plotter.models
from plotter.spectrum_ops import filter_df_from_queryset
from multidex_utils import model_metadata_df


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


def reduce_and_correlate(pca_data, corr_data, method=PCA):
    vectors = pca_data.T.to_dict("list")
    vectarray = np.array(tuple(vectors.values()))
    decomposer = method(n_components=8)
    transform = decomposer.fit_transform(vectarray)
    transform = pd.DataFrame(transform)
    transform.columns = ["P" + str(column + 1) for column in transform.columns]
    if "explained_variance_ratio_" in dir(decomposer):
        explained_variance = np.round(
            decomposer.explained_variance_ratio_ * 100, 2
        )
    else:
        explained_variance = None
    corr_frame = pd.concat([corr_data, transform], axis=1)
    correlations = correlation_matrix(corr_frame)
    return transform, correlations, explained_variance


def plot_mdex_pca(
    scale_to=None,
    instrument="ZCAM",
    explode_field=None,
    pca_fields="filters",
    r_star=False,
    corr_fields="filters",
    norm_values=False,
    search_terms=None,
    method=PCA,
    **plot_kwargs
):
    spec_model = plotter.models.INSTRUMENT_MODEL_MAPPING[instrument]
    filter_info = DERIVED_CAM_DICT[instrument]["filters"]
    filters = list(filter_info.keys())
    metadata_df = model_metadata_df(spec_model)

    data_df = filter_df_from_queryset(
        spec_model.objects.all(), r_star=r_star, scale_to=scale_to
    )
    corpus = pd.concat([metadata_df, data_df], axis=1)
    corpus["ltst"] = corpus["ltst"].map(s_from_midnight)
    corpus["avg"] = corpus[filters].mean(axis=1)
    if 'zoom' in corpus.columns:
        corpus['zoom'] = corpus['zoom'].astype('float16')

    if (explode_field is not None) and (explode_field in corpus.columns):
        exploded = explode_binary(corpus, explode_field)
        search = pd.concat([corpus.copy(), exploded], axis=1)
    else:
        exploded = None
        search = corpus.copy()

    # fields to do pca on
    pca_fields = translate_fields_for_corr_graphs(filters, pca_fields)

    # fields to compare with the PCs
    corr_fields = translate_fields_for_corr_graphs(filters, corr_fields)
    if exploded is not None:
        corr_fields += list(exploded.columns)
    # corr_fields += [band + "_err" for band in narrowband]
    pca_data, corr_data = preprocess_for_corrs(
        corr_fields, norm_values, pca_fields, search, search_terms
    )
    transform, correlations, explained_variance = reduce_and_correlate(
        pca_data, corr_data, method
    )

    title = title_corrs(
        instrument, norm_values, r_star, scale_to, search_terms
    )
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


def preprocess_for_corrs(
    corr_fields, norm_values, pca_fields, search, search_fields
):
    if search_fields is None:
        search_fields = []
    # search_terms = [('feature', 'rock')]
    data_fields = set(pca_fields).union(corr_fields)
    for query in search_fields:
        search = search.loc[search[query[0]] == query[1]]
    search = search.dropna(subset=data_fields, axis=0).reset_index(drop=True)
    if norm_values:
        normed = []
        for _, row in search[pca_fields].iterrows():
            if isinstance(norm_values, str):
                normed.append(row.values / row[norm_values])
            else:
                normed.append(normalize_range(row.values))
        search.loc[:, pca_fields] = np.vstack(normed)
    pca_data = search[pca_fields]
    corr_data = search[corr_fields]
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
    title="",
    which = "both",
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
    param_quadrant = correlations.loc[transform.columns, corr_fields].copy()
    # todo: ugly
    feature_quadrant = feature_quadrant.rename(index={'incidence_angle': '\u03b8i'})
    feature_quadrant = feature_quadrant.rename(columns={'incidence_angle': '\u03b8i'})
    param_quadrant = param_quadrant.rename(index={'incidence_angle': '\u03b8i'})
    param_quadrant = param_quadrant.rename(columns={'incidence_angle': '\u03b8i'})
    figs = {}
    for plot_type, quadrant in zip(
        ("features", "parameters"),
        (feature_quadrant, param_quadrant),
    ):
        if not ((which == "both") or (which == plot_type)):
            continue
        fig, ax = plt.subplots()
        corrchart = ax.imshow(quadrant, norm=norm, cmap=corr_cmap)
        ax.set_yticks(np.arange(len(quadrant.index)))
        ax.set_yticklabels([ix for ix in quadrant.index])
        ax.set_xticks(np.arange(len(quadrant.columns)))
        ax.set_xticklabels([ix for ix in quadrant.columns])
        plt.title(plot_type + "\n" + title)
        cax = attach_axis(ax, "right", "10%")
        plt.colorbar(corrchart, cax=cax)
        if (explained_variance is not None) and (
            plot_type == "parameters"
        ):
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

# from itertools import product
# instrument="ZCAM"
# #instrument = "ZCAM"
# filter_info = DERIVED_CAM_DICT[instrument]["filters"]
# filters = list(filter_info.keys())
# narrowband = [filt for filt in filters if not re.match(r"[LR]0[RGB]", filt)]
#
# plt.rcParams["figure.figsize"] = (11, 11)
# corrs = []
# transforms = []
# figlist = []
# for r_star, scale_to, search_terms, norm_values in product(
#         [False, True],
#         [None, ("L1", "R1")],
#         [None],
#         #     [None, [("feature", "rock")]],
#         [False, 'R6'],
#         #     [PCA, FactorAnalysis]
#
# ):
#     #     if search_terms is not None:
#     #         explode_field = "morphology"
#     #     else:
#     #         explode_field = None
#     explode_field = None
#     method = PCA
#     correlations, transform, figs = plot_mdex_pca(
#         scale_to=scale_to,
#         instrument=instrument,
#         explode_field=explode_field,
#         r_star=r_star,
#         search_terms=search_terms,
#         corr_fields=narrowband + [
#             'incidence_angle'
#         ],
#         corr_cmap="orange_teal",
#         fontsize=18,
#         method=method,
#         norm_values=norm_values
#
#     )
#     corrs.append(correlations)
#     transforms.append(transform)
#     figlist.append(figs)