import re
from collections.abc import Callable, Sequence
from typing import Optional, Union

import matplotlib.colors as mcolor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types
from marslab.imgops.pltutils import attach_axis
from matplotlib.cm import register_cmap

from multidex.plotter.reduction import (
    default_multidex_pipeline,
    explained_variance_ratios,
)


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
    return mcolor.ListedColormap(vals, name="orange_teal")


register_cmap(cmap=make_orange_teal_cmap())


def s_from_midnight(instant):
    if instant is None:
        return None
    return instant.hour * 3600 + instant.minute * 60 + instant.second


def explode_sequential(df, column, remove_nan=True):
    if remove_nan is True:
        unique_values = df[column].dropna().unique()
    else:
        unique_values = df[column].unique()
    exploded = pd.Series(index=df.index, dtype="uint8")
    for ix, value in enumerate(unique_values):
        exploded.loc[df[column] == value] = ix
    return exploded


def explode_binary(df, column, remove_nan=True):
    if remove_nan is True:
        unique_values = df[column].dropna().unique()
    else:
        unique_values = df[column].unique()
    exploded = pd.DataFrame(columns=unique_values, index=df.index).fillna(0)
    for value in unique_values:
        exploded.loc[df[column] == value, value] = 1
    return exploded


def numeric_columns(data: pd.DataFrame) -> list[str]:
    return [
        col
        for col in data.columns
        if pandas.api.types.is_numeric_dtype(data[col])
    ]


def reduce_and_correlate(
    pca_data, corr_data, pipeline=default_multidex_pipeline()
):
    vectors = pca_data.T.to_dict("list")
    vectarray = np.array(tuple(vectors.values()))
    transform = pipeline.fit_transform(vectarray)
    transform = pd.DataFrame(transform)
    transform.columns = ["P" + str(column + 1) for column in transform.columns]
    explained_variance = np.round(
        explained_variance_ratios(transform) * 100, 2
    )
    corr_frame = pd.concat([corr_data, transform], axis=1)
    correlations = correlation_matrix(corr_frame)
    return transform, correlations, explained_variance


def correlation_matrix(
    data: Optional[pd.DataFrame] = None,
    matrix_columns: Optional[Sequence[Union[str, Sequence]]] = None,
    column_names: Optional[list[str]] = None,
    correlation_type: Union[str, Callable] = "coefficient",
    precision: Optional[int] = 2,
) -> pd.DataFrame:
    """
    return a correlation or covariance matrix from a dataframe and/or
    a sequence of ndarrays, lists, whatever

    matrix_columns adds additional columns, or restricts the columns --
    should be str (name of column in data) or a sequence. if not
    passed, just every numeric column in data is used.

    column names explicitly names the columns (unnecessary)

    correlation_type can be "coefficient" or "covariance" or
    a callable of your choice

    precision rounds the matrix (for readability) to the passed number
    of digits. pass None to not round.
    """
    if (matrix_columns is None) and (data is None):
        raise ValueError("This function has been invoked with no data at all.")

    if matrix_columns is None:
        matrix_columns = [col for col in numeric_columns(data)]

    # option 1: automatically name the columns
    if column_names is None:
        column_names = [
            col if isinstance(col, str) else "Unnamed: " + str(ix)
            for ix, col in enumerate(matrix_columns)
        ]
    # option 2: explicitly name each column
    elif len(column_names) != len(matrix_columns):
        raise IndexError(
            "Length of column names must match length of columns."
        )

    if correlation_type == "coefficient":
        corr_func = np.corrcoef
    elif correlation_type == "covariance":
        corr_func = np.cov
    elif callable(correlation_type):
        corr_func = correlation_type
    else:
        raise ValueError(
            "Only coefficient (of correlation) and covariance "
            + "are currently supported (or else explicitly "
            + "pass a callable correlation function)."
        )
    matrix = []
    for column in matrix_columns:
        if isinstance(column, str):
            matrix.append(data[column])
        else:
            matrix.append(column)
    matrix = pd.concat(matrix, axis=1).T
    output_matrix = pd.DataFrame(
        corr_func(matrix), columns=column_names, index=column_names
    )
    if precision is not None:
        output_matrix = output_matrix.round(precision)
    # drop garbage:
    for col in output_matrix.columns:
        if all(output_matrix[col].isna()):
            output_matrix.drop(col, axis=1, inplace=True)
            output_matrix.drop(col, axis=0, inplace=True)
    return output_matrix


def mutually_filter_fields(pca_data, corr_data, pca_fields, corr_fields):
    data_fields = set(pca_fields).union(corr_fields)
    corr_data = corr_data.dropna(subset=data_fields, axis=0).reset_index(
        drop=True
    )
    pca_data = pca_data.dropna(subset=data_fields, axis=0)
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


DEFAULT_CORR_NORM = mcolor.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)


def make_corr_chart(
    matrix, explained_variance=None, norm=DEFAULT_CORR_NORM, cmap="orange_teal"
):
    fig, ax = plt.subplots()
    corrchart = ax.imshow(matrix, norm=norm, cmap=cmap)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([ix for ix in matrix.index])
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([ix for ix in matrix.columns])
    # plt.title(plot_type + "\n" + title)
    cax = attach_axis(ax, "right", "8%")
    plt.colorbar(corrchart, cax=cax)
    if explained_variance is not None:
        for ix, ev in enumerate(explained_variance):
            ax.annotate(
                str(ev),
                (0, ix),
                # xytext=(0, 0),
                color="white",
                # textcoords="offset points",
            )
    return fig
