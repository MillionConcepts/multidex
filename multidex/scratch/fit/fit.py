"""
functions for fitting and modeling
"""
from functools import partial, wraps
from inspect import signature, Parameter
from typing import Callable, Union, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.api.types
from scipy import integrate, interpolate
from scipy.optimize import curve_fit
from scipy.stats import linregress


def coef_det(
        fit_curve: np.ndarray,
        dependent: np.ndarray
) -> float:
    """coefficient of determination"""
    # sample variance
    ss_t = sum((dependent - dependent.mean()) ** 2)
    # sum of squares of residuals
    ss_r = sum((dependent - fit_curve) ** 2)
    det = 1 - ss_r / ss_t
    return det


def all_equal(iterable):
    """
    are all elements of the iterable equal? (assuming python knows
    a valid equivalence relation over them)
    """
    iterator = iter(iterable)
    initial = next(iterator)
    for item in iterator:
        if not item == initial:
            return False
    return True

# visualization


def fit_line(xvar: np.ndarray, yvar: np.ndarray, do_plot=True):
    """dumb least-squares linear regression"""
    regression = linregress(xvar, yvar)
    intercept = regression.intercept
    slope = regression.slope
    if do_plot:
        plt.scatter(xvar, yvar)
        plt.plot(xvar, intercept + slope * xvar, label="fitted line")
    return regression


def plot_mesh(xvar: np.ndarray, yvar: np.ndarray, zvar: np.ndarray):
    """attempt to interpolate zvar over convex hull of xvar, yvar"""
    xmesh, ymesh = np.meshgrid(xvar, yvar)
    zmesh = interpolate.griddata(
        (xvar, yvar),
        zvar,
        (xmesh, ymesh),
        method="linear",
        fill_value=np.nan,
        rescale=False,
    )
    plt.figure()
    scan_cont = plt.contourf(xvar, yvar, zmesh)
    plt.colorbar(scan_cont)


# maths


# being dumb and thinking about a symmetrical gaussian beam


def delta_circle(theta, a, b, r):
    """
    distance to point on circle, where a = x displacement from circle center
    to origin, b = y displacement from circle center to origin, r = circle
    radius
    """
    return np.sqrt((a + r * np.cos(theta)) ** 2 + (b + r * np.sin(theta)) ** 2)


def gaussian(delta, w):
    """
    delta is distance from point to peak of gaussian (peak of gaussian is
    origin) w is FWHM of gaussian
    """
    sigma = w / 2.355
    return (
        1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-1 * (delta ** 2) / (2 * sigma ** 2))
    )


def n_gauss(delta, w):
    """
    gaussian normalized to peak height 0
    """
    return gaussian(delta, w) / gaussian(0, w)


def delta_gaussian(theta, w, a, b, r):
    """
    value of gaussian at point on circle displaced by a,b from origin
    """
    return n_gauss(delta_circle(theta, a, b, r), w)


def response_on_circle(r, w, a, b):
    """inner integral: gaussian response over exterior of circle"""
    # noinspection PyTypeChecker
    return integrate.quad(partial(delta_gaussian, w=w, a=a, b=b, r=r), 0, 2 * np.pi)[0]


def gaussian_circle(radius, w, d):
    """
    integral of gaussian over entire distant circle,
    treating it as a series of annular sections --
    should be something like power?

    n.b.: i think this approach is not wrong but it might
    be irrelevant. consider approach from "Power Transmittance
    of a Laterally Shifted Gaussian Beam through a Circular Aperture"
    (roughly: 𝑒𝑥𝑝(−2𝑑2𝑤𝑜𝑢𝑡2)∑2𝑘𝑑2𝑘𝑤𝑜𝑢𝑡2𝑘(𝑘!)2∞𝑘=0γ(𝑘+1,2𝑎2𝑤))
    https://arxiv.org/abs/1605.04241
    """
    # this actually depends only on distance
    # but i'm not sure how to solve the inmost integral.
    # for ease of use:
    a = np.sqrt(d ** 2 / 2)
    b = a
    # noinspection PyTypeChecker
    return integrate.quad(partial(response_on_circle, w=w, a=a, b=b), 0, radius)


def numeric_columns(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if pandas.api.types.is_numeric_dtype(data[col])]


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
        raise IndexError("Length of column names must match length of columns.")

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


# ##############   function - fitting class ################# <====>


class Fit:
    def __init__(
        self,
        underlying_function: Callable,
        dimensionality: int,
        data: Optional[pd.DataFrame] = None,
        guess: Optional[Sequence[float]] = None,
        vector: Optional[Sequence[Sequence[float]]] = None,
        dependent_variable: Union[Sequence[float], str, None] = None,
    ):
        """
        flexible interface to scipy.optimize.curve_fit

        The first dimensionality parameters of underlying_function
        are treated as components of the independent variable; the others
        as parameters to be fit. we could be more flexible about names,
        etc., but we're being strict here in order to be performant --
        the number of times the function is called means that in some cases
        keyword assignment will actually incur meaningful performance
        overhead -- and because a certain amount of excessively configurable
        sloppiness could arise. I might change my mind on this later.

        Here's the basic expected sequence:

        1. Initialize the object with at least the function
        2. Make the vector with make_vector
        3. fit with fit. It's ok to pass a dependent variable and a guess
            at this function call, or to set them explicitly beforehand.
        """
        sig = signature(underlying_function)
        assert dimensionality < len(sig.parameters), (
            "The underlying function must "
            + "have at least one 'free' "
            + "parameter to be a meaningful candidate"
            + " for curve-fitting."
        )
        self.underlying_function = underlying_function
        self.dimensionality = dimensionality
        self.independent_variables = [
            item
            for ix, item in enumerate(sig.parameters.values())
            if ix < dimensionality
        ]
        self.fit_parameters = [
            item
            for ix, item in enumerate(sig.parameters.values())
            if ix >= dimensionality
        ]
        self.data = data
        self.guess = guess
        self.vector = vector
        self.dependent_variable = dependent_variable

        self.curve_fit = None
        self.fitted_curve = None
        self.residual = None
        self.det = None

    def fit_wrap(self):
        @wraps(self.underlying_function)
        def wrapped_fit(independent_variable, *fit_parameters):
            variable_components = [
                independent_variable[n] for n in range(self.dimensionality)
            ]
            exploded_function = self.underlying_function(
                *variable_components, *fit_parameters
            )
            return exploded_function

        # rewrite the signature so that curve_fit will like it
        sig = signature(wrapped_fit)
        curve_fit_params = (
            Parameter("independent_variable", Parameter.POSITIONAL_ONLY),
            *self.fit_parameters,
        )
        wrapped_fit.__signature__ = sig.replace(parameters=curve_fit_params)
        return wrapped_fit

    # todo: make_vector_partial

    def make_vector(
        self,
        data: pd.DataFrame = None,
        independent_variables: Sequence[Union[np.ndarray, pd.Series, str]] = None,
    ):
        """
        independent_parameters is a list of either strings or
        sequences (ndarray, series, whatever). if it's a string,
        that string should match some column of the passed dataframe,
        which gets used as that dimension of the vector
        if it's a sequence, it just gets directly plugged in as that dimension
        of the vector.
        """
        if data is None:
            data = self.data
        if independent_variables is None:
            independent_variables = [var.name for var in self.independent_variables]
        vect = []
        for var in independent_variables:
            if isinstance(var, str):
                try:
                    vect.append(data[var])
                except KeyError:
                    raise KeyError(
                        var
                        + " not found in columns of passed DataFrame."
                        + " Consider using make_vector_partial and filling"
                        + " other dimensions in as needed."
                    )
            else:
                vect.append(var)
        assert all_equal(
            [len(var) for var in vect]
        ), "All elements of the vector must have equal length."
        self.vector = vect
        return vect

    def fit(
        self,
        dependent_variable: Union[np.ndarray, pd.Series, str, None] = None,
        guess: Optional[Sequence[float]] = None,
    ):
        """
        fits underlying_function to self.data[dependent_variable],
        or the passed dependent variable

        warning! overwrites dependent_variable, guess if explicitly-defined,
        and always overwrites self.fit &c attributes
        """
        if dependent_variable is None:
            dependent_variable = self.dependent_variable
        if isinstance(dependent_variable, str):
            dependent_variable = self.data[dependent_variable]
        if (dependent_variable is None) or (self.vector is None):
            raise ValueError(
                "Can't try to fit function parameters without both"
                + " independent and dependent variables."
            )
        self.dependent_variable = dependent_variable

        if guess is not None:
            self.guess = guess

        self.curve_fit = curve_fit(
            self.fit_wrap(),
            self.vector,
            self.dependent_variable,
            maxfev=20000,
            p0=self.guess,
        )
        self.fitted_curve = self.fit_wrap()(self.vector, *self.curve_fit[0])
        self.residual = self.dependent_variable - self.fitted_curve
        self.det = coef_det(self.fitted_curve, self.dependent_variable)

    def plot_residuals(self):
        plt.plot(self.fitted_curve)
        plt.plot(self.dependent_variable, "og", markersize=2)

    def correlations(self, data=None):
        if data is None:
            data = self.data
        return correlation_matrix(
            data,
            matrix_columns=[self.residual] + [col for col in numeric_columns(data)],
            column_names=["residual"] + [col for col in numeric_columns(data)],
        )
