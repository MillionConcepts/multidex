import matplotlib.pyplot as plt
import numpy as np


def distpeek(frozendist):
    peek_domain = np.linspace(frozendist.ppf(0.01), frozendist.ppf(0.99), 100)
    plt.scatter(peek_domain, frozendist.pdf(peek_domain))


def square_window(x, y, window_size, array_shape):
    min_x = max(x - window_size, 0)
    max_x = min(x + window_size, array_shape[1])
    min_y = max(y - window_size, 0)
    max_y = min(y + window_size, array_shape[0])
    return slice(min_y, max_y), slice(min_x, max_x)


def stat_window(array, x, y, window_size, stat_op):
    y_slice, x_slice = square_window(x, y, window_size, array.shape)
    window = array[y_slice, x_slice, ...]
    return stat_op(window, axis=None)


def windowed_stat_array(image_array, stat_op, window_size, verbose=None):
    blank = np.empty(image_array[:, :, 0].shape)
    for y_x, _ in np.ndenumerate(image_array[:, :, 0]):
        blank[y_x[0], y_x[1]] = stat_window(
            image_array, y_x[1], y_x[0], window_size, stat_op
        )
        if verbose is not None:
            if (y_x[0] + y_x[1]) % verbose == 0:
                print(y_x[0], y_x[1])
    return blank