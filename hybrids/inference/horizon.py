import numpy as np


def points_to_normal(x_1, y_1, x_2, y_2):
    """Convert two points to normal form of line: rho, theta."""
    if x_1 == x_2:  # vertical line
        return x_1, 0
    m = (y_2 - y_1) / (x_2 - x_1)
    a, b, c = -m, 1, y_1 - m * x_1
    rho = (y_1 - m * x_1) / np.sqrt(a**2 + b**2)
    theta = np.arctan2(b, a)
    return rho, theta


def points_to_slope_intercept(x_1, y_1, x_2, y_2):
    """Convert two points to slope-intercept form of line: m, b."""
    if x_1 == x_2:  # vertical line
        return np.inf, x_1
    m = (y_2 - y_1) / (x_2 - x_1)
    b = y_1 - m * x_1
    return m, b


def normal_to_points(rho, theta, h=1, w=1):
    """
    Convert normal form of line to two points. Points are located on the image border. Points are normalised unless h
    and w are provided.

    normal from --> slope-intercept form --> points.
    """
    m, b = normal_to_slope_intercept(rho, theta, h, w)
    if m == np.inf:
        x_1, y_1 = b, 0
        x_2, y_2 = b, h
    else:
        x_1, y_1 = 0, b
        x_2, y_2 = w, m * w + b
    return (x_1, y_1), (x_2, y_2)


def normal_to_slope_intercept(rho, theta, h=1, w=1):
    """Convert normal form of line to slope-intercept form."""
    if theta == 0:
        return np.inf, rho * w
    if theta == np.pi / 2:
        return 0, rho * h
    a, b, c = np.cos(theta), np.sin(theta), rho
    m = -a / b
    b = -c / b
    # take into account image dims
    m *= h / w
    b = -b * h
    return m, b
