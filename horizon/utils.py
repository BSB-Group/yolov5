import cv2
import numpy as np
import torch


def postprocess_x_pitch_theta(x_pitch, x_theta):
    """
    TODO: refactor postprocessing
    """
    # get maximum pitch and theta and normalise
    x_pitch, x_theta = x_pitch.softmax(1), x_theta.softmax(1)
    conf_pitch, pitch = x_pitch.max(1, keepdim=True)
    pitch = pitch / x_pitch.shape[-1]
    conf_theta, theta = x_theta.max(1, keepdim=True)
    theta = theta / x_theta.shape[-1]
    return torch.cat((pitch, conf_pitch), 1), torch.cat((theta, conf_theta), 1)


def points_to_pitch_theta(x1: float, y1: float, x2: float, y2: float):
    """Parameterize line that is spanned by the two points (x1, y1) and (x2, y2) through pitch and theta
       pitch ... y-value of line where x=0.5 (pitch=0 is at the bottom, pitch=1 is at the top)
       theta ... angle between line and y=0.5 (aka horizontal line)

       Assumptions:
       Coordinates are in range [0, 1].
       Origin is top left corner.

    Args:
        x1 (float): x-coordinate of point 1
        y1 (float): y-coordinate of point 1
        x2 (float): x-coordinate of point 2
        y2 (float): y-coordinate of point 1

    Returns:
        pitch (float): in [0,1] (pitch=0.25 is bottom, pitch=0.75 is top)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)
    """
    assert x1 != x2, "Line is not allowed to be perfectly vertical or pitch would be infinite"

    if (x1 > x2):
        tmp_x, tmp_y = x2, y2
        x2 = x1
        y2 = y1
        x1 = tmp_x
        y1 = tmp_y

    y1 = 1 - y1
    y2 = 1 - y2

    m, b = points_to_slope_intercept(x1, y1, x2, y2)
    pitch = m * 0.5 + b
    theta = np.arctan(m)  # rad [-pi/2, pi/2]

    # "encode" to [0, 1]
    pitch = 0.7 * pitch + 0.15  # [0.15, 0.85] is inside the image
    theta = (theta + 0.5 * np.pi) / np.pi  # [0, 1]

    return pitch, theta


def pitch_theta_to_slope_intercept(pitch: float, theta: float):
    """
    Convert pitch and theta to slope-intercept form of line: m, b.

    Args:
        pitch (float): in [0,1] (pitch=0.25 is bottom, pitch=0.75 is top)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)

    Returns:
        m, b
    """

    # "decode" from [0, 1]
    pitch = (pitch - 0.15) / 0.7  # [0, 1]
    theta = theta * np.pi - 0.5 * np.pi  # rad [-pi/2, pi/2]

    m = np.tan(theta)
    b = pitch - m * 0.5
    return m, b


def pitch_theta_to_points(pitch: float, theta: float, w: int = 1, h: int = 1):
    """
    Convert pitch and theta to two points.

    Args:
        pitch (float): in [0,1] (pitch=0.25 is bottom, pitch=0.75 is top)
        theta (float): in [0,1] (theta=0 is -pi/2, theta=1 is pi/2)

    Returns:
        (x1, y1), (x2, y2)
    """
    m, b = pitch_theta_to_slope_intercept(pitch, theta)
    (x1, y1), (x2, y2) = slope_intercept_to_points(m, b, w, h)
    y1, y2 = h - y1, h - y2  # invert y-axis since origin is top left corner
    return (x1, y1), (x2, y2)


def slope_intercept_to_points(m: float, b: float, w: int = 1, h: int = 1):
    if m == np.inf:
        x_1, y_1 = b * h, 0
        x_2, y_2 = b * h, h
    else:
        x_1, y_1 = 0, b * h
        x_2, y_2 = w, m * w + b * h
    return (x_1, y_1), (x_2, y_2)


def points_to_hough(x_1, y_1, x_2, y_2):
    """Convert two points to hough form of line: rho, theta."""
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


def hough_to_points(rho, theta, h=1, w=1):
    """
    Convert hough form of line to two points.
    Points are located on the image border.
    Points are normalised unless h and w are provided.

    hough form --> slope-intercept form --> points.
    """
    m, b = hough_to_slope_intercept(rho, theta, h, w)
    (x_1, y_1), (x_2, y_2) = slope_intercept_to_points(m, b, w, h)
    return (x_1, y_1), (x_2, y_2)


def hough_to_slope_intercept(rho, theta, h=1, w=1):
    """Convert hough form of line to slope-intercept form."""
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


def draw_horizon(image,
                 keypoints=None,
                 pitch_theta=None,
                 hough=None,
                 color=(0, 255, 0),
                 diameter=2):
    """
    Visualize horizon line on image.
    """
    assert np.any([arg is not None for arg in [keypoints, pitch_theta, hough]]), \
        "Provide at least one of keypoints, theta_pitch, hough"

    image = image.copy()

    if keypoints is not None:
        x1, y1, x2, y2 = np.array(keypoints).flatten()
        for (x, y) in keypoints:
            cv2.circle(image, (int(x), int(y)), diameter * 5, (0, 255, 0), -1)

    if pitch_theta is not None:
        (x1, y1), (x2, y2) = pitch_theta_to_points(*pitch_theta, image.shape[1], image.shape[0])

    if hough is not None:
        (x1, y1), (x2, y2) = hough_to_points(*hough, image.shape[1], image.shape[0])

    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, diameter)
    return image
