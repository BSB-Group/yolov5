import numpy as np

def pitch_and_theta_from_points(x1: float, y1: float, x2: float, y2: float):
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
        pitch (float): in [0,1]
        theta (float): in radians normalized to [0, 1]
    """ 
    assert x1 != x2, "Line is not allowed to be perfectly vertical or pitch would be infinite"

    if (x1 > x2):
        tmp_x, tmp_y = x2, y2
        x2 = x1
        y2 = y1
        x1 = tmp_x
        y1 = tmp_y

    y1 = 1-y1
    y2 = 1-y2
    
    m, b = points_to_slope_intercept(x1, y1, x2, y2)
    pitch = m*0.5+b
    theta = np.arctan(m) # rad [-pi/2, pi/2]
    theta = (theta + 0.5*np.pi)/np.pi # [0, 1]
    
    return pitch, theta

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
    if m == np.inf:
        x_1, y_1 = b, 0
        x_2, y_2 = b, h
    else:
        x_1, y_1 = 0, b
        x_2, y_2 = w, m * w + b
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