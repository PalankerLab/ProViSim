"""
utils.py

This module contains general utility functions.
"""

import numpy as np

def validate_gray_img(gray_img):
    """Validate the given image for several requirements."""
    if not isinstance(gray_img, np.ndarray):
        raise TypeError("gray_img must be a NumPy array.")
    if gray_img.ndim != 2:
        raise ValueError("gray_img must be a 2D array.")
    if gray_img.shape[0] != gray_img.shape[1]:
        raise ValueError(f"gray_img must be square, but got shape {gray_img.shape}.")
    if not np.issubdtype(gray_img.dtype, np.floating):
        raise TypeError("gray_img must have a floating-point data type.")
    if not np.all((gray_img >= 0) & (gray_img <= 1)):
        raise ValueError("gray_img values must be in the range [0, 1].")


# Define the linear function (y = x)
def linear_function(x):
    return x
