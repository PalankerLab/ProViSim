"""
utils.py

This module contains general utility functions.
"""
from typing import Union

import numpy as np
from PIL import Image


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


def crop_to_square(input_image: Union[str, np.ndarray]) -> np.ndarray:
	"""
	Load an image from a path or accept an NumPy array, and crop it to a centered square.
	"""
	# if input is a path, load image
	if isinstance(input_image, str):
		arr = np.array(Image.open(input_image))

	# if input is an array, use it directly
	elif isinstance(input_image, np.ndarray):
		arr = input_image

	# check if already square
	h, w = arr.shape[:2]
	if h == w:
		return arr

	# crop to square
	h, w = arr.shape[:2]
	size = min(h, w)
	top = (h - size) // 2
	left = (w - size) // 2

	# return square image
	return arr[top:top + size, left:left + size]
