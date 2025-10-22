"""
image_processing.py

This module contains functions for processing images such as spatial filtering and contrast changing.
"""
# standard imports
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from typing import Union
from typing import Optional
from skimage.filters import window

# project imports
from utils import validate_gray_img, crop_to_square


def rgb_to_gray(img_input: Union[str, np.ndarray, Image.Image], invert_color: bool = False) -> np.ndarray:
	"""
	Convert an RGB image to grayscale in [0,1].
	Do nothing if input is already grayscale in [0,1].
	"""
	# load image if input is a file path
	if isinstance(img_input, str):
		img = Image.open(img_input)
	else:
		img = img_input

	# convert PIL Image to NumPy array
	if isinstance(img, Image.Image):
		img = np.array(img)

	# if already grayscale [0,1], return as is
	if img.ndim == 2 and np.min(img) >= 0.0 and np.max(img) <= 1.0:
		return img

	# if RGB float [0,1] → convert to grayscale without rescaling
	if not isinstance(img, Image.Image):
		img = Image.fromarray(img)
	gray_img = np.array(img.convert('L')).astype(np.float32) / 255.0

	# if invert color, invert the grayscale image
	if invert_color:
		gray_img = invert_black_and_white(gray_img)

	return gray_img


def quantize_image(image, num_steps=14):
	"""
	Quantize an image into discrete levels.

	Args:
		image (np.ndarray): The input image with values in [0, 1].
		num_steps (int): The number of quantization levels. For PRIMA, it's 14.

	Returns:
		np.ndarray: The quantized image, still in float.
	"""
	# Validate input image
	validate_gray_img(image)

	# Quantize to the nearest level
	quantized_image = np.round(image * (num_steps - 1)) / (num_steps - 1)

	# Validate output image
	validate_gray_img(quantized_image)

	return quantized_image


def fft_filter(gray_img: np.ndarray, fourier_filter: np.ndarray) -> np.ndarray:
	"""
	Apply a Fourier-based filter to a grayscale image.

	This function performs the following steps:
	1. Computes the 2D FFT of the input grayscale image.
	2. Applies the provided Fourier filter in the frequency domain.
	3. Transforms the filtered image back to the spatial domain.
	4. Clips the real part of the filtered image to the range [0, 1].

	Args:
		gray_img (np.ndarray): Input grayscale image as a 2D array with values in [0, 1].
		fourier_filter (np.ndarray): Fourier filter to be applied,
									 must have the same shape as `gray_img`.

	Returns:
		np.ndarray: Filtered grayscale image with values clipped to [0, 1].
	"""
	# Validate inputs
	validate_gray_img(gray_img)
	validate_gray_img(fourier_filter)
	if gray_img.shape != fourier_filter.shape:
		raise ValueError("gray_img and fourier_filter must have the same shape.")

	# Perform FFT and apply the filter
	gray_img_FFT = np.fft.fftshift(np.fft.fft2(gray_img))
	gray_img_FFT_filt = gray_img_FFT * fourier_filter
	gray_img_filt = np.fft.ifft2(np.fft.ifftshift(gray_img_FFT_filt))

	# Return the real part of the result, clipped to [0, 1]
	return np.clip(gray_img_filt.real, 0, 1)


def create_tukey_window(cutoff_frequency: int, image_width: int, alpha: float = 0.3, radius_expansion_factor: float = 0.3) -> np.ndarray:
	"""
	Create a Tukey window centered within a square image.

	The Tukey window is generated based on the filter radius and padded to fit the given image width.

	Args:
		cutoff_frequency (int): The cutoff frequency for the lowpass filter, specified in cycles per image.
		image_width (int): The width (and height) of the square image in which the window will be centered.
		alpha (float): The shape parameter of the Tukey window, controlling the proportion of the window length that is tapered.
				- `alpha = 0`: The window becomes a rectangular window with no tapering.
				- `alpha = 1`: The window becomes a Hann window with full tapering.
				- Intermediate values (e.g., `0.3`) result in partial tapering, balancing the trade-off between
				  frequency resolution and spectral leakage reduction. Must be in the range `[0, 1]`.
		radius_expansion_factor (float): The fraction of the filter radius by which to expand the window radius to account for
								  saccadic eye movements. This adjustment helps capture additional frequencies introduced
								  by these movements. For example, a value of `0.3` increases the window radius by 30%
								  of the original filter radius. Must be in the range `[0, 1]`.

	Returns:
		np.ndarray: A 2D Tukey window padded to the specified image size.

	Raises:
		ValueError: If the calculated Tukey window size exceeds the given image width.
	"""
	# Compute the Tukey window size
	window_radius = int(round(2 * cutoff_frequency * (1 + radius_expansion_factor))) + 1

	# Ensure the Tukey window size fits within the image dimensions
	if window_radius > image_width:
		raise ValueError(
			f"The calculated Tukey window size ({window_radius}) exceeds the specified image width ({image_width})."
		)

	# Create the Tukey window
	w = window(('tukey', alpha), (window_radius, window_radius))

	# Calculate padding to center the Tukey window in the image
	pad_left = (image_width - window_radius) // 2
	pad_right = image_width - window_radius - pad_left

	# Pad the Tukey window to fit the image size
	tukey_win = np.pad(w, ((pad_left, pad_right), (pad_left, pad_right)), mode='constant')

	return tukey_win


def gamma_transform(input: np.ndarray, gamma: float) -> np.ndarray:
	"""
	Apply a gamma transformation to the input array.

	This function performs the following steps:
	1. Validates that the input is a floating-point array with values in [0, 1].
	2. Shifts and scales the input values to the range [-1, 1].
	3. Applies gamma correction, preserving the sign of the input values.
	4. Reverses the scaling to return the output to the range [0, 1].

	Args:
		input (np.ndarray): The input array with values in [0, 1].
		gamma (float): The gamma correction factor.

	Returns:
		np.ndarray: The gamma-transformed array with values in [0, 1].
	"""

	# Shift input to range [-1, 1]
	input_shifted = 2.0 * input - 1.0

	# Apply gamma correction
	input_gamma_corrected = np.sign(input_shifted) * (np.abs(input_shifted) ** gamma)

	# Reverse scaling to return to range [0, 1]
	input_reversed = (input_gamma_corrected + 1.0) / 2.0

	return input_reversed


def sigmoid_transform(input: np.ndarray, gain: float, x_shift: float) -> np.ndarray:
	"""
	Apply a sigmoid transformation to the input array.

	Args:
		input (np.ndarray): Input array with values in the range [0, 1].
		gain (float): Controls the steepness of the sigmoid curve.
		x_shift (float): Controls the horizontal shift of the sigmoid curve.

	Returns:
		np.ndarray: The transformed array with values in the range [0, 1].
	"""

	return 1 / (1 + np.exp(-gain * (input - x_shift)))


def inverse_sigmoid(y: np.ndarray, gain: float, x_shift: float) -> np.ndarray:
	"""
	Compute the inverse sigmoid function, with clipping chosen to
	avoid negative outputs and infinities.
	"""
	# small constant to prevent numerical instability
	epsilon = 1e-10

	# clip y to avoid division by zero or log of negative values
	y = np.clip(y, epsilon, 1 - epsilon)

	# compute the inverse sigmoid
	inverse = (-1 / gain) * np.log((1 / y) - 1) + x_shift

	# replace negative values with 0
	inverse = np.where(inverse < 0, 0.0, inverse)

	# replace inf with 1
	inverse = np.where(~np.isfinite(inverse), 1.0, inverse)

	return inverse

def invert_black_and_white(image: np.ndarray) -> np.ndarray:
	"""
	Swap bright and dark regions in a grayscale image.

	This function inverts the pixel values of the input grayscale image,
	effectively swapping bright areas with dark areas.

	Args:
		image (np.ndarray): Input grayscale image as a 2D array with values in [0, 1].

	Returns:
		np.ndarray: Grayscale image with bright and dark regions swapped.
	"""
	# Validate input image
	validate_gray_img(image)

	# Swap bright and dark regions
	swapped_image = 1.0 - image

	# Validate output image
	validate_gray_img(swapped_image)

	return swapped_image

def is_face_dark(detector, image: np.ndarray, threshold: float = 120.0) -> Optional[bool]:
	"""
	Determine if a detected face is dark based on the given threshold, focusing on skin regions only.
	"""
	# crop square and convert for MediaPipe
	image = crop_to_square(image)
	mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

	detection_result = detector.detect(mp_image)
	if not detection_result.face_landmarks:
		return None

	landmarks = detection_result.face_landmarks[0]
	h, w, _ = image.shape

	# convert normalized landmarks to pixel coordinates
	coords = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])

	# approximate skin region: cheeks + forehead + jaw
	skin_indices = list(range(10, 338))

	skin_points = coords[skin_indices]
	mask = np.zeros((h, w), dtype=np.uint8)
	cv2.fillPoly(mask, [skin_points], 255)

	# convert to YCbCr and take Y channel
	ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
	y_channel = ycrcb[:, :, 0]

	# compute mean luminance over skin only
	mean_y = np.mean(y_channel[mask == 255])

	return mean_y < threshold
