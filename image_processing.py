"""
image_processing.py

This module contains functions for processing images such as spatial filtering and contrast changing.
"""

import numpy as np

from PIL import Image
from skimage.filters import window

from utils import validate_gray_img

def rgb_to_gray(image_path: str=None, img=None) -> np.ndarray:
    """
    Converts an RGB image to grayscale.

    This function reads an image from the provided path, converts it to grayscale using
    the 'L' mode (luminosity), normalizes the grayscale values to the range [0, 1],
    and validates the result.

    Args:
        image_path (str): The file path to the input image.
        image (np.ndarray): The numpy array of the input image.

    Returns:
        np.ndarray: A 2D NumPy array representing the grayscale image with values in the range [0, 1].

    Raises:
        FileNotFoundError: If the specified image path does not exist.
        IOError: If the file at the given path is not a valid image file.
    """
    try:
        # Open the image and convert to grayscale ('L' mode represents luminosity)
        if img is None and image_path is not None:
          img = Image.open(image_path)
        if img is not None and image_path is None:
          img = Image.fromarray(img)
        if img is None and image_path is None:
          raise ValueError("Either image_path or img must be provided.")
        gray_img = np.array(img.convert('L'))

        if gray_img.dtype == np.uint8:
            # Normalize pixel values to the range [0, 1]
            gray_img = gray_img.astype(np.float32) / 255.0

        # Validate the grayscale image (ensure values are within the expected range)
        validate_gray_img(gray_img)

        return gray_img

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at '{image_path}' was not found.")
    except IOError:
        raise IOError(f"The file at '{image_path}' is not a valid image file.")


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
    Compute the inverse sigmoid function.

    Parameters:
        y (np.ndarray): The sigmoid output values, must be in the range (0, 1).
        gain (float): The steepness of the sigmoid curve.
        x_shift (float): The horizontal shift of the sigmoid curve.

    Returns:
        np.ndarray: The inverse sigmoid values for the input `y`.

    """
    # Clip y to avoid division by zero or log of negative values
    epsilon = 1e-10  # Small constant to prevent numerical instability
    y = np.clip(y, epsilon, 1 - epsilon)

    # Compute the inverse sigmoid
    inverse = (-1 / gain) * np.log((1 / y) - 1) + x_shift

    return inverse
