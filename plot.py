"""
plot.py

This module contains functions to plot figures and save them.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import validate_gray_img


def fft_plot(gray_img: np.ndarray) -> np.ndarray:
    """
    Compute and return the log-scaled magnitude spectrum of the Fourier Transform of a grayscale image.

    This function performs the following steps:
    1. Computes the 2D Fourier Transform of the input grayscale image.
    2. Shifts the zero-frequency component to the center of the spectrum.
    3. Calculates the magnitude of the Fourier Transform.
    4. Applies a logarithmic scale for better visualization of the spectrum.

    Args:
        gray_img (np.ndarray): Input grayscale image as a 2D array with values in [0, 1].

    Returns:
        np.ndarray: The log-scaled magnitude spectrum of the Fourier Transform.

    Raises:
        ValueError: If the input image is not a 2D array.
    """
    # Validate the input image
    validate_gray_img(gray_img)

    # Compute the Fourier Transform and shift the zero-frequency component
    gray_img_FFT = np.fft.fftshift(np.fft.fft2(gray_img))

    # Compute the magnitude spectrum and apply log scaling
    magnitude_spectrum = np.abs(gray_img_FFT)
    log_magnitude_spectrum = np.log1p(magnitude_spectrum)

    return log_magnitude_spectrum

class CampbellRobsonChart:
    """
    A class to generate a Campbell-Robson chart, a 2D visualization of spatial frequency
    (x-axis) and contrast (y-axis).

    The chart visualizes a range of spatial frequencies (left to right) and contrast levels
    (top to bottom), often used in vision research.
    """
    def __init__(self, image_size: int = 4096, freq_start: float = -1, freq_end: float = 1.8,
                contrast_start: float = 0, contrast_end: float = -3):
        """
        Initializes the CampbellRobsonChart object with given parameters for image size,
        frequency range, and contrast range.

        Parameters:
            image_size (int): Size of the square chart (both width and height).
            freq_start (float): Start of the frequency range (logarithmic scale).
            freq_end (float): End of the frequency range (logarithmic scale).
            contrast_start (float): Start of the contrast range (logarithmic scale).
            contrast_end (float): End of the contrast range (logarithmic scale).
        """
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        if freq_start >= freq_end:
            raise ValueError("freq_start must be less than freq_end.")
        if contrast_start <= contrast_end:
            raise ValueError("contrast_start must be greater than contrast_end.")

        self.image_size = image_size
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.contrast_start = contrast_start
        self.contrast_end = contrast_end

        # Generate the chart and ticks
        self.chart_data = self.draw_campbell_robson_chart()
        self.ticks = self.generate_tick_positions()

    def draw_campbell_robson_chart(self) -> np.ndarray:
        """
        Generates and returns the Campbell-Robson chart as a 2D numpy array, 
        normalized to the range [0, 1].

        Returns:
            np.ndarray: The Campbell-Robson chart.
        """

        # Frequency increases from left to right
        frequencies = np.array([10 ** x for x in np.linspace(self.freq_start, self.freq_end, self.image_size)])

        # Contrast increases from top to bottom
        contrast = np.array([10 ** x for x in np.linspace(self.contrast_start, self.contrast_end, self.image_size)])

        # Create a 2D meshgrid for coordinates
        y_indices, x_indices = np.meshgrid(range(self.image_size), range(self.image_size), indexing='ij')

        # Compute the sinusoidal grating modulated by contrast
        chart = contrast[y_indices] * np.sin(2 * np.pi * frequencies[x_indices])

        # Normalize the chart to [0, 1]
        chart_normalized = (chart - chart.min()) / (chart.max() - chart.min())

        return chart_normalized


    def generate_tick_positions(self):
        """
        Generate tick positions and labels for a plot based on the provided frequencies and contrast ranges.
        The tick values are automatically chosen as powers of 10 (10^integer) with intermediate ticks, inclusive of the range start and end.

        Returns:
            dict: A dictionary with keys:
                - 'y_positions': Positions of y-axis ticks.
                - 'y_labels': Labels for y-axis ticks.
                - 'x_positions': Positions of x-axis ticks.
                - 'x_labels': Labels for x-axis ticks.
        """
        # Generate frequencies and contrast arrays
        frequencies = np.array([10 ** x for x in np.linspace(self.freq_start, self.freq_end, self.image_size)])
        contrast = np.array([10 ** x for x in np.linspace(self.contrast_start, self.contrast_end, self.image_size)])

        # Determine y-tick values as 10^(integer) within the contrast range
        y_min_log = int(np.floor(np.log10(min(contrast))))
        y_max_log = int(np.ceil(np.log10(max(contrast))))
        y_tick_values = [10 ** i for i in range(y_min_log, y_max_log + 1)]

        # Find y-tick positions
        y_tick_positions = [np.argmin(np.abs(contrast - tick)) for tick in y_tick_values]
        y_tick_labels = [f"{tick:g}" for tick in y_tick_values]

        # Generate intermediate y-tick values (logarithmic spacing)
        y_intermediate_ticks = []
        for i in range(len(y_tick_values) - 1):
            lower, upper = y_tick_values[i], y_tick_values[i + 1]
            y_intermediate_ticks.extend(np.linspace(lower, upper, 10)[1:-1])

        # Find intermediate y-tick positions
        y_intermediate_positions = [
            np.argmin(np.abs(contrast - tick)) for tick in y_intermediate_ticks
        ]

        # Determine x-tick values as 10^(integer) within the frequencies range
        x_min_log = int(np.floor(np.log10(min(frequencies))))
        x_max_log = int(np.ceil(np.log10(max(frequencies))))
        x_tick_values = [10 ** i for i in range(x_min_log, x_max_log + 1)]

        # Find x-tick positions
        x_tick_positions = [np.argmin(np.abs(frequencies - tick)) for tick in x_tick_values]
        x_tick_labels = [f"{tick:g}" for tick in x_tick_values]

        # Generate intermediate x-tick values (logarithmic spacing)
        x_intermediate_ticks = []
        for i in range(len(x_tick_values) - 1):
            lower, upper = x_tick_values[i], x_tick_values[i + 1]
            x_intermediate_ticks.extend(np.linspace(lower, upper, 10)[1:-1])

        # Find intermediate x-tick positions
        x_intermediate_positions = [
            np.argmin(np.abs(frequencies - tick)) for tick in x_intermediate_ticks
        ]

        # Combine y-ticks
        all_y_positions = y_tick_positions + y_intermediate_positions
        all_y_labels = y_tick_labels + ["" for _ in range(len(y_intermediate_positions))]

        # Combine x-ticks
        all_x_positions = x_tick_positions + x_intermediate_positions
        all_x_labels = x_tick_labels + ["" for _ in range(len(x_intermediate_positions))]

        return {
            "y_positions": all_y_positions,
            "y_labels": all_y_labels,
            "x_positions": all_x_positions,
            "x_labels": all_x_labels,
        }


    def save_campbell_robson_chart_png(self, filename, data=None, axes_label=True):
        """
        Saves the Campbell-Robson chart as a PNG file with optional axis labels.

        Parameters:
            filename (str): The file path to save the chart.
            axes_label (bool): Whether to include axis labels in the saved image.
        """
        if data is None:
            data = self.chart_data

        fig = plt.figure(figsize=(2, 2))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax)
        ax.imshow(data, cmap='gray', origin='lower')

        # Set x-ticks for frequency labels
        ax.set_xticks(self.ticks['x_positions'])
        ax.set_xticklabels(self.ticks['x_labels'])
        if axes_label:
            ax.set_xlabel('Spatial Frequency (cpd)')

        # Set y-ticks for frequency labels
        ax.set_yticks(self.ticks['y_positions'])
        ax.set_yticklabels(self.ticks['y_labels'])
        if axes_label:
            ax.set_ylabel('Contrast')

        fig.savefig(filename, dpi=data.shape[0], bbox_inches='tight')
        plt.close(fig)


def draw_landolt_c(gap_width: int) -> np.ndarray:
    """
    Draws a signle Landolt C image (white letter on black background).

    This function generates a Landolt C, commonly used in vision tests, which consists of an outer circle
    (white) and an inner circle (black) with a gap in the black ring. The gap is drawn as a black rectangle.

    Parameters:
        image_size (int): The size of the generated image (height and width in pixels).
        gap_width (int): The width of the black gap in the Landolt C.

    Returns:
        np.ndarray: A binary image (0 for black, 255 for white) of the Landolt C symbol.
    """
    # The image_size, inner and outer radii are defined based on the gap width
    inner_radius = np.round(gap_width / 2 * 3).astype(np.uint16)
    outer_radius = np.round(gap_width / 2 * 5).astype(np.uint16)
    image_size = outer_radius * 2 + 1

    # Ensure the image size is large enough for the circles
    if image_size < outer_radius * 2:
        raise ValueError("image_size must be at least twice the outer_radius to fit the Landolt C.")

    # Initialize an empty black image
    img = np.zeros((image_size, image_size), dtype=np.uint8)

    # Define the center of the image
    center = (image_size // 2, image_size // 2)

    # Draw the outer white circle
    cv2.circle(img, center, outer_radius, 255, -1)  # Outer circle (white)

    # Draw the inner black circle
    cv2.circle(img, center, inner_radius, 0, -1)    # Inner circle (black)

    # Create the gap in the Landolt-C by drawing a black rectangle
    gap_start = center[0]
    gap_end = center[0] + outer_radius
    top_left = (gap_start, center[1] - gap_width // 2)
    bottom_right = (gap_end, center[1] + gap_width // 2)
    cv2.rectangle(img, top_left, bottom_right, 0, -1)  # Draw the black rectangle gap

    return img


def draw_landolt_c_panel(
    gap_width_ratios: np.ndarray = np.array([1.2, 1.0, 0.8, 0.7, 0.6]),
    positions: np.ndarray = np.array([(70, 100), (90, 330), (350, 100), (360, 260), (370, 400)]),
    canvas_size: int = 550,
    num_pixels_per_width: int = 20
) -> np.ndarray:
    """
    Draws a panel of Landolt C symbols with varying gap widths at specified positions on a canvas.

    This function generates an image of size `canvas_size` x `canvas_size`, placing multiple Landolt C
    symbols on the canvas. Each symbol has a gap width determined by the ratio in `gap_width_ratios`
    and is placed at the corresponding coordinates in `positions`.

    Parameters:
        gap_width_ratios (np.ndarray): An array of ratios for the gap widths in the Landolt C symbols.
                                       Each ratio will be multiplied by the implant pixel width to
                                       determine the gap size.
        positions (np.ndarray): A list of tuples specifying the (y, x) positions where each Landolt C symbol
                                 will be placed on the canvas.
        canvas_size (int): The size of the canvas (height and width), in pixels.
        num_pixels_per_width (int): Number of implant pixels on the implant.

    Returns:
        np.ndarray: A 2D NumPy array representing the canvas with the drawn Landolt C symbols.
    """
    # Initialize a blank canvas
    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    # Determine the width of an implant pixel given the canvas size
    implant_pixel_width = canvas_size // num_pixels_per_width

    # Calculate the gap widths based on the given ratios
    gap_widths = implant_pixel_width * gap_width_ratios

    # Draw each Landolt C symbol at the specified positions
    for pos, gap_width in zip(positions, gap_widths):
        # Create the Landolt C symbol with the calculated gap width
        landolt_c = draw_landolt_c(np.round(gap_width).astype(np.uint16))

        # Extract the position
        y, x = pos

        # Ensure that the symbol fits within the canvas
        if y + landolt_c.shape[0] <= canvas_size and x + landolt_c.shape[1] <= canvas_size:
            # Place the Landolt C symbol onto the canvas at the specified position
            canvas[y : y + landolt_c.shape[0], x : x + landolt_c.shape[1]] = landolt_c
        else:
            raise ValueError(f"Landolt C symbol at position {pos} does not fit on the canvas.")

    # Normalize to float in range [0, 1]
    canvas_normalized = (canvas - canvas.min()) / (canvas.max() - canvas.min())

    return canvas_normalized
