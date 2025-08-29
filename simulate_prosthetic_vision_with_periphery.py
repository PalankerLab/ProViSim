#!/usr/bin/env python3
"""
A Tool for Simulating Prosthetic Vision with Peripheral Blur and Central Scotoma
Simulates reduced peripheral resolution and a central scotoma with an overlay of prosthetic vision in the central
visual field.

The text recognition model used is the open-source EasyOCR.

Authors: Anna Kochnev Goldstein
Date: 2025-08-26
"""

import os
import cv2
import math
import easyocr
import numpy as np
from typing import Tuple
from PIL import Image, ImageDraw, ImageFont

from image_processing import rgb_to_gray, create_tukey_window, fft_filter, gamma_transform, sigmoid_transform
from utils import validate_gray_img


class VisionSimulator:
	"""Main class for peripheral vision simulation with a central scotoma."""

	def __init__(self, input_image, output_image, viewing_distance_cm: float = 60.0,
				 image_width_cm: float = 14.0, image_height_cm: float = 20.0,
				 sigma_center: float = 0.5, eccentricity_k: float = 0.8,
				 scotoma_radius_deg: float = 12.0, implant_radius_deg: float = 8.0,
				 scotoma_fill_color: Tuple[int, int, int] = (0, 0, 0), cutoff_freq: int = 10,
			     tukey_alpha: float = 0.3, tukey_radius_expansion_factor: float = 0.3,
				 gamma: float = 3.5, zoom_factor: float = 2.0, apply_contrast_reduction: bool = True):

		# save input parameters
		self.input_image = input_image
		self.output_image = output_image
		self.viewing_distance_cm = viewing_distance_cm
		self.image_width_cm = image_width_cm
		self.image_height_cm = image_height_cm
		self.sigma_center = sigma_center
		self.eccentricity_k = eccentricity_k
		self.scotoma_radius_deg = scotoma_radius_deg
		self.implant_radius_deg = implant_radius_deg
		self.scotoma_fill_color = scotoma_fill_color
		self.cutoff_freq = cutoff_freq
		self.tukey_alpha = tukey_alpha
		self.tukey_radius_expansion_factor = tukey_radius_expansion_factor
		self.gamma = gamma
		self.zoom_factor = zoom_factor
		self.apply_contrast_reduction = apply_contrast_reduction

		# preload OCR text recognition model
		model_path = os.path.join(os.path.dirname(__file__), 'text_recognition_models')
		self.reader = easyocr.Reader(['en'], model_storage_directory=model_path, download_enabled=False)

	def degrees_to_pixels(self, image_array: np.ndarray, degrees: float) -> float:
		"""
		Convert a visual angle in degrees to pixel distance from image center,
		accounting for the image's physical dimensions and pixel dimensions
		in both axes.

		Args:
			image_array: NumPy array of the image to get pixel dimensions
			degrees: Visual angle in degrees (radial)

		Returns:
			Pixel distance (radial) corresponding to the visual angle
		"""
		# get the height and width of the image
		height_px, width_px = image_array.shape[:2]

		# physical size in cm corresponding to the visual angle
		half_angle_rad = math.radians(degrees / 2.0)
		size_cm = 2.0 * self.viewing_distance_cm * math.tan(half_angle_rad)

		# pixel size along each axis
		pixel_size_x_cm = self.image_width_cm / width_px
		pixel_size_y_cm = self.image_height_cm / height_px

		# convert size_cm to pixels along each axis
		pixels_x = size_cm / pixel_size_x_cm
		pixels_y = size_cm / pixel_size_y_cm

		# compute radial pixel distance
		radial_pixels = math.sqrt(pixels_x ** 2 + pixels_y ** 2) / math.sqrt(2)

		return radial_pixels

	def create_eccentricity_map(self, image_array: np.ndarray) -> np.ndarray:
		"""
		Create an eccentricity map in degrees from the image center.
		"""
		# get the height and width of the image
		height_px, width_px = image_array.shape[:2]

		# use them to calculate the center of the image
		center_y, center_x = height_px // 2, width_px // 2

		# convert to pixel distances
		y_coords, x_coords = np.ogrid[:height_px, :width_px]
		dx_px = x_coords - center_x
		dy_px = y_coords - center_y

		# convert pixel distances to cm
		pixel_size_x_cm = self.image_width_cm / width_px
		pixel_size_y_cm = self.image_height_cm / height_px
		distances_cm = np.sqrt((dx_px * pixel_size_x_cm) ** 2 + (dy_px * pixel_size_y_cm) ** 2)

		eccentricity_deg = np.degrees(np.arctan(distances_cm / self.viewing_distance_cm))

		return eccentricity_deg

	def apply_eccentricity_blur(self, image: np.ndarray) -> np.ndarray:
		"""
		Apply spatially varying Gaussian blur based on eccentricity.

		Blur increases with eccentricity: sigma_px = sigma_center + k * eccentricity_deg

		Args:
		    image: Input image as numpy array

		Returns:
		    Blurred image with peripheral resolution reduction
		"""
		if len(image.shape) == 3:
			height, width, channels = image.shape
			result = np.zeros_like(image)

			# Process each channel separately
			for c in range(channels):
				result[:, :, c] = self._apply_blur_single_channel(image[:, :, c])
		else:
			result = self._apply_blur_single_channel(image)

		return result

	def _apply_blur_single_channel(self, channel: np.ndarray) -> np.ndarray:
		"""Apply eccentricity blur to a single channel using the calculated sigma map."""
		# obtain an eccentricity map
		eccentricity_map = self.create_eccentricity_map(channel)

		# use the eccentricity map to calculate a sigma (blur) map: sigma increases with eccentricity
		sigma_map = self.sigma_center + self.eccentricity_k * eccentricity_map

		# apply spatially varying blur using a layer-based approach
		result = np.zeros_like(channel, dtype=np.float64)
		weight_sum = np.zeros_like(channel, dtype=np.float64)
		
		# create blur layers with different sigma values
		min_sigma = max(0.1, np.min(sigma_map))
		max_sigma = np.max(sigma_map)
		sigma_values = np.linspace(min_sigma, max_sigma, 6)
		
		for sigma in sigma_values:
			# apply Gaussian blur with current sigma
			if sigma < 0.1:
				blurred_layer = channel.astype(np.float64)
			else:
				blurred_layer = cv2.GaussianBlur(channel.astype(np.float64), (0, 0), sigma)
			
			# weight based on distance from target sigma
			weight = np.exp(-((sigma_map - sigma) ** 2) / (2 * (max_sigma / 6) ** 2))
			
			# accumulate weighted results
			result += weight * blurred_layer
			weight_sum += weight
		
		# normalize by total weights to preserve brightness
		weight_sum = np.maximum(weight_sum, 1e-10)
		result = result / weight_sum

		return result.astype(channel.dtype)

	def create_scotoma(self, image: np.ndarray, edge_width_deg: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
		"""
		Create a central scotoma with a soft edge (partial transparency).

		Args:
			image: Input image (H x W x 3 or H x W)
			edge_width_deg: Width of the fading edge in degrees

		Returns:
			Tuple of (image_with_scotoma, scotoma_mask)
		"""
		# get the height and width of the image
		height, width = image.shape[:2]

		# use them to calculate the center coordinates
		center_y, center_x = height // 2, width // 2

		# convert scotoma radius and edge width from degrees to pixels
		scotoma_radius_px = self.degrees_to_pixels(image, degrees=self.scotoma_radius_deg)  # central 12°
		edge_width_px = self.degrees_to_pixels(image, degrees=edge_width_deg)

		# create a distance map from the center
		y_coords, x_coords = np.ogrid[:height, :width]
		distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

		# create a soft mask: 1 = full scotoma, 0 = no scotoma, linear fade in edge
		scotoma_mask = np.clip((scotoma_radius_px + edge_width_px - distances) / edge_width_px, 0, 1)

		# apply scotoma with a soft edge
		result = image.copy().astype(np.float32)
		if len(image.shape) == 3:
			for c in range(3):
				result[:, :, c] = result[:, :, c] * (1 - scotoma_mask) + self.scotoma_fill_color[c] * scotoma_mask
		else:
			result = result * (1 - scotoma_mask) + self.scotoma_fill_color[0] * scotoma_mask

		return result.astype(image.dtype), scotoma_mask

	def extract_central_field_image(self, image: np.ndarray, field_radius_deg: float = 4.0) -> tuple:
		"""
		Extract the central field as a square image based on the specified visual angle,
		and return the circular mask indices within that square.

		Args:
			image: Input image
			field_radius_deg: Radius of central field in degrees (default: 4.0)

		Returns:
			Tuple of:
			- Square image containing the central field content
			- Circular mask of shape (square_size, square_size) with True inside the circular field
		"""
		# extract the height and width of the image
		height, width = image.shape[:2]

		# use them to calculate the center coordinates
		center_y, center_x = height // 2, width // 2

		# convert field radius from degrees to pixels
		field_radius_px = self.degrees_to_pixels(image, field_radius_deg)

		# define a square bounding box
		square_length = int(2 * field_radius_px)
		half_size = square_length // 2

		start_y = max(0, center_y - half_size)
		end_y = min(height, center_y + half_size)
		start_x = max(0, center_x - half_size)
		end_x = min(width, center_x + half_size)

		# extract the calculated square
		if len(image.shape) == 3:
			central_square = image[start_y:end_y, start_x:end_x, :]
		else:
			central_square = image[start_y:end_y, start_x:end_x]

		actual_height, actual_width = central_square.shape[:2]

		# pad if necessary to maintain square size
		if actual_height != square_length or actual_width != square_length:
			if len(image.shape) == 3:
				padded_square = np.zeros((square_length, square_length, image.shape[2]), dtype=image.dtype)
			else:
				padded_square = np.zeros((square_length, square_length), dtype=image.dtype)

			pad_y = (square_length - actual_height) // 2
			pad_x = (square_length - actual_width) // 2

			if len(image.shape) == 3:
				padded_square[pad_y:pad_y + actual_height, pad_x:pad_x + actual_width, :] = central_square
			else:
				padded_square[pad_y:pad_y + actual_height, pad_x:pad_x + actual_width] = central_square

			central_square = padded_square

		# create a circular mask of the same size
		y_coords, x_coords = np.ogrid[:square_length, :square_length]
		center_square = square_length / 2
		circular_mask = (x_coords - center_square) ** 2 + (y_coords - center_square) ** 2 <= field_radius_px ** 2

		return central_square, circular_mask

	def zoom_in_image(self, img: np.ndarray, zoom_factor: float = 2.0) -> np.ndarray:
		"""
		Zoom in on the center of the image while keeping the same output dimensions.

		Args:
			img: Input image as a NumPy array (grayscale or RGB).
			zoom_factor: Factor by which to zoom in (>1 means zoom in).

		Returns:
			Zoomed-in image as a NumPy array with the same shape as the input.
		"""
		# if no zoom, return the original image
		if zoom_factor == 1:
			return img.copy()

		# else, extract dimensions of the original image
		height, width = img.shape[:2]

		# convert to PIL image
		pil_img = Image.fromarray(img)

		# calculate the new size for zooming
		new_width = int(width * zoom_factor)
		new_height = int(height * zoom_factor)

		# resize the image (zoom)
		zoomed_img = pil_img.resize((new_width, new_height), resample=Image.BICUBIC)

		# crop the center back to the original size
		left = (new_width - width) // 2
		top = (new_height - height) // 2
		right = left + width
		bottom = top + height
		zoomed_img = zoomed_img.crop((left, top, right, bottom))

		# convert back to NumPy array
		zoomed_array = np.array(zoomed_img)

		# return the zoomed image
		return zoomed_array

	def enhance_and_invert_text(self, image: np.ndarray) -> np.ndarray:
		"""
		Returns an image with white letters on black background,
		using EasyOCR to detect words and render letters proportionally
		to exactly fit the original bounding box in width and height,
		centered vertically to prevent overlapping lines.
		"""
		# convert to grayscale and enhance contrast
		if image.ndim == 3:
			gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		else:
			gray = image.copy()
		gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8) if gray.max() <= 1.0 else gray.astype(np.uint8)

		bg = cv2.medianBlur(gray, 31)
		contrast = cv2.divide(gray, bg + 1, scale=128)
		contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

		h, w = contrast.shape[:2]

		# detect words using EasyOCR
		results = self.reader.readtext(contrast, detail=1)

		# if no words detected, return the contrast-enhanced image
		if not results:
			return contrast

		# else, initialize a blank canvas (black background)
		canvas = Image.new("L", (w, h), 0)
		draw = ImageDraw.Draw(canvas)

		# load font
		font_path = os.path.join(os.path.dirname(__file__), "fonts", "SourceSans3-Regular.ttf")

		# iterate over the words and render them on the canvas
		for bbox, text, _ in results:

			# if not text, skip
			if not text:
				continue

			# get original word bounding box
			x_min = int(np.min([p[0] for p in bbox]))
			x_max = int(np.max([p[0] for p in bbox]))
			y_min = int(np.min([p[1] for p in bbox]))
			y_max = int(np.max([p[1] for p in bbox]))
			word_width = x_max - x_min
			word_height = y_max - y_min
			if word_width <= 0 or word_height <= 0:
				continue

			# start with a reasonable font size
			font_size = max(5, int(word_height * 1.2))
			font = ImageFont.truetype(font_path, font_size)

			# measure text size
			bbox_text = draw.textbbox((0, 0), text, font=font)
			text_w, text_h = bbox_text[2] - bbox_text[0], bbox_text[3] - bbox_text[1]

			# scale text to fit the bounding box
			scale_w = word_width / text_w if text_w > 0 else 1.0
			scale_h = word_height / text_h if text_h > 0 else 1.0
			scale = min(scale_w, scale_h)
			font = ImageFont.truetype(font_path, max(5, int(font_size * scale)))

			# recompute text height after scaling
			bbox_text = draw.textbbox((0, 0), text, font=font)
			text_h = bbox_text[3] - bbox_text[1]

			# center vertically inside the original bounding box
			y_centered = y_min + (word_height - text_h) // 2

			# draw text
			draw.text((x_min, y_centered), text, font=font, fill=255)

		# return the new canvas
		return np.array(canvas, dtype=np.uint8)

	def overlay_prosthetic_image_on_scotoma(self, scotoma_image: np.ndarray, prosthetic_image: np.ndarray,
											circular_mask: np.ndarray) -> np.ndarray:
		"""
		Overlay a square prosthetic image onto a 3-channel scotoma image using a circular mask.
		The prosthetic image is assumed to correspond to the circular central field.

		Args:
			scotoma_image: Original 3-channel image (H, W, 3)
			prosthetic_image: Grayscale square image to overlay (Hc, Wc)
			circular_mask: Boolean mask (Hc, Wc) with True inside the circle

		Returns:
			result: Scotoma image with the circular region replaced by gamma_image
		"""
		# initialize the output image
		result = scotoma_image.copy()

		# extract dimensions of the original image
		h_img, w_img = scotoma_image.shape[:2]

		# obtain the center coordinates of the original image
		center_y, center_x = h_img // 2, w_img // 2

		# extract dimensions of the prosthetic image
		h_pros, w_pros = prosthetic_image.shape[:2]

		# compute the top-left corner to overlay the prosthetic image centered
		start_y = center_y - h_pros // 2
		start_x = center_x - w_pros // 2
		end_y = start_y + h_pros
		end_x = start_x + w_pros

		# compute slices, handling image boundaries
		img_slice_y_start = max(start_y, 0)
		img_slice_x_start = max(start_x, 0)
		img_slice_y_end = min(end_y, h_img)
		img_slice_x_end = min(end_x, w_img)

		pros_slice_y_start = img_slice_y_start - start_y
		pros_slice_x_start = img_slice_x_start - start_x
		pros_slice_y_end = pros_slice_y_start + (img_slice_y_end - img_slice_y_start)
		pros_slice_x_end = pros_slice_x_start + (img_slice_x_end - img_slice_x_start)

		# overlay the prosthetic image using a circular mask
		mask_cropped = circular_mask[pros_slice_y_start:pros_slice_y_end,
		pros_slice_x_start:pros_slice_x_end]

		for c in range(3):
			result[img_slice_y_start:img_slice_y_end,
			img_slice_x_start:img_slice_x_end, c][mask_cropped] = prosthetic_image[pros_slice_y_start:pros_slice_y_end,
			pros_slice_x_start:pros_slice_x_end][mask_cropped]

		# return the composite image
		return result

	def simulate_vision(self, image: np.ndarray) -> np.ndarray:
		# apply gradual blur as a function of eccentricity
		print("Applying eccentricity blur...")
		blurred_image = self.apply_eccentricity_blur(image)

		# show blurred image
		Image.fromarray(blurred_image).show()

		# create a scotoma at the center of the visual field (AMD vision)
		print("Creating central scotoma...")
		scotoma_image, scotoma_mask = self.create_scotoma(blurred_image)

		# show image with scotoma
		Image.fromarray(scotoma_image).show()

		# extract central field image to process with ProViSim
		print("Extracting central field image...")
		center_image, center_mask = self.extract_central_field_image(image, field_radius_deg=self.implant_radius_deg)

		# show extracted central field image
		Image.fromarray(center_image).show()

		# zoom in on the central field, if requested
		zoomed_in_image = self.zoom_in_image(center_image, zoom_factor=self.zoom_factor)

		# show zoomed-in central field image
		Image.fromarray(zoomed_in_image).show()

		# if the central image contains text, enhance contrast and invert
		enhanced_image = self.enhance_and_invert_text(zoomed_in_image)

		# show enhanced image
		Image.fromarray(enhanced_image).show()

		# process this image with our ProViSim tools
		print("Processing central field image with ProViSim...")

		# convert to grayscale
		gray_img = rgb_to_gray(image_path=None, img=enhanced_image)

		# show the grayscale image
		Image.fromarray(gray_img * 255).show()

		# create a Tukey filter to reduce the resolution of the image, and apply it
		tukey_win = create_tukey_window(self.cutoff_freq, gray_img.shape[0], self.tukey_alpha, self.tukey_radius_expansion_factor)
		spatially_filtered_img = fft_filter(gray_img, tukey_win)

		# show the filtered image
		Image.fromarray(spatially_filtered_img * 255).show()

		# reduce the contrast of the filtered image using a gamma or sigmoid function
		transformed_float_img = spatially_filtered_img.copy()
		if self.apply_contrast_reduction:
			validate_gray_img(spatially_filtered_img)
			transformed_float_img = gamma_transform(spatially_filtered_img, self.gamma)
			# transformed_float_img = sigmoid_transform(spatially_filtered_img, gain=20, x_shift=0.2)

			# show the final transformed image
			Image.fromarray(transformed_float_img * 255).show(title="Transformed Image")

		# overlay the transformed image on the scotoma image
		final_image = self.overlay_prosthetic_image_on_scotoma(scotoma_image, transformed_float_img * 255, center_mask)

		# show the final image
		Image.fromarray(final_image).show(title="Final Image")

		# return the final image
		return final_image


def main():

	# define run parameters
	params = {
		'input_image': 'sample_input_images/4.jpeg',
		'distance_cm': 46,
		'image_width_cm': 20,
		'image_height_cm': 15,
		'sigma_center': 1.0,
		'eccentricity_k': 1.2,
		'scotoma_radius': 6.0,
		'implant_radius': 4.0,
		'cutoff_freq': 10, # 10 for 100um pixels; 50 for 20um pixels
		'tukey_alpha': 0.3,
		'tukey_radius_expansion_factor': 0.3,
		'apply_contrast_reduction': False,
		'gamma': 3.5,
		'zoom_factor': 3.2,
	}

	# define output image filename
	params['output_image'] = os.path.join(os.path.dirname(__file__), 'output_' + os.path.basename(params['input_image']))

	# validate input file
	if not os.path.exists(params['input_image']):
		print(f"Error: Input image '{params['input_image']}' not found")
		return

	# create a simulator with the given parameters
	simulator = VisionSimulator(
		input_image=params['input_image'],
		output_image=params['output_image'],
		viewing_distance_cm=params['distance_cm'],
		image_width_cm=params['image_width_cm'],
		image_height_cm=params['image_height_cm'],
		sigma_center=params['sigma_center'],
		eccentricity_k=params['eccentricity_k'],
		scotoma_radius_deg=params['scotoma_radius'],
		implant_radius_deg=params['implant_radius'],
		cutoff_freq=params['cutoff_freq'],
		tukey_alpha=params['tukey_alpha'],
		tukey_radius_expansion_factor=params['tukey_radius_expansion_factor'],
		gamma=params['gamma'],
		zoom_factor=params['zoom_factor'],
		apply_contrast_reduction=params['apply_contrast_reduction']

	)

	try:
		# load the given input image
		print(f"Loading image: {params['input_image']}")
		image = Image.open(params['input_image'])

		# convert image to ndarray for processing
		image = np.array(image)

		# show the loaded image
		Image.fromarray(image).show()

		# process the image
		result_image = simulator.simulate_vision(image)

		# save the result
		output_img = Image.fromarray(result_image.astype(np.uint8))
		output_img.save(params['output_image'])
		print(f"Vision simulation completed! Output saved as: {params['output_image']}")

	except Exception as e:
		print(f"Error processing image: {e}")
		import traceback
		traceback.print_exc()


if __name__ == "__main__":
	main()