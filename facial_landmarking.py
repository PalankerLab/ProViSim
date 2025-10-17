"""
facial_landmarking.py

This module contains functions to find and draw facial landmarks.
"""

# general imports
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

# landmarks model imports
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_IRIS, FACEMESH_RIGHT_IRIS

# project imports
import utils

FACEMESH_RIGHT_EYEBROW_BOTTOM = frozenset([(46, 53), (53, 52), (52, 65), (65, 55)])
FACEMESH_RIGHT_EYEBROW_TOP = frozenset([(70, 63), (63, 105),(105, 66), (66, 107)])
FACEMESH_LEFT_EYEBROW_BOTTOM = frozenset([(276, 283), (283, 282), (282, 295), (295, 285)])
FACEMESH_LEFT_EYEBROW_TOP = frozenset([(300, 293), (293, 334), (334, 296), (296, 336)])

LOWER_LIP_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIP_INDICES = [185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]


def normalized_to_pixel_coordinates(
	normalized_x: float, normalized_y: float, image_width: int,
	image_height: int):
	"""Converts normalized value pair to pixel coordinates."""

	# Checks if the float value is between 0 and 1.
	def _is_valid_normalized_value(value: float) -> bool:
		return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

	if not (_is_valid_normalized_value(normalized_x) and _is_valid_normalized_value(normalized_y)):
		raise ValueError("The values are not normalized between 0 and 1.")

	x_px = min(math.floor(normalized_x * image_width), image_width - 1)
	y_px = min(math.floor(normalized_y * image_height), image_height - 1)

	return x_px, y_px


def extract_indices_from_connections(landmark_connections):
	"""
	Extract unique landmark indices from a list of landmark connections.

	Parameters:
	landmark_connections (list of tuple): A list of connections, where each connection is represented by a tuple of two indices.

	Returns:
	indices: A set of unique indices derived from the connections.
	"""
	# Initialize an empty set to store unique indices
	indices = set()

	# Iterate through each connection and add both indices to the set
	for connection in landmark_connections:
		indices.update(connection)

	return indices

def get_actual_coords(frozen_set, idx_to_coordinates):
	"""
	Extract the coordinates of landmark connections from a frozen set.

	Parameters:
	frozen_set (frozenset of tuple): A frozenset of connections, where each connection is a tuple of two indices.

	Returns:
	list of tuple: A sorted list of coordinates corresponding to the indices in the connections.
	"""
	# Extract coordinates for each connection
	coords_list = [(idx_to_coordinates[start_idx], idx_to_coordinates[end_idx]) for start_idx, end_idx in frozen_set]

	# Sort the list of coordinates by the x-coordinate of the start point
	sorted_coords = sorted(coords_list, key=lambda x: x[0][0])

	return sorted_coords


def get_midline_eyebrows(frozensets, idx_to_coordinates):
	"""
	Calculate the midline coordinates of two sets of eyebrow connections (bottom and top).

	Parameters:
	frozensets(tuple of frozenset of tuple): A tuple of frozenset of connections for top and bottom eyebrow lines.

	Returns:
	list of tuple: A list of midline coordinates for an eyebrow.
	"""
	frozenset1, frozenset2 = frozensets

	# Get coordinates for both eyebrow sets
	coords1 = get_actual_coords(frozenset1, idx_to_coordinates)
	coords2 = get_actual_coords(frozenset2, idx_to_coordinates)

	# Calculate midline coordinates for each corresponding pair of connections
	midline_coords = []
	for line1, line2 in zip(coords1, coords2):
		(s1_x, s1_y), (e1_x, e1_y) = line1
		(s2_x, s2_y), (e2_x, e2_y) = line2

		mid_sx = (s1_x + s2_x) // 2
		mid_sy = (s1_y + s2_y) // 2
		mid_ex = (e1_x + e2_x) // 2
		mid_ey = (e1_y + e2_y) // 2

		midline_coords.append(((mid_sx, mid_sy), (mid_ex, mid_ey)))

	return midline_coords


def get_center_iris(iris_frozen_set, idx_to_coordinates):
	"""
	Calculate the center of the iris based on the coordinates in a frozen set.

	Parameters:
	iris_frozen_set (frozenset of tuple): A frozenset of connections representing the iris.

	Returns:
	tuple: The (x, y) coordinates of the center of the iris.
	"""
	# Get coordinates of the iris connections
	pairs = get_actual_coords(iris_frozen_set, idx_to_coordinates)

	# Extract the first points from each pair
	first_points = np.array([pair[0] for pair in pairs])

	# Calculate the mean of the first points
	center_of_iris = np.mean(first_points, axis=0)

	return center_of_iris.astype(np.uint8)


def compute_line_thickness(img, cycles_per_img=10, ratio=1):
	"""
	Computes the line thickness to draw the facial features.

	Parameters:
		img (np.ndarray): The image of the face to compute the line thickness for.
		cycles_per_img (int): Number of cycles that can be represented by the implant.
		ratio (float): Scaling factor applied to the calculated minimum line thickness.

	Returns:
		int: The calculated line thickness, adjusted by the specified ratio.
	"""
	# how many image pixels compose each implant pixel
	image_pixels_per_implant_pixel = img.shape[0] / (2 * cycles_per_img)

	# set minimum the line thickness as 0.8 implant pixels
	minimum_line_thickness = 0.8 * image_pixels_per_implant_pixel

	# apply ratio and round to integer
	line_thickness = round(minimum_line_thickness * ratio)

	# ensure at least 1 pixel thickness
	return max(line_thickness, 1)


def create_lip_mask(face_image, lip_indices):
	"""
	Generates a binary mask for the lips in a given face image.

	Parameters:
		face_image (np.ndarray): Input image containing a face.
		lip_indices (list of int): Indices of the landmarks corresponding to the lips.

	Returns:
		np.ndarray: Binary mask with the lips region filled with white (255),
					and the rest of the image in black (0).
	"""
	# Initialize MediaPipe FaceMesh model
	mp_face_mesh = mp.solutions.face_mesh
	face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

	# Process the image to get face landmarks
	results = face_mesh.process(face_image)

	if results.multi_face_landmarks:
		for face_landmarks in results.multi_face_landmarks:
			# Extract the 2D coordinates of the lip landmarks
			lip_points = [
				[int(face_landmarks.landmark[idx].x * face_image.shape[1]),
				 int(face_landmarks.landmark[idx].y * face_image.shape[0])]
				for idx in lip_indices
			]

			# Convert lip points to a NumPy array suitable for polygon filling
			lip_points = np.array([lip_points], dtype=np.int32)

			# Create a black mask the same size as the input image
			lip_mask = np.zeros_like(face_image)

			# Fill the polygon defined by lip landmarks with white color (255)
			cv2.fillPoly(lip_mask, lip_points, (255, 255, 255))

			return lip_mask

	# Return an empty mask if no landmarks are detected
	return np.zeros_like(face_image)


def safe_color_at(img, x, y):
	"""
	Sample color at (x,y) from an RGB image safely.
	Returns (R,G,B) tuple.
	"""
	h, w, _ = img.shape
	x = int(np.clip(x, 0, w - 1))
	y = int(np.clip(y, 0, h - 1))
	return tuple(int(c) for c in img[y, x])


def draw_colored_lines(dst_img, src_img, line_segments, thickness=2, landmarking_color=None):
	"""
	Draws line segments where each line is colored with the average of
	its endpoints' colors from src_img (both must be RGB).

	Parameters:
		dst_img (np.ndarray): Image to draw onto (RGB).
		src_img (np.ndarray): Image to sample colors from (RGB).
		line_segments (list of ((x1,y1),(x2,y2))): Line endpoints.
		thickness (int): Line thickness.
	"""
	for (sx, sy), (ex, ey) in line_segments:
		color_start = safe_color_at(src_img, sx, sy)
		color_end = safe_color_at(src_img, ex, ey)
		color_rgb = tuple((s + e) // 2 for s, e in zip(color_start, color_end)) if landmarking_color is None else landmarking_color
		cv2.line(dst_img, (sx, sy), (ex, ey), color=color_rgb, thickness=thickness)


def draw_colored_circles(dst_img, src_img, centers, radius=2, landmarking_color=None):
	"""
	Draws filled circles at given centers, each colored with the
	sampled pixel color from src_img.

	Parameters:
		dst_img (np.ndarray): Image to draw onto (RGB).
		src_img (np.ndarray): Image to sample colors from (RGB).
		centers (list of (x,y)): Circle centers.
		radius (int): Circle radius.
	"""
	for cx, cy in centers:
		color_rgb = safe_color_at(src_img, cx, cy) if landmarking_color is None else landmarking_color
		cv2.circle(dst_img, (int(cx), int(cy)), radius=radius, color=color_rgb, thickness=-1)


def fill_region_with_color(dst_img, src_img, mask, landmarking_color=None):
	"""
	Fills a region defined by a binary mask with the average color
	from that region sampled from src_img.

	Parameters:
		dst_img (np.ndarray): Image to fill (RGB).
		src_img (np.ndarray): Image to sample colors from (RGB).
		mask (np.ndarray): Boolean mask (H,W) or (H,W,3) ==255.
	"""
	if mask.ndim == 3:
		mask = np.all(mask == 255, axis=-1)
	if not np.any(mask):
		return

	# get corresponding region pixels
	region_pixels = src_img[mask]

	# determine landmarking color
	if landmarking_color is None:
		landmarking_color = np.round(region_pixels.mean(axis=0)).astype(np.uint8)

	# color destination image
	dst_img[mask] = tuple(int(c) for c in landmarking_color)

def draw_facial_landmarks_colored(
	image,
	detector,
	draw_eyebrows=True,
	draw_irises=True,
	draw_lips=True,
	cutoff_frequency=10,
	show=True,
	landmarking_color=True
):
	"""
	Detects face landmarks and draws eyebrows, irises, and lips in their
	original colors on a copy of the input image.

	Parameters:
		image (str or ndarray): Path to the image file or a numpy array representing the image.
		detector (mp.tasks.python.vision.FaceLandmarker): Initialized face landmarker.
		draw_eyebrows (bool): Whether to draw eyebrows.
		draw_irises (bool): Whether to draw irises.
		draw_lips (bool): Whether to draw lips.
		cutoff_frequency (int): Cycles per image for thickness computation.
		show (bool): Whether to show the image via matplotlib.
		landmarking_color (bool): Whether to use the original color of the landmarks. If not, will be colored black.

	Returns:
		np.ndarray: The RGB image with landmarks drawn.
	"""
	# ensure the image is square to square and load as mediapipe Image
	image = utils.crop_to_square(image)
	image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

	# run landmark detection on the image
	detection_result = detector.detect(image)

	# if no landmarks detected, return the original image
	if not detection_result.face_landmarks:
		return None

	face_landmarks = detection_result.face_landmarks[0]

	# wrap into a protobuf list
	face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
	face_landmarks_proto.landmark.extend([
		landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in face_landmarks
	])

	# convert to numpy (ensure RGB)
	np_image = np.asarray(image.numpy_view())

	# drop alpha if present
	if np_image.shape[-1] == 4:
		np_image = np_image[..., :-1]

	# get the dimensions of the image
	face_image = np.copy(np_image)
	h, w, _ = face_image.shape

	# map indices to pixel coordinates
	idx_to_coordinates = {
		idx: normalized_to_pixel_coordinates(lm.x, lm.y, w, h)
		for idx, lm in enumerate(face_landmarks_proto.landmark)
	}

	# get the midline of the eyebrows and the irises
	midline_right_eyebrow = get_midline_eyebrows(
		(FACEMESH_RIGHT_EYEBROW_BOTTOM, FACEMESH_RIGHT_EYEBROW_TOP),
		idx_to_coordinates
	)
	midline_left_eyebrow = get_midline_eyebrows(
		(FACEMESH_LEFT_EYEBROW_BOTTOM, FACEMESH_LEFT_EYEBROW_TOP),
		idx_to_coordinates
	)
	midline_eyebrows = midline_right_eyebrow + midline_left_eyebrow

	# get the center of the irises
	center_left_iris = get_center_iris(FACEMESH_LEFT_IRIS, idx_to_coordinates)
	center_right_iris = get_center_iris(FACEMESH_RIGHT_IRIS, idx_to_coordinates)

	# get the mask of the lips
	lip_mask = create_lip_mask(face_image, LOWER_LIP_INDICES + UPPER_LIP_INDICES)

	# compute the thickness for features
	line_thickness = compute_line_thickness(face_image, cycles_per_img=cutoff_frequency)

	# draw the features
	if draw_eyebrows:
		draw_colored_lines(face_image, np_image, midline_eyebrows, thickness=line_thickness, landmarking_color=landmarking_color)

	if draw_irises:
		draw_colored_circles(face_image, np_image,
							 [center_left_iris, center_right_iris],
							 radius=line_thickness, landmarking_color=landmarking_color)

	if draw_lips:
		fill_region_with_color(face_image, np_image, lip_mask, landmarking_color=landmarking_color)

	# show the result if requested
	if show:
		plt.title('Thickened facial features overlaid')
		plt.imshow(face_image)
		plt.axis('off')
		plt.show()

	return face_image
