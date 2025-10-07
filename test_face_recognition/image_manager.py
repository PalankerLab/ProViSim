import glob
import os
import re
from typing import Dict, List


class ImageManager:
	"""Manages loading and organizing face images for the trial."""

	def __init__(self):
		# use the built-in images folder from the project directory
		self.images_folder = os.path.join(os.path.dirname(__file__), 'images')

		# check if the images folder exists
		if not os.path.exists(self.images_folder):
			raise FileNotFoundError(f"Images folder not found: {self.images_folder}")

		# initialize data structures to hold images
		self.images_data = {}
		self._load_images()

	def _load_images(self):
		"""Load and organize images by person, gender, orientation, and emotion."""
		# get a list of all .jpg files in the images folder
		image_files = glob.glob(os.path.join(self.images_folder, "*.jpg"))

		#check if any images were found
		if not image_files:
			raise FileNotFoundError(f"No .jpg images found in: {self.images_folder}")

		# process each image file
		for img_path in image_files:

			# get file name
			filename = os.path.basename(img_path)

			# parse filename: MF0907_1100_NE.jpg -> person_id=MF0907, middle_part=1100, expression=NE
			match = re.match(r'(M[MF]\d+)_(\w+)_(\w+)\.jpg', filename)

			# if valid filename, extract components
			if match:

				# extract components from match
				person_id, session_num, suffix = match.groups()

				# determine gender
				gender = 'M' if person_id.startswith('MM') else 'F'

				# extract session number
				session_num = int(session_num[0])

				# determine if suffix is emotion (pure letters) or orientation (contains R/L/F with numbers)
				if re.match(r'^[A-Z]+$', suffix) and not re.search(r'[RLF]\d*$|^\d+[RLF]$', suffix):
					# pure letters that are NOT orientation patterns = emotion
					emotion = suffix
					orientation = '30L'
				else:
					# contains R/L/F patterns or numbers = orientation
					emotion = 'NE'
					orientation = self._parse_orientation(suffix)

				# skip orientations that cannot be landmarked
				if orientation in ['right_90', 'right_60', 'left_90', 'left_60']:
					continue

				# initialize a new person, if needed
				if person_id not in self.images_data:
					self.images_data[person_id] = {
						'gender': gender,
						'images': {}
					}

				# use just the suffix as the key and add image
				key = f'{session_num}_{suffix}'
				self.images_data[person_id]['images'][key] = {
					'path': img_path,
					'emotion': emotion,
					'orientation': orientation,
					'session_num': session_num,
					'suffix': suffix
				}

	@staticmethod
	def _parse_orientation(orientation_str: str) -> str:
		"""Extract orientation information from orientation string."""
		# look for orientation patterns like 00F, 30L, 60R, 90L, etc.
		if orientation_str == '00F' or 'F' in orientation_str:
			return 'front'

		# extract angle for left orientations
		elif 'L' in orientation_str:
			angle_match = re.search(r'(\d+)L', orientation_str)
			if angle_match:
				angle = int(angle_match.group(1))
				if angle == 30:
					return 'left_30'
				elif angle == 45:
					return 'left_45'
				elif angle == 60:
					return 'left_60'
				elif angle == 90:
					return 'left_90'
				else:
					return f'left_{angle}'
			return 'left'

		# extract angle for right orientations
		elif 'R' in orientation_str:
			angle_match = re.search(r'(\d+)R', orientation_str)
			if angle_match:
				angle = int(angle_match.group(1))
				if angle == 30:
					return 'right_30'
				elif angle == 45:
					return 'right_45'
				elif angle == 60:
					return 'right_60'
				elif angle == 90:
					return 'right_90'
				else:
					return f'right_{angle}'
			return 'right'
		else:
			return 'other'

	def get_person_images(self, person_id: str, exclude_session2: bool = False):
		"""Get all images for a specific person.
		"""
		images = self.images_data.get(person_id, {'images': {}})['images']

		# filter out images with session_num == 2
		if exclude_session2:
			images = {k: v for k, v in images.items() if v.get('session_num') != 2}

		return images

	def get_persons_by_gender(self, gender: str) -> List[str]:
		"""Get list of person IDs by gender."""
		return [pid for pid, data in self.images_data.items() if data['gender'] == gender]

	def get_all_persons(self) -> List[str]:
		"""Get list of all person IDs."""
		return list(self.images_data.keys())

	def get_person_gender(self, person_id: str) -> str:
		"""Get gender of a specific person."""
		return self.images_data.get(person_id, {}).get('gender', 'Unknown')
