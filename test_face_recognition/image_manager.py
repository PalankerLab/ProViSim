import os
import re
import glob
from typing import List


class ImageManager:
	"""Manages the loading and organizing of face images for the trial."""

	def __init__(self):
		# use the built-in images folder from the project directory
		self.images_folder = os.path.join(os.path.dirname(__file__), 'images')

		# check if the images folder exists
		if not os.path.exists(self.images_folder):
			raise FileNotFoundError(f"Images folder not found: {self.images_folder}")

		# initialize data structures to hold images
		self.images_data = {}

		# process the images and store in the data structures
		self._load_images()

		# initialize a list of ID's
		self.all_people = list(self.images_data.keys())
		self.people_with_emotions = []

		# initialize emotions-related structures
		self.emotions_dict = {
			'NE': 'neutral',
			'HA': 'happy',
			'SA': 'sad',
			'AN': 'angry',
			'DI': 'disgusted',
			'SU': 'surprised',
			'CO': 'confused',
			'FE': 'fearful'
		}
		self.available_emotions = []
		self.images_data_emotions = {}

		# organize and validate the loaded data
		self._sort_image_data()


	def _load_images(self):
		"""Load and organize images by person, gender, orientation, and emotion."""
		# get a list of all .jpg files in the images folder
		image_files = glob.glob(os.path.join(self.images_folder, "*.jpg"))

		# check if any images were found
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

				# use the suffix and photo session number as the key and add image
				key = f'{session_num}_{suffix}'
				self.images_data[person_id]['images'][key] = {
					'path': img_path,
					'emotion': emotion,
					'orientation': orientation,
					'session_num': session_num,
					'suffix': suffix
				}

	def _sort_image_data(self):
		# iterate the people, and if someone has less than three images available, remove them
		for person_id, person_data in self.images_data.items():
			if len(person_data['images']) < 3:
				# remove the data
				del self.images_data[person_id]

				# remove the person from the list of people
				self.all_people.remove(person_id)

		# check that we have enough people to run the experiment
		if len(self.all_people) < 10:
			raise Exception('Not enough people in the dataset to run this experiment.')

		# out of the remaining people, find those with at least four different emotions
		for person in self.all_people:

			# initialize structures for this person
			emotions = set()
			emotion_images = {}

			# get the person's images
			person_images = self.get_person_images(person)

			# find images that have emotions
			for key, img_data in person_images.items():
				if img_data['suffix'] in self.emotions_dict.keys():
					emotion = img_data['emotion']
					emotions.add(emotion)

					# if this emotion has not been added before for this person, initialize a list
					if emotion not in emotion_images:
						emotion_images[emotion] = []

					# add the image path to the list for this emotion
					emotion_images[emotion].append(img_data['path'])

			# if this person has at least four different emotions, add them to the list
			if len(emotions) >= 4:
				self.people_with_emotions.append(person)
				self.images_data_emotions[person] = emotion_images

		# if there are not enough people with emotions, raise an exception
		if len(self.people_with_emotions) < 1:
			raise Exception('Not enough people with emotions in the dataset to run this experiment.')

		# populate the list of all available emotions
		self.available_emotions = list({e for person in self.people_with_emotions for e in self.images_data_emotions[person]})

	@staticmethod
	def _parse_orientation(orientation_str: str) -> str:
		"""Extract orientation information from the orientation string."""
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
		# get all the images of the required person
		print(person_id)
		images = self.images_data.get(person_id, {'images': {}})['images']

		# filter out images with session_num == 2, if requested
		if exclude_session2:
			images = {k: v for k, v in images.items() if v.get('session_num') != 2}

		return images

	def get_person_emotion_images(self, person_id: str) -> dict:
		return self.images_data_emotions.get(person_id, {})

	def get_persons_by_gender(self, gender: str) -> List[str]:
		# get a list of person IDs by the requested gender
		return [pid for pid, data in self.images_data.items() if data['gender'] == gender]

	def get_all_persons(self) -> List[str]:
		# get a list of all person IDs
		return list(self.images_data.keys())

	def get_people_with_emotions(self) -> List[str]:
		return self.people_with_emotions

	def get_emotions(self) -> List[str]:
		return self.available_emotions

	def get_person_gender(self, person_id: str) -> str:
		# get the gender of the requested person
		return self.images_data.get(person_id, {}).get('gender', 'Unknown')

	def get_emotions_dict(self) -> dict:
		return self.emotions_dict
