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

		# process the images and store in the data structures
		self.all_images = {}
		self._load_all_images()

		# initialize a list of all people
		self.all_people = list(self.all_images.keys())

		# check that we have enough people to run the experiment
		if len(self.all_people) < 10:
			raise Exception('Not enough people in the dataset to run this experiment.')

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

		# with emotions
		self.emotion_images = {}
		self.people_with_emotions = []

		# without emotions
		self.emotion_free_images = {}
		self.people_without_emotions = []

		# initialize a dictionary of emotions and which people have them
		self.emotions_people_dict = {emotion:[] for emotion in self.emotions_dict.keys()}

		# organize and validate the loaded data
		self._sort_image_data()

	def _load_all_images(self):
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

			# parse filename: F0907_1100_NE.jpg -> person_id=MF0907, middle_part=1100, expression=NE
			match = re.match(r'([MF]\d+)_(\w+)_(\w+)\.jpg', filename)

			# if valid filename, extract components
			if match:

				# extract components from match
				person_id, session_num, suffix = match.groups()

				# determine gender
				gender = 'M' if person_id.startswith('M') else 'F'

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
				if person_id not in self.all_images:
					self.all_images[person_id] = {
						'gender': gender,
						'images': {}
					}

				# use the suffix and photo session number as the key and add image
				key = f'{session_num}_{suffix}'
				self.all_images[person_id]['images'][key] = {
					'path': img_path,
					'emotion': emotion,
					'orientation': orientation,
					'session_num': session_num,
					'suffix': suffix
				}

	def _sort_image_data(self):
		# sort emotions and orientations for each person
		for person in self.all_people:

			# initialize structures for this person
			emotions = set()
			non_emotions = {}
			emotion_images = {}
			non_emotion_images = {}

			# get the person's images
			person_images = self.get_all_person_images(person)

			# separate between emotions and orientations
			for key, img_data in person_images.items():

				# get session num
				session_num = img_data['session_num']

				# start session counter, if needed
				if session_num not in non_emotions:
					non_emotions[session_num] = 0

				# if the suffix is an emotion, add it to the list
				if img_data['suffix'] in self.emotions_dict.keys():
					emotion = img_data['emotion']
					emotions.add(emotion)

					# if this emotion has not been added before for this person, initialize a list
					if emotion not in emotion_images:
						emotion_images[emotion] = {}

					# add the key, if needed
					if key not in emotion_images[emotion]:
						emotion_images[emotion][key] = {}

					# add the image data
					emotion_images[emotion][key] = img_data

				else:
					# count non-emotion images
					non_emotions[session_num] += 1

					# add key, if needed
					if key not in non_emotion_images:
						non_emotion_images[key] = {}

					# add image data
					non_emotion_images[key] = img_data

			# if this person has at least four different emotions, add them to the list
			if len(emotions) >= 2:
				self.people_with_emotions.append(person)
				self.emotion_images[person] = {}

				# copy the original record
				copy_record = self.all_images[person].copy()
				self.emotion_images[person] = copy_record

				# replace the images with the emotion-filtered ones
				self.emotion_images[person]["images"] = emotion_images

				# add the person to the dictionary of people with this emotion
				for emotion in emotions:
					self.emotions_people_dict[emotion].append(person)

			# if there are at least 3 non-emotion images in each session, add images
			if all(count >= 3 for count in non_emotions.values()):
				self.people_without_emotions.append(person)
				self.emotion_free_images[person] = {}

				# copy the original record
				copy_record = self.all_images[person].copy()
				self.emotion_free_images[person] = copy_record

				# replace the images with the emotion-filtered ones
				self.emotion_free_images[person]["images"] = non_emotion_images

		# if there are not enough people with emotions, raise an exception
		if len(self.people_with_emotions) < 1:
			raise Exception('Not enough people with emotions in the dataset to run this experiment.')

		# populate the list of all available emotions
		self.available_emotions = list({e for person in self.people_with_emotions for e in self.emotion_images[
			person]["images"].keys()})

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

	def get_all_person_images(self, person_id: str, exclude_session2: bool = False):
		# get all the images of the required person
		images = self.all_images.get(person_id, {'images': {}})['images']

		# filter out images with session_num == 2, if requested
		if exclude_session2:
			images = {k: v for k, v in images.items() if v.get('session_num') != 2}

		return images

	def get_person_emotion_images(self, person_id: str, exclude_session2: bool = False) -> dict:
		images = self.emotion_images.get(person_id, {'images': {}})['images']

		# filter out images with session_num == 2, if requested
		if exclude_session2:
			images = {k: v for k, v in images.items() if v.get('session_num') != 2}

		return images

	def get_person_emotion_free_images(self, person_id: str, exclude_session2: bool = False) -> dict:
		images = self.emotion_free_images.get(person_id, {'images': {}})['images']

		# filter out images with session_num == 2, if requested
		if exclude_session2:
			images = {k: v for k, v in images.items() if v.get('session_num') != 2}

		return images

	def get_persons_by_gender(self, gender: str) -> List[str]:
		# get a list of person IDs by the requested gender
		return [pid for pid, data in self.all_images.items() if data['gender'] == gender]

	def get_all_people(self) -> List[str]:
		# get a list of all person IDs
		return list(self.all_images.keys())

	def get_all_people_with_emotions(self) -> List[str]:
		return sorted(self.people_with_emotions)

	def get_all_people_without_emotions(self) -> List[str]:
		return sorted(self.people_without_emotions)

	def get_emotions(self) -> List[str]:
		return sorted(self.available_emotions)

	def get_person_gender(self, person_id: str) -> str:
		# get the gender of the requested person
		return self.all_images.get(person_id, {}).get('gender', 'Unknown')

	def get_emotions_dict(self) -> dict:
		return self.emotions_dict

	def get_people_with_specific_emotion(self, emotion: str) -> list[str]:
		return sorted(self.emotions_people_dict.get(emotion, []))

