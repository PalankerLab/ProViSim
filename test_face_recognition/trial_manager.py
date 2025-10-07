import random
import re
from typing import Dict, Optional

import pandas as pd

from test_face_recognition.image_manager import ImageManager


class TrialManager:
	"""Manages the clinical trial scenarios and data collection."""

	def __init__(self, image_manager: ImageManager):
		# initialize variables
		self.results = {
			'phase1': {'odd_person_out': [], 'diff_gender': [], 'emotion': []},
			'phase2': {'odd_person_out': [], 'diff_gender': [], 'emotion': []}
		}
		self.trial_sequence = []
		self.random_seed = 713

		# save image manager
		self.image_manager = image_manager

		# get all persons and initialize usage counts
		self.persons = self.image_manager.get_all_persons()
		self.person_usage_counts_odd_one_out = {pid: 0 for pid in self.persons}
		self.emotion_usage_counts = {}

		# get all persons by gender and initialize usage counts
		self.male_persons = self.image_manager.get_persons_by_gender('M')
		self.female_persons = self.image_manager.get_persons_by_gender('F')
		self.person_usage_counts_diff_gender = {pid: 0 for pid in self.persons}

		# number of trials per scenario
		self.num_trials_per_scenario = 26
		self.current_trial = 0

		# number of different scenarios
		self.num_scenarios = 3
		self.current_scenario = None
		self.scenario_names = ['odd_person_out', 'diff_gender', 'emotion']

		# number of phases (without and with landmarks)
		self.num_phases = 2
		self.current_phase = 1

		# setup trial order
		self.setup_trials()

	def setup_trials(self):
		"""Setup randomized trial order with fixed seed for consistency."""
		# fixed seed for consistency across subjects
		random.seed(self.random_seed)

		# create the requested number of trials for each scenario
		trials = []
		for scenario in range(0, self.num_scenarios):
			for trial_num in range(self.num_trials_per_scenario):
				trials.append(scenario)

		# shuffle the order
		random.shuffle(trials)

		# duplicate for all phases
		self.trial_sequence = trials * self.num_phases

	# def generate_odd_person_out_trial(self) -> Dict:
	# 	"""Scenario 1: Find the different person (3 same + 1 different)."""
	# 	# get all people available in the dataset
	# 	persons = self.image_manager.get_all_persons()
	#
	# 	# if there are less than 2 people, we cannot generate this scenario
	# 	if len(persons) < 2:
	# 		return None
	#
	# 	# select two different persons
	# 	selected_persons = random.sample(persons, 2)
	# 	same_person = selected_persons[0]
	# 	diff_person = selected_persons[1]
	#
	# 	# get the images for the same person
	# 	same_images = self.image_manager.get_person_images(same_person)
	# 	if len(same_images) < 3:
	# 		return None
	#
	# 	# get the images for the different person
	# 	diff_images = self.image_manager.get_person_images(diff_person)
	# 	if len(diff_images) < 1:
	# 		return None
	#
	# 	# select three images from the same person and one from the different person
	# 	same_selected = random.sample(list(same_images.keys()), 3)
	# 	diff_selected = random.choice(list(diff_images.keys()))
	#
	# 	# create an image list with paths
	# 	images = []
	# 	for key in same_selected:
	# 		images.append(same_images[key]['path'])
	# 	images.append(diff_images[diff_selected]['path'])
	#
	# 	# randomize the positions in which the images are shown
	# 	positions = list(range(4))
	# 	random.shuffle(positions)
	#
	# 	# set the position of the correct answer
	# 	correct_answer = positions.index(3)
	#
	# 	# reorder images according to shuffled positions
	# 	reordered_images = [''] * 4
	# 	for i, pos in enumerate(positions):
	# 		reordered_images[i] = images[pos]
	#
	# 	return {
	# 		'type': 'different_person',
	# 		'images': reordered_images,
	# 		'correct_answer': correct_answer,
	# 		'question': 'Who person is the odd one out?'
	# 	}
	def generate_odd_person_out_trial(self):
		"""Scenario 1: Find the different person (3 same + 1 different)."""
		# if there are less than 2 people, we cannot generate this scenario
		if len(self.persons) < 2:
			return None

		# select two people with minimal usage to maximize spread
		min_count = min(self.person_usage_counts_odd_one_out.values())
		least_used = [pid for pid, count in self.person_usage_counts_odd_one_out.items() if count == min_count]

		# fallback: pick randomly among all; reseed before selecting people
		random.seed(self.random_seed)
		if len(least_used) < 2:
			same_person, diff_person = random.sample(self.persons, 2)
		else:
			same_person, diff_person = random.sample(least_used, 2)

		# increment people's usage counts
		self.person_usage_counts_odd_one_out[same_person] += 1
		self.person_usage_counts_odd_one_out[diff_person] += 1

		# get images
		same_images = self.image_manager.get_person_images(same_person)
		if len(same_images) < 3:
			return None

		diff_images = self.image_manager.get_person_images(diff_person)
		if len(diff_images) < 1:
			return None

		# reseed before selecting images
		random.seed(self.random_seed)
		same_selected = random.sample(list(same_images.keys()), 3)
		diff_selected = random.choice(list(diff_images.keys()))

		# create image list
		images = [same_images[key]['path'] for key in same_selected]
		images.append(diff_images[diff_selected]['path'])

		# reseed before shuffling positions
		random.seed(self.random_seed)
		positions = list(range(4))
		random.shuffle(positions)

		# correct answer position (last image is the different person)
		correct_answer = positions.index(3)

		# reorder images according to shuffled positions
		reordered_images = [''] * 4
		for i, pos in enumerate(positions):
			reordered_images[i] = images[pos]

		return {
			'type': 'different_person',
			'images': reordered_images,
			'correct_answer': correct_answer,
			'question': 'Which person is the odd one out?'
		}

	# def generate_different_gender_trial(self) -> Dict:
	# 	"""Scenario 2: Find different gender (3 same gender + 1 different)."""
	# 	male_persons = self.image_manager.get_persons_by_gender('M')
	# 	female_persons = self.image_manager.get_persons_by_gender('F')
	#
	# 	if len(male_persons) < 3 or len(female_persons) < 1:
	# 		return None
	#
	# 	# randomly choose which gender to use as majority
	# 	if random.choice([True, False]):
	# 		same_gender_persons = random.sample(male_persons, 3)
	# 		diff_gender_person = random.choice(female_persons)
	# 	else:
	# 		same_gender_persons = random.sample(female_persons, 3)
	# 		diff_gender_person = random.choice(male_persons)
	#
	# 	# get one image from each person
	# 	images = []
	# 	for person in same_gender_persons:
	# 		person_images = self.image_manager.get_person_images(person)
	# 		if person_images:
	# 			img_key = random.choice(list(person_images.keys()))
	# 			images.append(person_images[img_key]['path'])
	#
	# 	diff_images = self.image_manager.get_person_images(diff_gender_person)
	# 	if diff_images:
	# 		img_key = random.choice(list(diff_images.keys()))
	# 		images.append(diff_images[img_key]['path'])
	#
	# 	if len(images) != 4:
	# 		return None
	#
	# 	# randomize positions
	# 	positions = list(range(4))
	# 	random.shuffle(positions)
	# 	correct_answer = positions.index(3)
	#
	# 	reordered_images = [''] * 4
	# 	for i, pos in enumerate(positions):
	# 		reordered_images[i] = images[pos]
	#
	# 	return {
	# 		'type': 'gender',
	# 		'images': reordered_images,
	# 		'correct_answer': correct_answer,
	# 		'question': 'Which person is a different gender?'
	# 	}
	def generate_different_gender_trial(self):
		"""Scenario 2: Find different gender (3 same gender + 1 different)."""
		# we need at least 3 persons of each gender for this scenario
		if len(self.male_persons) < 3 or len(self.female_persons) < 3:
			return None

		# reseed for deterministic choice of majority gender
		random.seed(self.random_seed)

		# choose majority gender
		if random.choice([True, False]):
			# male majority - pick three males with minimal usage
			min_count = min([self.person_usage_counts_diff_gender[pid] for pid in self.male_persons])
			least_used_males = [pid for pid in self.male_persons if self.person_usage_counts_diff_gender[pid] == min_count]
			random.seed(self.random_seed)
			same_gender_persons = random.sample(least_used_males, 3)

			# pick one female with minimal usage
			min_count_f = min([self.person_usage_counts_diff_gender[pid] for pid in self.female_persons])
			least_used_females = [pid for pid in self.female_persons if self.person_usage_counts_diff_gender[pid] == min_count_f]
			random.seed(self.random_seed)
			diff_gender_person = random.choice(least_used_females)
		else:
			# female majority
			min_count = min([self.person_usage_counts_diff_gender[pid] for pid in self.female_persons])
			least_used_females = [pid for pid in self.female_persons if self.person_usage_counts_diff_gender[pid] == min_count]
			random.seed(self.random_seed)
			same_gender_persons = random.sample(least_used_females, 3)

			min_count_m = min([self.person_usage_counts_diff_gender[pid] for pid in self.male_persons])
			least_used_males = [pid for pid in self.male_persons if self.person_usage_counts_diff_gender[pid] == min_count_m]
			random.seed(self.random_seed)
			diff_gender_person = random.choice(least_used_males)

		# increment usage counts
		for pid in same_gender_persons:
			self.person_usage_counts_diff_gender[pid] += 1
		self.person_usage_counts_diff_gender[diff_gender_person] += 1

		# get one image from each person
		images = []
		for person in same_gender_persons:
			person_images = self.image_manager.get_person_images(person)
			if person_images:
				random.seed(self.random_seed)
				img_key = random.choice(list(person_images.keys()))
				images.append(person_images[img_key]['path'])

		diff_images = self.image_manager.get_person_images(diff_gender_person)
		if diff_images:
			random.seed(self.random_seed)
			img_key = random.choice(list(diff_images.keys()))
			images.append(diff_images[img_key]['path'])

		if len(images) != 4:
			return None

		# randomize positions
		positions = list(range(4))
		random.seed(self.random_seed)
		random.shuffle(positions)
		correct_answer = positions.index(3)

		reordered_images = [''] * 4
		for i, pos in enumerate(positions):
			reordered_images[i] = images[pos]

		return {
			'type': 'gender',
			'images': reordered_images,
			'correct_answer': correct_answer,
			'question': 'Which person is a different gender?'
		}

	# def generate_emotion_trial(self):
	# 	"""Scenario 3: Find specific emotion (4 emotions of same person)."""
	# 	# Find a person with multiple emotions (only use images with pure letter suffixes)
	# 	person_with_emotions = None
	# 	for person in self.persons:
	# 		person_images = self.image_manager.get_person_images(person)
	# 		emotions = set()
	# 		emotion_images = {}
	#
	# 		for key, img_data in person_images.items():
	# 			# Only use images that have pure letter emotions (not R/L/F orientations)
	# 			if re.match(r'^[A-Z]+$', img_data['suffix']) and not re.search(r'[RLF]', img_data['suffix']):
	# 				emotion = img_data['emotion']
	# 				emotions.add(emotion)
	# 				if emotion not in emotion_images:
	# 					emotion_images[emotion] = []
	# 				emotion_images[emotion].append(img_data['path'])
	#
	# 		if len(emotions) >= 4:  # Need at least 4 different emotions
	# 			person_with_emotions = person
	# 			person_emotion_images = emotion_images
	# 			break
	#
	# 	# If no person with multiple emotions is found, return None
	# 	if not person_with_emotions:
	# 		return None
	#
	# 	# select 4 different emotions
	# 	available_emotions = list(person_emotion_images.keys())
	# 	random.seed(self.random_seed)
	# 	selected_emotions = random.sample(available_emotions, 4)
	# 	images = []
	#
	# 	for emotion in selected_emotions:
	# 		random.seed(self.random_seed)
	# 		images.append(random.choice(person_emotion_images[emotion]))
	#
	# 	# draw a random emotion from the list of available emotions to ask about
	# 	random.seed(self.random_seed)
	# 	target_emotion = random.choice(selected_emotions)
	# 	correct_answer = selected_emotions.index(target_emotion)
	#
	# 	# map emotion codes
	# 	emotion_names = {
	# 		'NE': 'neutral',
	# 		'HA': 'happy',
	# 		'SA': 'sad',
	# 		'AN': 'angry',
	# 		'FE': 'fearful',
	# 		'DI': 'disgusted',
	# 		'SU': 'surprised',
	# 		'CO': 'confused'
	# 	}
	#
	# 	target_emotion_name = emotion_names.get(target_emotion, target_emotion.lower())
	#
	# 	# Randomize positions
	# 	positions = list(range(4))
	# 	random.seed(self.random_seed)
	# 	random.shuffle(positions)
	# 	correct_answer = positions.index(correct_answer)
	#
	# 	reordered_images = [''] * 4
	# 	for i, pos in enumerate(positions):
	# 		reordered_images[i] = images[pos]
	#
	# 	return {
	# 		'type': 'emotion',
	# 		'images': reordered_images,
	# 		'correct_answer': correct_answer,
	# 		'question': f'Which face looks {target_emotion_name}?'
	# 	}

	def generate_emotion_trial(self):
		"""Scenario 3: Find specific emotion (4 emotions of same person)."""
		# find all people with at least 4 distinct emotions
		eligible_persons = []
		person_emotion_data = {}

		for person in self.persons:
			person_images = self.image_manager.get_person_images(person)
			emotions = set()
			emotion_images = {}

			for key, img_data in person_images.items():
				# only use images that have pure letter emotions (not R/L/F orientations)
				if re.match(r'^[A-Z]+$', img_data['suffix']) and not re.search(r'[RLF]', img_data['suffix']):
					emotion = img_data['emotion']
					emotions.add(emotion)
					if emotion not in emotion_images:
						emotion_images[emotion] = []
					emotion_images[emotion].append(img_data['path'])

					# initialize emotion usage if not seen
					if emotion not in self.emotion_usage_counts:
						self.emotion_usage_counts[emotion] = 0

			if len(emotions) >= 4:  # Need at least 4 different emotions
				eligible_persons.append(person)
				person_emotion_data[person] = emotion_images

		# if no suitable person is found, return None
		if not eligible_persons:
			return None

		# pick a person with multiple emotions
		random.seed(self.random_seed)
		person_with_emotions = random.choice(eligible_persons)
		person_emotion_images = person_emotion_data[person_with_emotions]

		# select 4 different emotions prioritizing least-used ones
		available_emotions = list(person_emotion_images.keys())
		random.seed(self.random_seed)
		available_emotions.sort(key=lambda e: self.emotion_usage_counts.get(e, 0))
		selected_emotions = available_emotions[:4] if len(available_emotions) >= 4 else available_emotions

		images = []
		for emotion in selected_emotions:
			random.seed(self.random_seed)
			images.append(random.choice(person_emotion_images[emotion]))
			# increment emotion usage
			self.emotion_usage_counts[emotion] += 1

		# draw a random emotion from the selected set as the target
		random.seed(self.random_seed)
		target_emotion = random.choice(selected_emotions)
		correct_answer = selected_emotions.index(target_emotion)

		# map emotion codes
		emotion_names = {
			'NE': 'neutral',
			'HA': 'happy',
			'SA': 'sad',
			'AN': 'angry',
			'FE': 'fearful',
			'DI': 'disgusted',
			'SU': 'surprised',
			'CO': 'confused'
		}

		target_emotion_name = emotion_names.get(target_emotion, target_emotion.lower())

		# R=randomize positions
		positions = list(range(4))
		random.seed(self.random_seed)
		random.shuffle(positions)
		correct_answer = positions.index(correct_answer)

		reordered_images = [''] * 4
		for i, pos in enumerate(positions):
			reordered_images[i] = images[pos]

		return {
			'type': 'emotion',
			'images': reordered_images,
			'correct_answer': correct_answer,
			'question': f'Which face looks {target_emotion_name}?'
		}

	def get_next_trial(self) -> Optional[Dict]:
		"""Get the next trial configuration."""
		if self.current_trial >= len(self.trial_sequence):
			return None

		scenario_num = self.trial_sequence[self.current_trial]

		# generate the next trial based on the scenario
		if scenario_num == 0:
			trial_config = self.generate_odd_person_out_trial()
		elif scenario_num == 1:
			trial_config = self.generate_different_gender_trial()
		elif scenario_num == 2:
			trial_config = self.generate_emotion_trial()
		else:
			return None

		if trial_config:
			trial_config['scenario'] = scenario_num
			trial_config['trial_number'] = self.current_trial + 1
			trial_config['phase'] = self.current_phase

		return trial_config

	def record_answer(self, trial_config: Dict, selected_answer: int, response_time: float,
					 correct: bool, timeout: bool = False):
		"""Record the subject's answer."""
		phase_key = f'phase{self.current_phase}'
		scenario_key = self.scenario_names[trial_config["scenario"]]

		result = {
			'trial_number': trial_config['trial_number'],
			'scenario': trial_config['scenario'],
			'scenario_type': trial_config['type'],
			'question': trial_config['question'],
			'correct_answer': trial_config['correct_answer'],
			'selected_answer': selected_answer,
			'correct': correct,
			'response_time': response_time,
			'timeout': timeout,
			'phase': self.current_phase
		}

		self.results[phase_key][scenario_key].append(result)

	def advance_trial(self):
		"""Move to the next trial."""
		self.current_trial += 1

		# check if we need to move to phase 2
		if self.current_trial == len(self.trial_sequence) // 2 and self.current_phase == 1:
			# advance to phase 2
			self.current_phase = 2

			# reset counters
			self.person_usage_counts_odd_one_out = {pid: 0 for pid in self.persons}
			self.person_usage_counts_diff_gender = {pid: 0 for pid in self.persons}
			self.emotion_usage_counts = {}

	def is_trial_complete(self) -> bool:
		"""Check if all trials are complete."""
		return self.current_trial >= len(self.trial_sequence)

	def get_results_summary(self) -> Dict:
		"""Get summary of results."""
		summary = {}

		for phase in ['phase1', 'phase2']:
			phase_name = 'Without Landmarks' if phase == 'phase1' else 'With Landmarks'
			summary[phase_name] = {}

			for scenario in self.scenario_names:
				results = self.results[phase][scenario]

				if results:
					total = len(results)
					correct = sum(1 for r in results if r['correct'])
					timeouts = sum(1 for r in results if r['timeout'])
					avg_time = sum(r['response_time'] for r in results if not r['timeout']) / max(1, total - timeouts)

					summary[phase_name][scenario] = {
						'total_trials': total,
						'correct_answers': correct,
						'accuracy': correct / total if total > 0 else 0,
						'timeouts': timeouts,
						'average_response_time': avg_time
					}

		return summary

	def save_results_to_excel(self, filename: str):
		"""Save detailed results to Excel file."""
		with pd.ExcelWriter(filename) as writer:
			# create a detailed results sheet
			all_results = []
			for phase in ['phase1', 'phase2']:
				for scenario in self.scenario_names:
					all_results.extend(self.results[phase][scenario])

			if all_results:
				df_detailed = pd.DataFrame(all_results)
				df_detailed.to_excel(writer, sheet_name='Detailed Results', index=False)

			# create a summary sheet
			summary = self.get_results_summary()
			summary_data = []

			for phase_name, phase_data in summary.items():
				total_trials_phase = 0  # track per-phase totals
				total_correct_phase = 0  # track per-phase correct
				total_response_time_phase = 0.0  # track total response time per phase

				for scenario, scenario_data in phase_data.items():
					# add scenario-level data
					row = {
						'Phase': phase_name,
						'Scenario': scenario,
						'Total Trials': scenario_data['total_trials'],
						'Correct Answers': scenario_data['correct_answers'],
						'Accuracy (%)': scenario_data['accuracy'] * 100,
						'Timeouts': scenario_data['timeouts'],
						'Average Response Time (s)': scenario_data['average_response_time']
					}
					summary_data.append(row)

					# accumulate per-phase totals
					total_trials_phase += scenario_data['total_trials']
					total_correct_phase += scenario_data['correct_answers']
					total_response_time_phase += scenario_data['average_response_time'] * scenario_data['total_trials']

				# add an overall per-phase row
				if total_trials_phase > 0:
					overall_accuracy = total_correct_phase / total_trials_phase * 100

					# average across phase
					overall_avg_response = total_response_time_phase / total_trials_phase

					summary_data.append({
						'Phase': phase_name,
						'Scenario': 'Overall',
						'Total Trials': total_trials_phase,
						'Correct Answers': total_correct_phase,
						'Accuracy (%)': overall_accuracy,
						'Timeouts': '',
						# phase average
						'Average Response Time (s)': overall_avg_response
					})

			if summary_data:
				df_summary = pd.DataFrame(summary_data)
				df_summary.to_excel(writer, sheet_name='Summary', index=False)
