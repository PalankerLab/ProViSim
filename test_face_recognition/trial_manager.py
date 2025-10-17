# standard imports
import re
import random
import pandas as pd
from typing import Dict, Optional

# project imports
from test_face_recognition.image_manager import ImageManager


class TrialManager:
	"""Manages the trial scenarios and data collection."""

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
		self.num_trials_per_scenario = 25
		self.current_trial = 0

		# number of different scenarios
		self.num_scenarios = 3
		self.current_scenario = None
		self.scenario_names = ['odd_person_out', 'diff_gender', 'emotion']

		# number of phases (without and with landmarks)
		self.num_phases = 2
		self.trials_per_phase = self.num_trials_per_scenario * self.num_scenarios
		self.current_phase = 1

		# emotions to test
		self.emotions = ['NE', 'HA', 'SA', 'AN', 'FE', 'DI', 'SU', 'CO']

		# setup trial order
		self.setup_trials()

	def setup_trials(self):
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

	def generate_odd_person_out_trial(self):
		"""Identify the different person (3 same + 1 different)"""
		# if there are less than 2 people, we cannot generate this scenario
		if len(self.persons) < 2:
			return None

		# select two people with minimal usage to maximize spread
		min_count = min(self.person_usage_counts_odd_one_out.values())
		least_used = [pid for pid, count in self.person_usage_counts_odd_one_out.items() if count == min_count]

		# set deterministic random for consistent person selection between runs
		rng = random.Random(self.random_seed)

		# if we got less then two people with minimal usage, we need to pick two from the full pool
		if len(least_used) < 2:
			same_person, diff_person = rng.sample(self.persons, 2)

		# otherwise, pick two from the least used people
		else:
			same_person, diff_person = rng.sample(least_used, 2)

		# increment people's usage counts for the chosen people
		self.person_usage_counts_odd_one_out[same_person] += 1
		self.person_usage_counts_odd_one_out[diff_person] += 1

		# get all the images of the same person
		same_images = self.image_manager.get_person_images(same_person, exclude_session2=True)
		if len(same_images) < 3:
			return None

		# get all the images of the different person
		diff_images = self.image_manager.get_person_images(diff_person, exclude_session2=True)
		if len(diff_images) < 1:
			return None

		# select 3 images for the same and one image for the different using global random
		same_selected = random.sample(list(same_images.keys()), 3)
		diff_selected = random.choice(list(diff_images.keys()))

		# create an image list
		images = [same_images[key]['path'] for key in same_selected]
		images.append(diff_images[diff_selected]['path'])

		# shuffle images to randomize the position of the correct answer
		images_combined = images[:]
		random.shuffle(images_combined)

		# get the index of the correct answer
		correct_answer = images_combined.index(images[-1])

		return {
			'type': 'different_person',
			'images': images_combined,
			'correct_answer': correct_answer,
			'question': 'Which person is the odd one out?'
		}

	def generate_different_gender_trial(self):
		"""Identify the different gender (3 same gender + 1 different)."""
		# we need at least 3 people of each gender for this scenario
		if len(self.male_persons) < 3 or len(self.female_persons) < 3:
			return None

		# choose majority gender deterministically across runs
		rng = random.Random(self.random_seed)
		if rng.choice([True, False]):

			# pick three males with minimal usage
			min_count = min([self.person_usage_counts_diff_gender[pid] for pid in self.male_persons])
			least_used_males = [pid for pid in self.male_persons if
								self.person_usage_counts_diff_gender[pid] == min_count]
			same_gender_persons = rng.sample(least_used_males, 3)

			# pick one female with minimal usage
			min_count_f = min([self.person_usage_counts_diff_gender[pid] for pid in self.female_persons])
			least_used_females = [pid for pid in self.female_persons if
								  self.person_usage_counts_diff_gender[pid] == min_count_f]
			diff_gender_person = rng.choice(least_used_females)

		else:

			# female majority - pick three females with minimal usage
			min_count = min([self.person_usage_counts_diff_gender[pid] for pid in self.female_persons])
			least_used_females = [pid for pid in self.female_persons if
								  self.person_usage_counts_diff_gender[pid] == min_count]
			same_gender_persons = rng.sample(least_used_females, 3)

			# pick one male with minimal usage
			min_count_m = min([self.person_usage_counts_diff_gender[pid] for pid in self.male_persons])
			least_used_males = [pid for pid in self.male_persons if
								self.person_usage_counts_diff_gender[pid] == min_count_m]
			diff_gender_person = rng.choice(least_used_males)

		# increment usage counts
		for pid in same_gender_persons:
			self.person_usage_counts_diff_gender[pid] += 1
		self.person_usage_counts_diff_gender[diff_gender_person] += 1

		# get one image from each person
		images = []
		for person in same_gender_persons:
			person_images = self.image_manager.get_person_images(person, exclude_session2=True)
			if person_images:
				img_key = random.choice(list(person_images.keys()))
				images.append(person_images[img_key]['path'])

		diff_images = self.image_manager.get_person_images(diff_gender_person, exclude_session2=True)
		if diff_images:
			img_key = random.choice(list(diff_images.keys()))
			images.append(diff_images[img_key]['path'])

		# ensure we end up with 4 images in the list
		if len(images) != 4:
			return None

		# shuffle images using local seed (position needs to be shuffled differently each time)
		images_combined = images[:]
		random.shuffle(images_combined)

		# get the index of the correct answer
		correct_answer = images_combined.index(images[-1])

		return {
			'type': 'gender',
			'images': images_combined,
			'correct_answer': correct_answer,
			'question': 'Which person is a different gender?'
		}

	def generate_emotion_trial(self):
		"""Identify a specific emotion."""
		# initialize data structures
		eligible_persons = []
		person_emotion_data = {}

		# iterate the people and find all people with at least 4 distinct emotions
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

					if emotion not in self.emotion_usage_counts:
						self.emotion_usage_counts[emotion] = 0

			# need at least 4 different emotions
			if len(emotions) >= 4:
				eligible_persons.append(person)
				person_emotion_data[person] = emotion_images

		# if there are no eligible persons, we cannot generate this scenario
		if not eligible_persons:
			return None

		# use deterministic global seed for consistent emotion selection between runs
		rng = random.Random(self.random_seed)

		# select 4 least-used emotions deterministically
		available_emotions = list({e for person in eligible_persons for e in person_emotion_data[person]})
		available_emotions.sort(key=lambda e: self.emotion_usage_counts.get(e, 0))
		if len(available_emotions) > 4:
			selected_emotions = rng.sample(available_emotions, 4)
		else:
			selected_emotions = available_emotions

		# select one image per emotion using local seed (image needs to be shuffled differently each time)
		images = []
		for emotion in selected_emotions:
			candidate_imgs = []
			for person in eligible_persons:
				imgs = person_emotion_data[person].get(emotion, [])
				candidate_imgs.extend(imgs)
			if not candidate_imgs:
				return None
			chosen_img = random.choice(candidate_imgs)
			images.append(chosen_img)
			self.emotion_usage_counts[emotion] += 1

		# select target emotion
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

		# shuffle images positions
		positions = list(range(4))
		random.shuffle(positions)

		# get the index of the correct answer
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
		# check if we've reached the end of the trial sequence
		if self.current_trial >= len(self.trial_sequence):
			return None

		# get the scenario number for the current trial
		scenario_num = self.trial_sequence[self.current_trial]

		# get the trial configuration for the current scenario
		trial_config = None
		if scenario_num == 0:
			while trial_config is None:
				trial_config = self.generate_odd_person_out_trial()
		elif scenario_num == 1:
			while trial_config is None:
				trial_config = self.generate_different_gender_trial()
		elif scenario_num == 2:
			while trial_config is None:
				trial_config = self.generate_emotion_trial()
		else:
			return None

		# add metadata
		trial_config['scenario'] = scenario_num
		trial_config['trial_number'] = self.current_trial + 1
		trial_config['phase'] = self.current_phase

		return trial_config

	def record_answer(self, trial_config: Dict, selected_answer: int, response_time: float,
					 correct: bool, timeout: bool = False):
		"""Record the participant's answer."""
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

			# reset counters for the second phase
			self.person_usage_counts_odd_one_out = {pid: 0 for pid in self.persons}
			self.person_usage_counts_diff_gender = {pid: 0 for pid in self.persons}
			self.emotion_usage_counts = {}

	def is_trial_complete(self) -> bool:
		"""Check if all trials are complete."""
		return self.current_trial >= len(self.trial_sequence)

	def get_results_summary(self) -> Dict:
		"""Get a summary of the results."""
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
		"""Save detailed results to an Excel file."""
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
