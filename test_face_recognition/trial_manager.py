# standard imports
import random
from enum import Enum
import pandas as pd
from collections import Counter
from typing import Dict, Optional

# project imports
from test_face_recognition.image_manager import ImageManager

# define constants
class Scenario(Enum):
	ODD_PERSON_OUT = 0
	DIFFERENT_GENDER = 1
	EMOTION = 2

# helper function to cycle through positions
def _balanced_cycle(num=4):
	i = random.randrange(num)
	while True:
		yield i
		i = (i + 1) % num


class TrialManager:
	"""Manages the trial scenarios and data collection."""

	def __init__(self, image_manager: ImageManager):
		# initialize output
		self.trial_screens = []
		self.emotion_tuples = []
		self.odd_person_out_people = []
		self.diff_gender_people = []

		# initialize variables
		self.results = {
			'phase1': {'odd_person_out': [], 'diff_gender': [], 'emotion': []},
			'phase2': {'odd_person_out': [], 'diff_gender': [], 'emotion': []}
		}

		# create a global random seed for consistency across subjects
		self.random_seed = 713
		self.global_random = random.Random(self.random_seed)

		# save image manager
		self.image_manager = image_manager

		# get all persons and initialize usage counts
		self.people = self.image_manager.get_all_persons()

		# get all persons by gender and initialize usage counts
		self.male_persons = self.image_manager.get_persons_by_gender('M')
		self.female_persons = self.image_manager.get_persons_by_gender('F')

		# ensure there are at least 10 people and 3 people of each gender
		if len(self.male_persons) < 3 or len(self.female_persons) < 3:
			raise Exception('Not enough people in the dataset to run this experiment.')

		# get people with emotions and initialize usage counts
		self.people_with_emotions = self.image_manager.get_people_with_emotions()

		# number of trials per scenario
		self.num_screens_per_scenario = 24
		self.current_trial = 0

		# number of different scenarios
		self.current_scenario = None
		self.scenario_names = ['odd_person_out', 'diff_gender', 'emotion']
		self.num_scenarios = len(self.scenario_names)

		# number of phases (without and with landmarks)
		self.num_phases = 2
		self.screens_per_phase = self.num_screens_per_scenario * self.num_scenarios
		self.current_phase = 1

		# emotions to test
		self.emotions = self.image_manager.get_emotions()

		# start a position cycle function for each scenario
		self.odd_person_out_cycle = _balanced_cycle()
		self.diff_gender_cycle = _balanced_cycle()
		self.emotion_cycle = _balanced_cycle()

		# generate emotion tuples and create an iterator
		self.emotion_tuples_iter = iter(self._generate_emotion_tuples())

		# setup trial order
		self._setup_trials()

	def _generate_emotion_tuples(self):
		# create a balanced list of the correct answers
		base, r = divmod(self.num_screens_per_scenario, len(self.emotions))
		corrects = [e for e in self.emotions for _ in range(base)] + self.emotions[:r]

		# shuffle the list of correct answers
		self.global_random.shuffle(corrects)

		# create screen tuples
		trials = []
		for c in corrects:
			# get the next position for the correct answer
			p = next(self.emotion_cycle)

			# sample 3 other emotions that are not the correct one
			opts = random.sample([e for e in self.emotions if e != c], 3)

			# insert the correct answer into the position that was randomly selected for the correct answer
			opts.insert(p, c)

			# create output tuple: (emotion1,emotion2,emotion3,emotion4,correct)
			trials.append(tuple(opts + [c]))

		return trials

	def _setup_trials(self):
		# initialize local output variables
		phase_screens = []
		phase_odd_person_out_people = []
		phase_diff_gender_people = []
		phase_emotions = []

		# create the requested number of screens for each scenario
		for scenario in range(0, self.num_scenarios):
			for screen_num in range(self.num_screens_per_scenario):

				# add the screen number to the list of screens for this phase
				phase_screens.append(scenario)

				# random generator for consistency across participants but variability between screens
				local_random = random.Random(self.random_seed + screen_num)

				# if the scenario is odd person out, pick two different people
				if scenario == Scenario.ODD_PERSON_OUT.value:
					same_person, diff_person = local_random.sample(self.people, 2)
					phase_odd_person_out_people.append((same_person, diff_person))
				else:
					phase_odd_person_out_people.append((None, None))

				# if the scenario is a different gender, pick two different genders
				if scenario == Scenario.DIFFERENT_GENDER.value:
					# randomize majority gender
					if random.choice([True, False]):
						same_gender_person = local_random.sample(self.male_persons, 3)
						diff_gender_person = local_random.sample(self.female_persons, 1)
					else:
						same_gender_person = local_random.sample(self.female_persons, 3)
						diff_gender_person = local_random.sample(self.male_persons, 1)
					phase_diff_gender_people.append((same_gender_person, diff_gender_person))
				else:
					phase_diff_gender_people.append((None, None))

				# if the scenario is emotions, choose 4 emotions
				if scenario == Scenario.EMOTION.value:
					selected_emotions = next(self.emotion_tuples_iter)
					phase_emotions.append(selected_emotions)
				else:
					phase_emotions.append([])

		# create a vector the same length as the number of screens per phase
		initial_order = list(range(0, self.screens_per_phase))

		# copy the vector to shuffle the order of the trials for each phase
		shuffled_order = initial_order.copy()

		# shuffle the order of the trials for each phase
		self.global_random.shuffle(shuffled_order)

		# create a mapping of the shuffle
		shuffle_mapping = {orig: shuffled_order.index(orig) for orig in initial_order}

		# construct the trial sequence
		for phase in range(0, self.num_phases):

			# # reseed the random generator for each phase
			# self.global_random.seed(self.random_seed + phase)

			# shuffle the all structures according to the mapping
			shuffled_phase_screens = [phase_screens[shuffle_mapping[i]] for i in range(self.screens_per_phase)]
			shuffled_phase_odd_person_out_people = [phase_odd_person_out_people[shuffle_mapping[i]] for i in range(self.screens_per_phase)]
			shuffled_phase_diff_gender_people = [phase_diff_gender_people[shuffle_mapping[i]] for i in range(self.screens_per_phase)]
			shuffled_phase_emotions = [phase_emotions[shuffle_mapping[i]] for i in range(self.screens_per_phase)]

			# add the phase screens to the trial sequence
			self.trial_screens.extend(shuffled_phase_screens)
			self.odd_person_out_people.extend(shuffled_phase_odd_person_out_people)
			self.diff_gender_people.extend(shuffled_phase_diff_gender_people)
			self.emotion_tuples.extend(shuffled_phase_emotions)

	def _generate_odd_person_out_scenario(self):
		"""Identify the different person (3 same + 1 different)"""
		# initialize output
		images_to_display = []

		# get the indices of the people for this screen number
		same_person, diff_person = self.odd_person_out_people[self.current_trial]

		# get all the images of the same person and the other person
		same_images = self.image_manager.get_person_images(same_person, exclude_session2=True)
		diff_images = self.image_manager.get_person_images(diff_person, exclude_session2=True)

		# initialize local random generator
		local_random = random.Random()

		# randomly, select 3 images for the same person and one image for the different person
		same_selected = local_random.sample(list(same_images.keys()), 3)
		diff_selected = local_random.sample(list(diff_images.keys()), 1)

		# generate a new local random generator
		local_random = random.Random()

		# shuffle the images of the same person
		local_random.shuffle(same_selected)

		# create an image list and add the images of the same person
		images_to_display.extend([same_images[key]['path'] for key in same_selected])

		# get the next position for the correct answer
		correct_answer = next(self.odd_person_out_cycle)

		# insert the different person into the position that was randomly selected for the correct answer
		images_to_display.insert(correct_answer, diff_images[diff_selected[0]]['path'])

		return {
			'type': 'odd_person_out',
			'images': images_to_display,
			'correct_answer': correct_answer,
			'question': 'Which person is the odd one out?'
		}

	def _generate_different_gender_scenario(self):
		"""Identify the different gender (3 same gender + 1 different)."""
		# initialize output
		images_to_display = []

		# get the indices of the people for this screen number
		same_gender_persons, diff_gender_person = self.diff_gender_people[self.current_trial]

		# create a local random generator to pick different images and positions each time
		local_random = random.Random()

		# get the next position for the correct answer
		correct_answer = next(self.diff_gender_cycle)

		# iterate over the same gender persons and select one image from each
		for person in same_gender_persons:
			person_images = self.image_manager.get_person_images(person, exclude_session2=True)
			if person_images:
				img_key = local_random.choice(list(person_images.keys()))
				images_to_display.append(person_images[img_key]['path'])

		# shuffle the same gender images
		local_random.shuffle(images_to_display)

		# get the other person's images and select one image
		diff_images = self.image_manager.get_person_images(diff_gender_person[0], exclude_session2=True)
		if diff_images:
			img_key = local_random.choice(list(diff_images.keys()))

			# insert the different person into the position that was randomly selected for the correct answer
			images_to_display.insert(correct_answer, diff_images[img_key]['path'])

		# correct_answer, images_to_display_shuffled = self._get_shuffled_correct_answer(images_to_display, original_correct_index=original_correct_answer)

		return {
			'type': 'diff_gender',
			'images': images_to_display,
			'correct_answer': correct_answer,
			'question': 'Which person is a different gender?'
		}

	def _generate_emotion_scenario(self):
		"""Identify a specific emotion."""
		# initialize output
		images_to_display = []

		# get the emotions for this screen number
		selected_emotions = self.emotion_tuples[self.current_trial]

		# create a local random generator to pick different images and positions each time
		local_random = random.Random()

		# for each emotion, get all images of this emotion from all people with emotions
		for emotion in selected_emotions[:4]:

			# initialize candidate images for this emotion
			candidate_imgs = []

			# iterate over people with emotions
			for person in self.people_with_emotions:

				# get all images of this person with this emotion
				imgs = self.image_manager.get_person_emotion_images(person).get(emotion, [])

				# store images
				candidate_imgs.extend(imgs)

			# choose one image at random from the candidates for this emotion
			chosen_img = local_random.choice(candidate_imgs)

			# save the chosen image and update the usage count
			images_to_display.append(chosen_img)

		# the last emotion in the tuple is the correct answer
		correct_emotion = selected_emotions[-1]

		# get the full emotion name
		correct_emotion_name = self.image_manager.get_emotions_dict().get(correct_emotion, correct_emotion.lower())

		# get the index of the correct answer
		correct_answer= selected_emotions[:4].index(correct_emotion)

		return {
			'type': 'emotion',
			'images': images_to_display,
			'correct_answer': correct_answer,
			'question': f'Which face looks {correct_emotion_name}?'
		}

	def advance_trial(self):
		"""Move to the next trial."""
		# update counter
		self.current_trial += 1

		# check if we need to move to phase 2
		if self.current_trial == len(self.trial_screens) // 2 and self.current_phase == 1:
			self.current_phase = 2

	def get_next_trial(self) -> Optional[Dict]:
		"""Get the next trial configuration."""
		# check if we've reached the end of the trial
		if self.current_trial >= len(self.trial_screens):
			return None

		# get the scenario number for the current trial
		scenario_num = self.trial_screens[self.current_trial]

		# get the trial configuration for the current scenario
		if scenario_num == Scenario.ODD_PERSON_OUT.value:
			trial_config = self._generate_odd_person_out_scenario()
		elif scenario_num == Scenario.DIFFERENT_GENDER.value:
			trial_config = self._generate_different_gender_scenario()
		elif scenario_num == Scenario.EMOTION.value:
			trial_config = self._generate_emotion_scenario()
		else:
			raise Exception(f'No implementation for scenario number: {scenario_num}')

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
