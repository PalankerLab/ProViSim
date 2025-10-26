import os
import re
import glob
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy.stats import ttest_rel, ttest_ind


class TrialAnalysis:

	def __init__(self, input_folder):

		# define scenarios
		self.all_scenarios = ["odd_person_out", "diff_gender", "emotion", "Overall"]

		# validate the input folder
		if not os.path.isdir(input_folder):
			raise NotADirectoryError(f"Folder not found: {input_folder}")

		# path to the folder containing all the Excel files
		self.folder = os.path.join(input_folder, "results_*.xlsx")

		# initialize variables
		self.individual_records = []
		self.pivot_accuracy = None
		self.pivot_response_time = None

		# loop through all Excel files in the folder and populate the individual_records
		for file in glob.glob(self.folder):

			# find files that match the expected naming convention
			match = re.search(r"results_(\d+)_(f|m)_(\d{8})_(\d+)\.xlsx", file, re.IGNORECASE)

			# skip files that do not
			if not match:
				continue

			# extract participant's ID, gender, date, and time
			subject_id, gender, date, time = match.groups()

			# ensure gender is lowercase
			gender = gender.lower()

			# load the Summary tab from the Excel file to a data frame
			summary_df = pd.read_excel(file, sheet_name="Summary")

			# add the extracted information to the DataFrame
			summary_df["Subject"] = subject_id
			summary_df["Gender"] = gender
			self.individual_records.append(summary_df)

		# combine all subject data into a single DataFrame
		self.combined_records = pd.concat(self.individual_records, ignore_index=True)

	def run_analysis(self):
		# pivot the data
		self._pivot_data()

		# run paired t-tests
		significance_dict = self._student_t_test()

		# # check the normality of the data
		# self._plot_distributions()

		# add boxplots that showcase the difference between before and after
		self._add_boxplots(significance_dict)

		# add bar charts
		self._add_bar_charts()

		# analyze specific emotions
		for emotion in ["happy", "sad", "confused", "neutral", "disgusted", "fearful", "surprised", "angry"]:
			# run a paired t-test for this emotion
			significance_dict_emotion = self.specific_emotion_ttest(emotion)

			# plot this emotion's boxplot
			self.specific_emotion_boxplot(significance_dict_emotion, emotion)

	def _pivot_data(self):
		# pivot accuracy for side-by-side comparison
		self.pivot_accuracy = self.combined_records.pivot_table(
			index=["Subject", "Gender", "Scenario"],
			columns="Phase",
			values="Accuracy (%)"
		).reset_index()

		# pivot response time for side-by-side comparison
		self.pivot_response_time = self.combined_records.pivot_table(
			index=["Subject", "Gender", "Scenario"],
			columns="Phase",
			values="Average Response Time (s)"
		).reset_index()

		# compute within-subject differences and add to the pivoted data
		self.pivot_accuracy["Delta"] = self.pivot_accuracy["With Landmarks"] - self.pivot_accuracy["Without Landmarks"]
		self.pivot_response_time["Delta"] = self.pivot_response_time["Without Landmarks"] - self.pivot_response_time[
			"With Landmarks"]

	def _student_t_test(self):
		# initialize results
		results_within = []
		results_between_male_and_female = []
		results_gender_within = []  # new

		# run paired t-tests across all subjects for each scenario
		for scenario in self.all_scenarios:
			scenario_accuracy = self.pivot_accuracy[self.pivot_accuracy["Scenario"] == scenario]
			scenario_response_time = self.pivot_response_time[self.pivot_response_time["Scenario"] == scenario]

			# paired t-tests
			accuracy_t, accuracy_p = ttest_rel(
				scenario_accuracy["With Landmarks"],
				scenario_accuracy["Without Landmarks"]
			)
			response_time_t, response_time_p = ttest_rel(
				scenario_response_time["With Landmarks"],
				scenario_response_time["Without Landmarks"]
			)

			# save accuracy and response time results for each scenario
			results_within.append({
				"Scenario": scenario,
				"Metric": "Accuracy (%)",
				"t": accuracy_t,
				"p": accuracy_p,
				"Significant": accuracy_p < 0.05,
				"Significance_Level": (
					"***" if accuracy_p < 0.001 else
					"**" if accuracy_p < 0.01 else
					"*" if accuracy_p < 0.05 else ""
				)
			})
			results_within.append({
				"Scenario": scenario,
				"Metric": "Response Time (s)",
				"t": response_time_t,
				"p": response_time_p,
				"Significant": response_time_p < 0.05,
				"Significance_Level": (
					"***" if response_time_p < 0.001 else
					"**" if response_time_p < 0.01 else
					"*" if response_time_p < 0.05 else ""
				)
			})

		# run paired t-tests across subjects of the same gender
		for scenario in self.all_scenarios:
			for gender in ["f", "m"]:
				acc_sub = self.pivot_accuracy[
					(self.pivot_accuracy["Scenario"] == scenario) &
					(self.pivot_accuracy["Gender"] == gender)
					]
				rt_sub = self.pivot_response_time[
					(self.pivot_response_time["Scenario"] == scenario) &
					(self.pivot_response_time["Gender"] == gender)
					]

				if len(acc_sub) > 1:
					accuracy_t, accuracy_p = ttest_rel(
						acc_sub["With Landmarks"],
						acc_sub["Without Landmarks"]
					)
					results_gender_within.append({
						"Scenario": scenario,
						"Gender": gender,
						"Metric": "Accuracy (%)",
						"t": accuracy_t,
						"p": accuracy_p,
						"Significant": accuracy_p < 0.05,
						"Significance_Level": (
							"***" if accuracy_p < 0.001 else
							"**" if accuracy_p < 0.01 else
							"*" if accuracy_p < 0.05 else ""
						)
					})

				if len(rt_sub) > 1:
					response_time_t, response_time_p = ttest_rel(
						rt_sub["With Landmarks"],
						rt_sub["Without Landmarks"]
					)
					results_gender_within.append({
						"Scenario": scenario,
						"Gender": gender,
						"Metric": "Response Time (s)",
						"t": response_time_t,
						"p": response_time_p,
						"Significant": response_time_p < 0.05,
						"Significance_Level": (
							"***" if response_time_p < 0.001 else
							"**" if response_time_p < 0.01 else
							"*" if response_time_p < 0.05 else ""
						)
					})

		# run regular t-tests between female vs. male subjects on improvement data
		for scenario in self.all_scenarios:
			acc_sub = self.pivot_accuracy[self.pivot_accuracy["Scenario"] == scenario]
			rt_sub = self.pivot_response_time[self.pivot_response_time["Scenario"] == scenario]

			# get improvement data by gender
			f_acc = acc_sub[acc_sub["Gender"] == "f"]["Delta"]
			m_acc = acc_sub[acc_sub["Gender"] == "m"]["Delta"]
			f_rt = rt_sub[rt_sub["Gender"] == "f"]["Delta"]
			m_rt = rt_sub[rt_sub["Gender"] == "m"]["Delta"]

			# run t-tests (independent data sets)
			accuracy_t, accuracy_p = ttest_ind(f_acc, m_acc, equal_var=False)
			response_time_t, response_time_p = ttest_ind(f_rt, m_rt, equal_var=False)

			# save accuracy and response time results for each scenario
			results_between_male_and_female.append({
				"Scenario": scenario,
				"Metric": "Accuracy Improvement (%)",
				"t": accuracy_t,
				"p": accuracy_p,
				"Significant": accuracy_p < 0.05,
				"Significance_Level": (
					"***" if accuracy_p < 0.001 else
					"**" if accuracy_p < 0.01 else
					"*" if accuracy_p < 0.05 else ""
				)
			})
			results_between_male_and_female.append({
				"Scenario": scenario,
				"Metric": "Response Time Improvement (s)",
				"t": response_time_t,
				"p": response_time_p,
				"Significant": response_time_p < 0.05,
				"Significance_Level": (
					"***" if response_time_p < 0.001 else
					"**" if response_time_p < 0.01 else
					"*" if response_time_p < 0.05 else ""
				)
			})

		# convert to dataframes for readable output
		within_df = pd.DataFrame(results_within)
		between_df = pd.DataFrame(results_between_male_and_female)
		gender_within_df = pd.DataFrame(results_gender_within)

		print("\n=== within-subject (with vs without landmarks) ===")
		print(within_df.round(4))

		print("\n=== between-group (female vs male improvements) ===")
		print(between_df.round(4))

		print("\n=== within-subject by gender (paired t-tests) ===")
		print(gender_within_df.round(4))

		# group descriptive statistics (mean ± sd by gender and phase)
		desc = (
			self.combined_records.groupby(["Gender", "Scenario", "Phase"])
			.agg(
				Accuracy_mean=("Accuracy (%)", "mean"),
				Accuracy_std=("Accuracy (%)", "std"),
				RT_mean=("Average Response Time (s)", "mean"),
				RT_std=("Average Response Time (s)", "std")
			)
			.reset_index()
		)

		print("\n=== descriptive statistics (mean ± sd by gender and phase) ===")
		print(desc.round(3))

		return {
			"within": within_df,
			"between": between_df,
			"gender_within": gender_within_df,
			"desc": desc
		}

	def _plot_distributions(self):
		for metric, df in [("Accuracy (%)", self.pivot_accuracy),
						   ("Response Time (s)", self.pivot_response_time)]:
			for scenario in self.all_scenarios:
				d = df[df["Scenario"] == scenario]["Delta"].dropna()
				if d.empty:
					continue
				plt.figure(figsize=(10,4))
				plt.subplot(1,2,1)
				plt.hist(d, bins=10, alpha=0.7)
				plt.title(f"{scenario} - {metric} differences")
				plt.xlabel("difference"); plt.ylabel("count")
				plt.subplot(1,2,2)
				stats.probplot(d, dist="norm", plot=plt)
				plt.title("qq plot")
				plt.tight_layout()
				plt.show()

	def _add_bar_charts(self):
		# define colors to use for plotting
		colors = ["tab:blue", "tab:grey", "tab:pink"]
		light_factor = 0.5

		# pivot the data so we have before/after columns
		pivoted = self.combined_records.pivot_table(
			index=["Subject", "Scenario"],
			columns="Phase",
			values="Accuracy (%)"
		).reset_index()

		# create a full plot for all the scenarios together
		fig, ax = plt.subplots(figsize=(20, 6))
		bar_width = 0.25
		gap_between_scenarios = 0.05
		gap_between_subjects = 0.5

		# exclude the "Overall" scenario
		scenarios = [s for s in self.all_scenarios if s != "Overall"]

		# get unique subjects sorted numerically
		subjects = sorted(pivoted["Subject"].astype(int).unique())
		num_scenarios = len(scenarios)

		# compute cluster width per subject
		cluster_width = num_scenarios * (2 * bar_width + gap_between_scenarios)
		plotted_labels = set()
		cluster_centers = []

		for i, subject in enumerate(subjects):
			subj_data = pivoted[pivoted["Subject"].astype(int) == subject].set_index("Scenario")
			x_start = i * (cluster_width + gap_between_subjects)
			cluster_centers.append(x_start + cluster_width / 2 - bar_width)

			for j, scenario in enumerate(scenarios):
				if scenario not in subj_data.index:
					continue
				vals = subj_data.loc[scenario]

				# add lighter bars for the before values
				color_before = mcolors.to_rgba(colors[j % len(colors)], alpha=light_factor)
				before_val = vals["Without Landmarks"] if "Without Landmarks" in vals else 0
				label_before = f"{scenario} Before"
				ax.bar(
					x_start + j * (2 * bar_width + gap_between_scenarios) - bar_width / 2,
					before_val,
					bar_width,
					color=color_before,
					label=label_before if label_before not in plotted_labels else ""
				)
				plotted_labels.add(label_before)

				# add darker bars for the after values
				color_after = colors[j % len(colors)]
				after_val = vals["With Landmarks"] if "With Landmarks" in vals else 0
				label_after = f"{scenario} After"
				ax.bar(
					x_start + j * (2 * bar_width + gap_between_scenarios) + bar_width / 2,
					after_val,
					bar_width,
					color=color_after,
					label=label_after if label_after not in plotted_labels else ""
				)
				plotted_labels.add(label_after)

		# finish formatting the figure
		ax.set_xticks(cluster_centers)
		ax.set_xticklabels([str(s) for s in subjects], rotation=45)
		ax.set_ylabel("Accuracy [%]")
		ax.set_xlabel("Subject ID")
		ax.set_title("Per Subject Accuracy - All Scenarios")
		ax.grid(alpha=0.3)
		ax.legend()
		plt.tight_layout()
		plt.show()

	def _add_boxplots(self, stats_results):
		# pivot the data so we have before/after columns
		acc_long = self.pivot_accuracy.melt(
			id_vars=["Subject", "Gender", "Scenario"],
			value_vars=["Without Landmarks", "With Landmarks"],
			var_name="Phase",
			value_name="Accuracy (%)"
		)

		rt_long = self.pivot_response_time.melt(
			id_vars=["Subject", "Gender", "Scenario"],
			value_vars=["Without Landmarks", "With Landmarks"],
			var_name="Phase",
			value_name="Average Response Time (s)"
		)

		# get scenarios to plot (exclude "Overall")
		scenarios = acc_long["Scenario"].unique()
		scenarios = [s for s in scenarios if s != "Overall"]

		phases = ["Without Landmarks", "With Landmarks"]
		colors = ["tab:grey", "tab:pink"]

		# define positions for the boxplots
		bar_width = 0.35
		spacing = 0.4
		x_positions = np.arange(len(scenarios)) * (len(phases) * bar_width + spacing)

		# create an accuracy plot
		fig, ax = plt.subplots(figsize=(10, 6))
		max_value = 0
		for i, scenario in enumerate(scenarios):
			scenario_data = acc_long[acc_long["Scenario"] == scenario]
			whiskers_max = []
			stats_text = ""  # for console output

			for j, phase in enumerate(phases):
				sub = scenario_data[scenario_data["Phase"] == phase]
				pos = x_positions[i] + j * bar_width
				bp = ax.boxplot(
					sub["Accuracy (%)"],
					positions=[pos],
					widths=bar_width * 0.9,
					patch_artist=True,
					boxprops=dict(facecolor=colors[j], color=colors[j]),
					medianprops=dict(color="black"),
					whiskerprops=dict(color=colors[j]),
					capprops=dict(color=colors[j])
				)
				whiskers_max.append(bp['whiskers'][1].get_ydata()[1])

				# compute mean and median
				mean_val = sub["Accuracy (%)"].mean()
				median_val = sub["Accuracy (%)"].median()
				stats_text += f"{phase}: Mean={mean_val:.1f}, Median={median_val:.1f}\n"

			max_value = max(max_value, max(whiskers_max))

			# print stats to console
			print(f"\nAccuracy - Scenario: {scenario}\n{stats_text.strip()}")

			# significance marking
			sig_row = stats_results["within"][
				(stats_results["within"]["Scenario"] == scenario) &
				(stats_results["within"]["Metric"] == "Accuracy (%)")
				]
			if not sig_row.empty and sig_row.iloc[0]["Significant"]:
				star = sig_row.iloc[0]["Significance_Level"]
				ax.text(
					x_positions[i] + bar_width / 2,
					max(whiskers_max) + 0.05 * max_value,
					star,
					ha="center",
					va="bottom",
					color="black",
					fontsize=14,
					fontweight="bold"
				)

		ax.set_xticks(x_positions + bar_width / 2)
		ax.set_xticklabels(scenarios, rotation=30)
		ax.set_ylabel("Accuracy [%]")
		ax.set_title("Accuracy")
		ax.legend(["Without Enhancements", "With Enhancements"])
		ax.set_ylim(top=max_value * 1.1)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.show()

		# create response time plot
		fig, ax = plt.subplots(figsize=(10, 6))
		max_value = 0
		for i, scenario in enumerate(scenarios):
			scenario_data = rt_long[rt_long["Scenario"] == scenario]
			whiskers_max = []
			stats_text = ""  # for console output

			for j, phase in enumerate(phases):
				sub = scenario_data[scenario_data["Phase"] == phase]
				pos = x_positions[i] + j * bar_width
				bp = ax.boxplot(
					sub["Average Response Time (s)"],
					positions=[pos],
					widths=bar_width * 0.9,
					patch_artist=True,
					boxprops=dict(facecolor=colors[j], color=colors[j]),
					medianprops=dict(color="black"),
					whiskerprops=dict(color=colors[j]),
					capprops=dict(color=colors[j])
				)
				whiskers_max.append(bp['whiskers'][1].get_ydata()[1])

				# compute mean and median
				mean_val = sub["Average Response Time (s)"].mean()
				median_val = sub["Average Response Time (s)"].median()
				stats_text += f"{phase}: Mean={mean_val:.2f}s, Median={median_val:.2f}s\n"

			max_value = max(max_value, max(whiskers_max))

			# print stats to console
			print(f"\nResponse Time - Scenario: {scenario}\n{stats_text.strip()}")

			# significance marking
			sig_row = stats_results["within"][
				(stats_results["within"]["Scenario"] == scenario) &
				(stats_results["within"]["Metric"] == "Response Time (s)")
				]
			if not sig_row.empty and sig_row.iloc[0]["Significant"]:
				star = sig_row.iloc[0]["Significance_Level"]
				ax.text(
					x_positions[i] + bar_width / 2,
					max(whiskers_max) + 0.05 * max_value,
					star,
					ha="center",
					va="bottom",
					color="black",
					fontsize=14,
					fontweight="bold"
				)

		ax.set_xticks(x_positions + bar_width / 2)
		ax.set_xticklabels(scenarios, rotation=30)
		ax.set_ylabel("Average response time [s]")
		ax.set_title("Average Response Time")
		ax.legend(["Without Enhancements", "With Enhancements"])
		ax.set_ylim(top=max_value * 1.1)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=10))  # <- NEW: more y-axis ticks
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.show()

	def specific_emotion_ttest(self, emotion):
		"""Perform a paired t-test for happy emotion accuracy across all participants from Detailed Results."""
		all_records = []

		# iterate through all result files
		for file in glob.glob(self.folder):
			match = re.search(r"results_(\d+)_(f|m)_(\d{8})_(\d+)\.xlsx", file, re.IGNORECASE)
			if not match:
				continue
			subject_id, _, _, _ = match.groups()
			df = pd.read_excel(file, sheet_name="Detailed Results")
			df["Subject"] = subject_id
			all_records.append(df)

		# combine all participants
		combined_df = pd.concat(all_records, ignore_index=True)

		# filter only happy emotion trials
		happy_df = combined_df[combined_df["question"].str.contains(emotion, case=False)]

		# pivot to get before/after per participant
		pivot = happy_df.pivot_table(
			index="Subject",
			columns="phase",
			values="correct"
		).reset_index()

		# paired t-test
		t_stat, p_val = ttest_rel(pivot[2], pivot[1])

		# determine the significance
		if p_val < 0.001:
			sig = "***"
		elif p_val < 0.01:
			sig = "**"
		elif p_val < 0.05:
			sig = "*"
		else:
			sig = ""

		result = {
			"t_stat": t_stat,
			"p_val": p_val,
			"significance": sig,
			"before": pivot[1],
			"after": pivot[2]
		}
		return result

	@staticmethod
	def specific_emotion_boxplot(significance_dict, emotion):
		"""Plot happy emotion accuracy before vs after with significance for all participants."""
		before = significance_dict["before"] * 100
		after = significance_dict["after"] * 100
		sig = significance_dict["significance"]

		data = [before, after]
		labels = ["Without Enhancements", "With Enhancements"]
		colors = ["tab:grey", "tab:pink"]

		fig, ax = plt.subplots(figsize=(6, 6))
		bp = ax.boxplot(data, patch_artist=True, labels=labels,
						boxprops=dict(facecolor=colors[0], color=colors[0]),
						medianprops=dict(color="black"))
		for patch, color in zip(bp['boxes'], colors):
			patch.set_facecolor(color)

		# print statistics
		print(f"\n{str.capitalize(emotion)} Emotion Accuracy (all participants):")
		print(f"Without Enhancements - Mean: {before.mean():.2f}, Median: {before.median():.2f}")
		print(f"With Enhancements    - Mean: {after.mean():.2f}, Median: {after.median():.2f}")
		print(f"Paired t-test: t={significance_dict['t_stat']:.4f}, p={significance_dict['p_val']:.4f}, Significance: {sig}")

		# determine y-limit to fit significance inside plot
		max_val = max(after.max(), before.max())
		ylim_top = max_val * 1.2
		ax.set_ylim(0, ylim_top)
		ax.yaxis.set_major_locator(MaxNLocator(nbins=10))

		# add significance star
		ax.text(1.5, max(after.max(), before.max()) * 1.05, sig,
				ha='center', va='bottom', fontsize=16, fontweight='bold')

		ax.set_ylabel("Accuracy [%]")
		ax.set_xticklabels(labels, rotation=30)
		ax.set_yticks([10 * i for i in range(1, 11)])
		ax.set_ylabel("Accuracy [%]")
		ax.set_title(f"{str.capitalize(emotion)} Emotion Accuracy")
		ax.grid(alpha=0.3)
		plt.tight_layout()
		plt.show()


if __name__ == "__main__":
	input_folder = os.path.join(os.getcwd(), "results")
	analysis = TrialAnalysis(input_folder=input_folder)
	analysis.run_analysis()