import os
import sys
import time
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from prosthetic_vision import apply_enhanced_prosthetic_vision, apply_prosthetic_vision
from test_face_recognition.image_manager import ImageManager
from test_face_recognition.trial_manager import TrialManager


class TrialGUI:
	"""main gui for prosthetic vision testing"""

	def __init__(self, root, debug=False, disable_processing=False):
		# initialize GUI
		self.root = root
		self.root.title("Face Recognition with Prosthetic Vision")
		self.root.geometry("1200x800")
		self.root.configure(bg="white")

		# initialize variables
		self.image_manager = None
		self.trial_manager = None
		self.current_trial_config = None
		self.trial_start_time = None
		self.timeout_id = None
		self.subject_id = ""
		self.image_buttons = []

		# set debug flags
		self.debug = debug
		self.debug_label = None
		self.disable_processing = disable_processing

		# create a FaceLandmarker object once and reuse it for all images
		model_asset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
										'face_landmarker_v2_with_blendshapes.task')
		base_options = python.BaseOptions(
			model_asset_path=model_asset_path,
			delegate=python.BaseOptions.Delegate.CPU
		)
		options = vision.FaceLandmarkerOptions(
			base_options=base_options,
			output_face_blendshapes=False,
			output_facial_transformation_matrixes=False,
			num_faces=1
		)

		# create the detector
		self.detector = vision.FaceLandmarker.create_from_options(options)

		# create the setup screen
		self.create_setup_screen()

	def create_setup_screen(self):
		# clear the root
		self.clear_root()

		# create root frame with full grid
		main = tk.Frame(self.root, bg="white")
		main.grid(row=0, column=0, sticky="nsew")

		self.root.grid_rowconfigure(0, weight=1)
		self.root.grid_columnconfigure(0, weight=1)

		for i in range(3):
			main.grid_rowconfigure(i, weight=1)
		main.grid_columnconfigure(0, weight=1)
		main.grid_columnconfigure(1, weight=1)
		main.grid_columnconfigure(2, weight=1)

		# title
		title = tk.Label(
			main,
			text="Face Recognition with Prosthetic Vision",
			font=("Arial", 24, "bold"),
			bg="white",
		)
		title.grid(row=0, column=0, columnspan=3, sticky="nsew")

		# create the subject ID label
		label = tk.Label(main, text="Subject ID:", font=("Arial", 14), bg="white")
		label.grid(row=1, column=0, sticky="e")

		# create the subject ID entry
		self.subject_entry = tk.Entry(main, font=("Arial", 14))
		self.subject_entry.grid(row=1, column=1, sticky="ew")

		# make it grab focus on any click
		self.subject_entry.bind("<Button-1>", lambda e: self.subject_entry.focus_set())

		# when the user presses Enter, start the trial
		self.root.bind("<Return>", lambda e: self.start_trial())

	def create_trial_screen(self):
		# clear the root
		self.clear_root()

		# create root frame with full grid
		frame = tk.Frame(self.root, bg="white")
		frame.grid(row=0, column=0, sticky="nsew")
		self.root.grid_rowconfigure(0, weight=1)
		self.root.grid_columnconfigure(0, weight=1)

		# setup grid for 2 rows: question and images
		frame.grid_rowconfigure(0, weight=1)
		frame.grid_rowconfigure(1, weight=6)
		for i in range(2):
			frame.grid_columnconfigure(i, weight=1)

		# create a question label
		self.question_label = tk.Label(
			frame, text="", font=("Arial", 18, "bold"), bg="white"
		)
		self.question_label.grid(row=0, column=0, columnspan=2, sticky="nsew")

		# create an image grid (2x2)
		grid = tk.Frame(frame, bg="white")
		grid.grid(row=1, column=0, columnspan=2, sticky="nsew")
		for r in range(2):
			grid.grid_rowconfigure(r, weight=1)
		for c in range(2):
			grid.grid_columnconfigure(c, weight=1)

		# define labels for the images (not clickable)
		self.image_buttons = []
		for i in range(2):
			for j in range(2):
				idx = i * 2 + j
				lbl = tk.Label(
					grid,
					text=f"{idx + 1}",
					font=("Arial", 18, "bold"),
					bg="#d9d9d9",
					relief="raised",
					width=25,
					height=10,
				)
				lbl.grid(row=i, column=j, sticky="nsew")
				self.image_buttons.append(lbl)

		# add keyboard bindings
		for key in ["1", "2", "3", "4"]:
			self.root.bind(key, self.on_key_press)

	def start_trial(self):
		# validate the subject ID
		subject = self.subject_entry.get().strip()
		if not subject:
			messagebox.showerror("Error", "Please enter a Subject ID")
			return

		# store subject ID
		self.subject_id = subject

		# initialize managers
		self.image_manager = ImageManager()
		self.trial_manager = TrialManager(self.image_manager)

		# move to the trial screen
		self.create_trial_screen()

		# start trial
		self.next_trial()

	def next_trial(self):
		# cancel the timeout if it is running
		if self.timeout_id:
			self.root.after_cancel(self.timeout_id)

		# get the next trial configuration
		self.current_trial_config = self.trial_manager.get_next_trial()
		if not self.current_trial_config:
			self.finish_trial()
			return

		# display a new question
		self.question_label.config(text=self.current_trial_config["question"])

		# load the images
		self.load_images()

		# start timer
		self.trial_start_time = time.time()

		# timeout trial after 15 seconds
		self.timeout_id = self.root.after(20000, self.timeout_trial)

	def load_images(self):
		# define common parameters
		apply_enhancements = self.trial_manager.current_phase == 2
		cutoff_frequency, gain, shift = 10, 30, 0.2

		try:
			# iterate over the images and prepare them for display
			for i, path in enumerate(self.current_trial_config["images"]):
				img = Image.open(path).convert("RGB")
				img_array = np.array(img)

				# get original dimensions
				orig_h, orig_w = img_array.shape[:2]

				# automatically zoom in on non-white regions (head area)
				try:
					# convert to grayscale and find mask of non-white pixels
					gray = np.mean(img_array, axis=2)
					mask = gray < 240

					if np.any(mask):
						ys, xs = np.where(mask)
						ymin, ymax = ys.min(), ys.max()
						xmin, xmax = xs.min(), xs.max()

						# expand box slightly (5%) for natural framing
						h, w = img_array.shape[:2]
						x_pad = int(0.05 * (xmax - xmin))
						y_pad = int(0.05 * (ymax - ymin))
						xmin = max(xmin - x_pad, 0)
						xmax = min(xmax + x_pad, w)
						ymin = max(ymin - y_pad, 0)
						ymax = min(ymax + y_pad, h)

						# crop to bounding box
						cropped = img_array[ymin:ymax, xmin:xmax]

						# resize back to fit within the same aspect ratio
						crop_h, crop_w = cropped.shape[:2]
						target_ratio = orig_w / orig_h
						crop_ratio = crop_w / crop_h

						if abs(crop_ratio - target_ratio) > 0.01:
							# pad with white to maintain proportions
							if crop_ratio > target_ratio:
								# too wide - pad vertically
								new_h = int(crop_w / target_ratio)
								pad_top = (new_h - crop_h) // 2
								pad_bottom = new_h - crop_h - pad_top
								cropped = np.pad(
									cropped,
									((pad_top, pad_bottom), (0, 0), (0, 0)),
									mode="constant",
									constant_values=255,
								)
							else:
								# too high - pad horizontally
								new_w = int(crop_h * target_ratio)
								pad_left = (new_w - crop_w) // 2
								pad_right = new_w - crop_w - pad_left
								cropped = np.pad(
									cropped,
									((0, 0), (pad_left, pad_right), (0, 0)),
									mode="constant",
									constant_values=255,
								)

						# resize back to original image dimensions
						img_array = np.array(Image.fromarray(cropped).resize((orig_w, orig_h)))

				except Exception as e:
					print(f"(Zoom skipped: {e})")

				# if processing is disabled for debug, skip it
				if self.disable_processing:
					processed = img_array / 255

				else:
					# process image using enhancements
					if apply_enhancements:
						processed = apply_enhanced_prosthetic_vision(
							img_array,
							cutoff_frequency,
							gamma=None,
							gain=gain,
							shift=shift,
							show=False,
							landmarking=True,
							inverse_tone_curve=True,
							landmarking_color=(0, 0, 0),
							invert_colors=False,
							detector=self.detector
						)
					# process with no enhancements
					else:
						processed = apply_prosthetic_vision(
							img_array, cutoff_frequency, gamma=None, gain=gain, shift=shift, show=False
						)

				# prepare image for display
				display_img = Image.fromarray((processed * 255).astype(np.uint8))
				display_img.thumbnail((500, 400), Image.Resampling.LANCZOS)
				photo = ImageTk.PhotoImage(display_img)

				# clear previous content
				self.image_buttons[i].config(image="", text="")
				for widget in self.image_buttons[i].winfo_children():
					widget.destroy()

				# create a frame inside the image_button label
				container = tk.Frame(self.image_buttons[i], bg="white")
				container.pack(expand=True, fill="both")

				# create an image label
				img_label = tk.Label(container, image=photo, bd=0)
				img_label.image = photo
				img_label.pack(expand=True, fill="both")

				# add a number in the top-center
				badge = tk.Label(
					container,
					text=str(i + 1),
					font=("Arial", 16, "bold"),
					bg="#007acc",
					fg="white",
					padx=5,
					pady=2
				)
				# place the badge using absolute coordinates relative to container
				badge.place(relx=0.5, y=5, anchor="n")

		except Exception as e:
			print(f"Error processing image: {e}")
			return

	def on_key_press(self, event):
		# get the key pressed
		key = event.keysym

		# if there is no trial active, do nothing
		if not self.current_trial_config:
			return

		# if the key is a 1-4 digit, record the answer
		if key in ["1", "2", "3", "4"]:

			# convert key to an index
			idx = int(key) - 1

			# cancel the timeout
			if self.timeout_id:
				self.root.after_cancel(self.timeout_id)
				self.timeout_id = None

			# calculate response time
			response_time = time.time() - self.trial_start_time

			# check if the answer is correct
			correct = idx == self.current_trial_config["correct_answer"]

			# show debug info on GUI if debug mode is on
			if getattr(self, "debug", False):

				# remove previous label if exists
				if hasattr(self, "debug_label") and self.debug_label:
					self.debug_label.destroy()

				# set up a new label with debug info
				self.debug_label = tk.Label(
					self.root,
					text=f"{'Correct!' if correct else 'Incorrect!'}",
					font=("Arial", 16, "bold"),
					bg="yellow" if correct else "red",
					fg="black"
				)
				self.debug_label.place(relx=0.5, rely=0.05, anchor="n")  # top center

				# auto-remove after 1 second
				self.root.after(1000, lambda: self.debug_label.destroy())

			# record the answer
			self.trial_manager.record_answer(self.current_trial_config, idx, response_time, correct, timeout=False)

			# move to the next trial
			self.trial_manager.advance_trial()
			self.next_trial()

	def timeout_trial(self):
		# record wrong answer
		self.trial_manager.record_answer(self.current_trial_config, -1, 15.0, False, timeout=True)

		# move to the next trial
		self.trial_manager.advance_trial()
		self.next_trial()

	def finish_trial(self):
		# clear the GUI
		self.clear_root()

		# create a frame to display the trial complete message
		frame = tk.Frame(self.root, bg="white")
		frame.grid(row=0, column=0, sticky="nsew")
		self.root.grid_rowconfigure(0, weight=1)
		self.root.grid_columnconfigure(0, weight=1)

		# add a trial complete message
		tk.Label(
			frame,
			text=f"Trial complete! Thank you.",
			font=("Arial", 18),
			bg="white",
		).grid(row=0, column=0, sticky="nsew")

		# force the UI to update so the message is visible
		self.root.update_idletasks()

		# save the results slightly later to avoid blocking UI immediately
		self.root.after(500, self.save_results)

	def save_results(self):
		# define filename based on subject ID and current time to store the results
		filename = os.path.join(os.getcwd(), 'results', f"results_{self.subject_id}_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")

		# save the results to Excel
		self.trial_manager.save_results_to_excel(filename)

		# close the GUI
		self.root.destroy()

	def clear_root(self):
		# destroy all widgets in the root
		for w in self.root.winfo_children():
			w.destroy()


def main():
	try:
		root = tk.Tk()
		TrialGUI(root, debug=False, disable_processing=False)
		root.mainloop()
	except Exception as e:
		print(f"Error: {e}")
		sys.exit(1)


if __name__ == "__main__":
	main()
