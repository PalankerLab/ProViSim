# general imports
import os
from matplotlib import pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# project imports
import facial_landmarking
from utils import crop_to_square
from image_processing import rgb_to_gray, create_tukey_window, fft_filter, sigmoid_transform, gamma_transform, \
	quantize_image, inverse_sigmoid


def apply_resolution_reduction(image, cutoff_frequency, show=True):
	# create a low-pass filter window based on image size and cutoff frequency
	tukey_win = create_tukey_window(cutoff_frequency, image.shape[0])

	# apply resolution reduction
	reduced_res_img = fft_filter(image, tukey_win)

	# plot the reduced resolution image, if desired
	if show:
		plt.imshow(reduced_res_img, cmap='gray')
		plt.axis('off')
		plt.show()

	# return the reduced-resolution image
	return reduced_res_img

def apply_contrast_reduction(image, gamma=None, gain=None, shift=None, show=True):
	if gain is not None and shift is not None:
		reduced_con_img = sigmoid_transform(image, gain, shift)
	elif gamma is not None:
		reduced_con_img = gamma_transform(image, gamma)
	else:
		reduced_con_img = image

	# plot the contrast reduced image, if desired
	if show:
		plt.imshow(reduced_con_img, cmap='gray')
		plt.axis('off')
		plt.show()

	# return the contrast reduced image
	return reduced_con_img

def apply_prosthetic_vision(image, cutoff_frequency, gamma=None, gain=None, shift=None, show=True):
	# if the image is not square, crop it to square
	image = crop_to_square(image)

	# convert image to grayscale
	gray_img = rgb_to_gray(image)

	# apply resolution reduction
	reduced_res_img = apply_resolution_reduction(gray_img, cutoff_frequency, show=show)

	# apply contrast reduction
	reduced_res_and_con_img = apply_contrast_reduction(reduced_res_img, gamma=gamma, gain=gain, shift=shift, show=show)

	# plot the final result, if desired
	if show:
		plt.imshow(reduced_res_and_con_img, cmap='gray')
		plt.axis('off')
		plt.show()

	# return the final processed image
	return reduced_res_and_con_img

def apply_enhanced_prosthetic_vision(image, cutoff_frequency, gamma=None, gain=None, shift=None, show=False,
									 landmarking=True, inverse_tone_curve=True, landmarking_color=None,
									 invert_colors=False, detector=None):

	# initialize the preprocessed image
	preprocessed_image = None

	# run preemptive facial landmarking, if desired
	if landmarking:
		# initialize flag to close detector
		close_detector = False if detector else True

		if detector is None:
			# create a FaceLandmarker object
			model_asset_path = os.path.join(os.path.dirname(__file__), 'face_landmarker_v2_with_blendshapes.task')
			base_options = python.BaseOptions(model_asset_path=model_asset_path, delegate=python.BaseOptions.Delegate.CPU)
			options = vision.FaceLandmarkerOptions(base_options=base_options,
												   output_face_blendshapes=False,
												   output_facial_transformation_matrixes=False,
												   num_faces=1)

			# run detection
			detector = vision.FaceLandmarker.create_from_options(options)

		# recreate the input image with thickened features
		preprocessed_image = facial_landmarking.draw_facial_landmarks_colored(
			image,
			detector,
			draw_eyebrows=True,
			draw_irises=True,
			draw_lips=True,
			cutoff_frequency=cutoff_frequency,
			show=show,
			landmarking_color=landmarking_color,
		)

		# close detector if needed
		if close_detector:
				detector.close()

	# convert the image to grayscale
	if preprocessed_image is not None:
		preprocessed_image = rgb_to_gray(preprocessed_image, invert_colors)
	else:
		preprocessed_image = rgb_to_gray(image, invert_colors)

	# apply an inverse tone curve
	if inverse_tone_curve:

		if gamma is not None:
			preprocessed_image = gamma_transform(preprocessed_image, 1 / gamma)
		elif gain is not None and shift is not None:
			preprocessed_image = inverse_sigmoid(preprocessed_image, gain, shift)

	# add image quantization to simulate the limited range of the DMD system
	preprocessed_image = quantize_image(preprocessed_image)

	# apply prosthetic vision with both resolution and contrast reduction
	prosthetic_image = apply_prosthetic_vision(preprocessed_image, cutoff_frequency, gamma=gamma, gain=gain,
											   shift=shift, show=show)

	return prosthetic_image