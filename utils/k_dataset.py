import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

class DatasetTrial(object):
	"""docstring for ClassName"""
	def __init__(self):
		super(DatasetTrial, self).__init__()
		self.dataset, self.info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
		print(self.info)
		
		# self.arg = arg
		

	def resize(input_image, input_mask):
		input_image = tf.image.resize(input_image, (128, 128), method="nearest")
		input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")

		return input_image, input_mask

	def augment(input_image, input_mask):
		if tf.random.uniform(()) > 0.5:
			# Random flipping of the image and mask
			input_image = tf.image.flip_left_right(input_image)
			input_mask = tf.image.flip_left_right(input_mask)

		return input_image, input_mask

	def normalize(input_image, input_mask):
		input_image = tf.cast(input_image, tf.float32) / 255.0
		input_mask -= 1
		return input_image, input_mask

	def load_image_train(datapoint):
		input_image = datapoint["image"]
		input_mask = datapoint["segmentation_mask"]
		input_image, input_mask = resize(input_image, input_mask)
		input_image, input_mask = augment(input_image, input_mask)
		input_image, input_mask = normalize(input_image, input_mask)

		return input_image, input_mask

	def load_image_test(datapoint):
		input_image = datapoint["image"]
		input_mask = datapoint["segmentation_mask"]
		input_image, input_mask = resize(input_image, input_mask)
		input_image, input_mask = normalize(input_image, input_mask)

		return input_image, input_mask

	def get_train_dataset():
		train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)

		return train_dataset

	def get_test_dataset():
		test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

		return test_dataset

# train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
# train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
# test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)