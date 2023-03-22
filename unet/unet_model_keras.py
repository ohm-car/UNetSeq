import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

class UNet(object):
	"""docstring for ClassName"""
	def __init__(self):

		super(UNet, self).__init__()
		# self.unet_model = self.create_model()
		# print(self.unet_model.summary())
		# self.args = 0
		# self.arg = arg
		
	def double_conv_block(self, x, n_filters):

		# Conv2D then ReLU activation
		conv2d1 = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")
		out1 = layers.TimeDistributed(conv2d1)(x)
		# Conv2D then ReLU activation
		conv2d2 = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")
		out2 = layers.TimeDistributed(conv2d2)(out1)
		# print("TimeDist shape:", out2.shape)

		return out2

	def downsample_block(self, x, n_filters):
		f = self.double_conv_block(x, n_filters)
		pool = layers.MaxPool2D(2)
		p1 = layers.TimeDistributed(pool)(f)

		drop = layers.Dropout(0.3)
		p = layers.TimeDistributed(drop)(p1)

		return f, p

	def upsample_block(self, x, conv_features, n_filters):
		# upsample
		conv2dT1 = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")
		out1 = layers.TimeDistributed(conv2dT1)(x)
		# print("Upsample shape:", out1.shape)
		# concatenate
		x = layers.concatenate([out1, conv_features])
		# dropout
		drop = layers.Dropout(0.3)
		x = layers.TimeDistributed(drop)(x)
		# Conv2D twice with ReLU activation
		x = self.double_conv_block(x, n_filters)

		return x

	def create_model(self, seqlen):

		 # inputs
		inputs = layers.Input(shape=(seqlen, 384,384,3))

		# encoder: contracting path - downsample
		# 1 - downsample
		f1, p1 = self.downsample_block(inputs, 64)
		# 2 - downsample
		f2, p2 = self.downsample_block(p1, 128)
		# 3 - downsample
		f3, p3 = self.downsample_block(p2, 256)
		# 4 - downsample
		f4, p4 = self.downsample_block(p3, 512)

		# 5 - bottleneck
		bottleneck = self.double_conv_block(p4, 1024)

		# decoder: expanding path - upsample
		# 6 - upsample
		u6 = self.upsample_block(bottleneck, f4, 512)
		# 7 - upsample
		u7 = self.upsample_block(u6, f3, 256)
		# 8 - upsample
		u8 = self.upsample_block(u7, f2, 128)
		# 9 - upsample
		u9 = self.upsample_block(u8, f1, 64)
		# print("U9:", u9.shape)

		# u10 = layers.Reshape((2,384,384,32))(u9)
		# print(u10.shape)

		u11 = layers.ConvLSTM2D(16, kernel_size=3, padding='same')(u9)
		# print("U11:", u11.shape)

		# outputs
		outputs = layers.Conv2D(1, 3, padding="same", activation = "sigmoid")(u11)
		# print("outputs:", outputs.shape)

		# unet model with Keras Functional API
		unet_model = tf.keras.Model(inputs, outputs, name="Sequential_U-Net")

		return unet_model


# print("Test model")
# model = create_model()
# print(model.summary())
# print("Done")