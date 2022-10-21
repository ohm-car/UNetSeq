import tensorflow as tf
from utils.k_dataset import DatasetTrial
from unet.unet_model_keras import UNet

unet_model = UNet().create_model()
dataset = DatasetTrial()

print(type(unet_model))
# unet_model.build(input_shape = (128,128,3))
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

print(unet_model.summary())

# NUM_EPOCHS = 20

# TRAIN_LENGTH = info.splits["train"].num_examples
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# VAL_SUBSPLITS = 5
# TEST_LENTH = info.splits["test"].num_examples
# VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS

# model_history = unet_model.fit(train_batches,
#                               epochs=NUM_EPOCHS,
#                               steps_per_epoch=STEPS_PER_EPOCH,
#                               validation_steps=VALIDATION_STEPS,
#                               validation_data=test_batches)

