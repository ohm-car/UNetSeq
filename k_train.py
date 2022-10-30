import tensorflow as tf
# from utils.k_dataset import DatasetTrial
from unet.unet_model_keras import UNet
from utils.k_datasetF import DatasetUSound
# import skimage

# dataset = DatasetTrial()
# data_tr = dataset.get_train_dataset()
# data_ts = dataset.get_test_dataset()
# info = dataset.get_info()

dataset = DatasetUSound()

imgs, masks = dataset.load_dataset()

unet_model = UNet().create_model()

print(len(imgs))
print(len(masks))

# print(type(unet_model))
# unet_model.build(input_shape = (128,128,3))
unet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss="sparse_categorical_crossentropy",
                  metrics="accuracy")

# print(unet_model.summary())

# BATCH_SIZE = 16
# BUFFER_SIZE = 1000
# train_batches = data_tr.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
# train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
# validation_batches = data_ts.take(3000).batch(BATCH_SIZE)
# test_batches = data_ts.skip(3000).take(669).batch(BATCH_SIZE)

# NUM_EPOCHS = 20

# TRAIN_LENGTH = info.splits["train"].num_examples
# STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

# VAL_SUBSPLITS = 5
# TEST_LENTH = info.splits["test"].num_examples
# VALIDATION_STEPS = TEST_LENTH // BATCH_SIZE // VAL_SUBSPLITS

# print(type(info))

model_history = unet_model.fit(
    x=imgs,
    y=masks,
    batch_size=2,
    epochs=1,
    verbose='auto',
    callbacks=None,
    validation_split=None,
    validation_data=(imgs, masks),
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
