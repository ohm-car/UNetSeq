import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# from utils.k_dataset import DatasetTrial
from unet.unet_model_keras import UNet
from utils.k_datasetG import DatasetUSound
# import skimage

# dataset = DatasetTrial()
# data_tr = dataset.get_train_dataset()
# data_ts = dataset.get_test_dataset()
# info = dataset.get_info()

def load_sequences_lists(seqlen):

        trainseq = list()
        trainmseq = list()
        valseq = list()
        valmseq = list()

        for i in range(2694):

            if((i - 5) % 6 == 0):
                
                tmpseq = list()
                for j in range(seqlen//2):
                    k = i - 6*(seqlen//2  - j)
                    tmpseq.append(k if k >=0 else (k + 6*(int(abs(k/6)) + 1)))
                tmpseq.append(i)
                for j in range(seqlen//2):
                    k = i + 6*(j+1)
                    if(i <= 2675):
                        tmpseq.append(k if k <= 2693 else (k - 6*(int(abs((k-2693)/6)) + 1)))
                    else:
                        tmpseq.append(k if k <= 2693 else (k - 6*(int(abs((k-2693)/6)))))

                valseq.append(tmpseq)
                valmseq.append(i)

            else:
                
                tmpseq = list()
                for j in range(seqlen//2):
                    k = i - 6*(seqlen//2  - j)
                    if(i%6 == 0):
                        tmpseq.append(k if k >=0 else (k + 6*(int(abs(k/6)))))
                    else:
                        tmpseq.append(k if k >=0 else (k + 6*(int(abs(k/6)) + 1)))
                tmpseq.append(i)
                for j in range(seqlen//2):
                    k = i + 6*(j+1)
                    tmpseq.append(k if k <= 2693 else (k - 6*(int(abs((k-2693)/6)) + 1)))

                trainseq.append(tmpseq)
                trainmseq.append(i)

        return trainseq, valseq, trainmseq, valmseq

trSeq, valSeq, trMasks, valMasks = load_sequences_lists(5)

# imageDir = '/home/omkar/ArteryProj/data/Img_All_Squared/'
# masksDir = '/home/omkar/ArteryProj/data/Masks_All_Squared/'
# checkpoint_path = "/home/omkar/ArteryProj/UNetSeq/checkpoints/model_{epoch:03d}"

imageDir = '/nfs/ada/oates/users/omkark1/ArteryProj/data/Img_All_Squared/'
masksDir = '/nfs/ada/oates/users/omkark1/ArteryProj/data/Masks_All_Squared/'
checkpoint_path = "/nfs/ada/oates/users/omkark1/ArteryProj/UNetSeq/checkpoints/model_{epoch:03d}"

seqlen = 9

train_gen = DatasetUSound(4, imageDir, masksDir, trSeq, trMasks, seqlen)
val_gen = DatasetUSound(4, imageDir, masksDir, valSeq, valMasks, seqlen)

# dataset = DatasetUSound()
print(train_gen.__class__.__bases__)
print(train_gen.__getitem__(14)[0].shape)
print(train_gen.__getitem__(14)[1].shape)
# imgs, masks = dataset.load_dataset()

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():

    unet_model = UNet().create_model(seqlen = seqlen)

    # print(len(imgs))
    # print(len(masks))

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

#create callbacks

callbacks = [
            # keras.callbacks.TensorBoard(log_dir=self.log_dir,
            #                             histogram_freq=0, write_graph=True, write_images=False),
            tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                            verbose=0, save_weights_only=False, save_freq = 5*train_gen.__len__()),
        ]

# model_history = unet_model.fit(
#     x=train_gen,
#     batch_size=1,
#     epochs=1,
#     verbose='auto',
#     callbacks=None,
#     validation_split=None,
#     validation_data=(imgs, masks),
#     shuffle=True,
#     class_weight=None,
#     sample_weight=None,
#     initial_epoch=0,
#     steps_per_epoch=None,
#     validation_steps=None,
#     validation_batch_size=None,
#     validation_freq=1,
#     max_queue_size=10,
#     workers=1,
#     use_multiprocessing=False
# )

model_history = unet_model.fit(
    x=train_gen,
    # batch_size=1,
    epochs=80,
    verbose='auto',
    callbacks=callbacks,
    # validation_split=None,
    validation_data=val_gen,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    # validation_steps=None,
    # validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)

# unet_model.save('trialModel')
# unet_model.save('trm1', save_format='h5')
# m2 = tf.keras.models.load_model('trialModel')