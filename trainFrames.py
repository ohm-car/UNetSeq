import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
# from utils.k_dataset import DatasetTrial
from unet.unet_model_keras import UNet
from utils.datasetUSoundSeq import DatasetUSound
# import skimage
from pathlib import Path

# dataset = DatasetTrial()
# data_tr = dataset.get_train_dataset()
# data_ts = dataset.get_test_dataset()
# info = dataset.get_info()

def load_sequences_lists(seqlen, vid_ids):

    seq_list = list()
    mask_list = list()

    for vid_id in vid_ids:

        for i in range(1, 450):

            seq = list()

            for j in range((i - seqlen//2), (i + seqlen//2) + 1):
                seq.append(j)

            for j in range(len(seq)):
                if(seq[j] < 1):
                    seq[j] = vid_id + 'f1'
                elif(seq[j] > 449):
                    seq[j] = vid_id + 'f449'
                else:
                    seq[j] = vid_id + 'f' + str(seq[j])
            seq_list.append(seq)
            mask_list.append(vid_id + 'f' + str(i))

    return seq_list, mask_list


if __name__ == '__main__':

    seqlen = 9
    BATCH_SIZE = 4

    # seq_list, mask_list = load_sequences_lists(seqlen)

    rootDir = Path(__file__).resolve().parent.parent
    print(rootDir)
    imageDir = os.path.join(rootDir, 'data/Img_All_Squared/')
    masksDir = os.path.join(rootDir, 'data/Masks_All_Squared/')
    checkpoint_path = os.path.join(rootDir, 'UNetSeq/checkpoints/model_{epoch:03d}')

    # imageDir = '/nfs/ada/oates/users/omkark1/ArteryProj/data/Img_All_Squared/'
    # masksDir = '/nfs/ada/oates/users/omkark1/ArteryProj/data/Masks_All_Squared/'
    # checkpoint_path = "/nfs/ada/oates/users/omkark1/ArteryProj/UNetSeq/checkpointsR4/model_{epoch:03d}"

    train_vid_ids = ['v1', 'v2', 'v3', 'v4', 'v6']
    val_vid_ids = ['v5']

    print("Train Videos: ", train_vid_ids)
    print("Val Videos: ", val_vid_ids)
    print("Sequence Length: ", seqlen)
    print("Batch Size: ", BATCH_SIZE)

    # print(load_sequences_lists(3, train_vid_ids))

    train_seq, train_mask = load_sequences_lists(seqlen, train_vid_ids)
    val_seq, val_mask = load_sequences_lists(seqlen, val_vid_ids)

    train_gen = DatasetUSound(BATCH_SIZE, imageDir, masksDir, train_seq, train_mask, seqlen)
    val_gen = DatasetUSound(BATCH_SIZE, imageDir, masksDir, val_seq, val_mask, seqlen)

    # dataset = DatasetUSound()
    print(train_gen.__class__.__bases__)
    # print(train_gen.__getitem__(14)[0].shape)
    # print(train_gen.__getitem__(14)[1].shape)
    # print(train_gen.__getitem__(14)[1])
    # imgs, masks = dataset.load_dataset()

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # with strategy.scope():

    unet_model = UNet().create_model(seqlen = seqlen)

    # print(len(imgs))
    # print(len(masks))

    # print(type(unet_model))
    # unet_model.build(input_shape = (128,128,3))
    unet_model.compile(optimizer=tf.keras.optimizers.Adam(clipvalue=0.2),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics="accuracy")

    print(unet_model.summary())

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
        verbose=1,
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
