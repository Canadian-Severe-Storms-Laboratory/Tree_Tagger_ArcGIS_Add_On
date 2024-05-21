import itertools
import os
from glob import glob
import random
from fastcore.foundation import L

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import segmentation_models as sm
from tqdm import tqdm
import tensorflow as tf
import keras
keras.backend.set_image_data_format('channels_last')
from keras.callbacks import Callback
from keras.callbacks import CSVLogger
from keras import backend as K
import keras_tuner as kt
import matplotlib.pyplot as plt


def jaccard_distance(y_true, y_pred, smooth=1e-6):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1 - jac


def TverskyLoss(targets, inputs, alpha=0.367879441, beta=2.718281828, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return 1 - Tversky

def FocalTverskyLoss(targets, inputs, alpha=0.367879441, beta=2.718281828, gamma=0.75, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    FocalTversky = K.pow((1 - Tversky), gamma)

    return FocalTversky


def tversky(targets, inputs, alpha=0.367879441, beta=2.718281828, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    # True Positives, False Positives & False Negatives
    TP = K.sum((inputs * targets))
    FP = K.sum(((1 - targets) * inputs))
    FN = K.sum((targets * (1 - inputs)))

    Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    return Tversky

def focal_tversky(targets, inputs, alpha=0.367879441, beta=2.718281828, smooth=1e-6):
    return K.pow(tversky(targets, inputs, alpha, beta, smooth), 0.75)

# def dice(y_pred, y_true):
#     intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
#     union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
#     # if y_pred.sum() == 0 and y_pred.sum() == 0:
#     #     return 1.0
#
#     return 2*intersection / union
#
# def dice_loss(y_pred, y_true):
#     intersection = K.sum(K.sum(K.abs(y_true * y_pred), axis=-1))
#     union = K.sum(K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1))
#
#     return 1 - 2*intersection / union


def dice(targets, inputs, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    return (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)


def dice_loss(targets, inputs, smooth=1e-6):
    # flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def f1_loss(y_true, y_pred):
    true_positives = K.sum(K.clip(y_true * y_pred, 0, 1))
    possible_positives = K.sum(K.clip(y_true, 0, 1))
    predicted_positives = K.sum(K.clip(y_pred, 0, 1))

    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 1 - 2*((precision*recall)/(precision+recall+K.epsilon()))


def weighted_bincrossentropy(true, pred, weight_zero=0.01, weight_one=1):
    """
    Calculates weighted binary cross entropy. The weights are fixed.

    This can be useful for unbalanced catagories.

    Adjust the weights here depending on what is required.

    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives
        will be penalize 10 times as much as false negatives.
    """

    # calculate the binary cross entropy
    bin_crossentropy = keras.backend.binary_crossentropy(true, pred)

    # apply the weights
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return keras.backend.mean(weighted_bin_crossentropy)


def wbce(true, pred, weight_zero=0.01, weight_one=1):
    return 1 - weighted_bincrossentropy(true, pred, weight_zero, weight_one)


def focal_tversky_plus_wbce(true, pred):
    wbce = weighted_bincrossentropy(true, pred, weight_zero=0.05, weight_one=1)
    tversky = TverskyLoss(true, pred, alpha=0.5, beta=1.0)

    return K.pow(tversky, 0.75) + wbce

class CheckpointsCallback(Callback):
    def __init__(self, checkpoints_path):
        super().__init__()
        self.checkpoints_path = checkpoints_path

    def on_epoch_end(self, epoch, logs=None):
        if self.checkpoints_path is not None:
            self.model.save_weights(self.checkpoints_path + "weights" + str(epoch+1) + ".h5")
            print("saved ", self.checkpoints_path + "weights" + str(epoch+1) + ".h5")


def image_segmentation_generator(images, segs, indices, batch_size):

    random.shuffle(indices)
    last_idx = indices[len(indices) - 1]

    zipped = itertools.cycle(indices)

    while True:
        X = []
        Y = []

        for i in range(batch_size):
            idx = next(zipped)

            X.append(images[idx])
            #Y.append(np.reshape(segs[idx], (256*256, 2)))
            Y.append(segs[idx])
            if idx == last_idx:
                random.shuffle(indices)
                last_idx = indices[len(indices) - 1]
                zipped = itertools.cycle(indices)

        yield np.array(X), np.array(Y)


def train_model(model, x_path, x_val_path, y_path, y_val_path, save_path, batch_size=16, epochs=5, w_path=''):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.000005),
        loss=f1_loss,
        metrics=['accuracy', precision_m, recall_m, f1_m]
    )

    csv_logger = CSVLogger(save_path + "model_history_log.csv", append=True)

    callbacks = [
        CheckpointsCallback(save_path),
        csv_logger
    ]

    if w_path != '':
        model.load_weights(w_path)

    print("loading data")
    X_on_disk = np.load(x_path, mmap_mode='r')
    X_val_on_disk = np.load(x_val_path, mmap_mode='r')
    Y_on_disk = np.load(y_path, mmap_mode='r')
    Y_val_on_disk = np.load(y_val_path, mmap_mode='r')

    length = X_on_disk.shape[0]

    train_indices = list(range(X_on_disk.shape[0]))
    validation_indices = list(range(X_val_on_disk.shape[0]))

    random.seed(257)

    random.shuffle(train_indices)
    random.shuffle(validation_indices)

    #
    # print("splitting validation")
    # for i in tqdm(range(length)):
    #     if random.randint(0, 4) == 0:
    #         validation_indices.append(i)
    #     else:
    #         train_indices.append(i)

    print("itemifying data")
    X_on_disk = L(*X_on_disk)
    Y_on_disk = L(*Y_on_disk)
    X_val_on_disk = L(*X_val_on_disk)
    Y_val_on_disk = L(*Y_val_on_disk)

    # a = 1616
    #
    # cv2.imshow("test1", (X_on_disk[a]*255.0).astype(np.uint8))
    # cv2.imshow("test2", (X_on_disk[a + 1] * 255.0).astype(np.uint8))
    # cv2.imshow("test3", (X_on_disk[a + 2] * 255.0).astype(np.uint8))
    # cv2.imshow("test4", (X_on_disk[a + 3] * 255.0).astype(np.uint8))
    # cv2.imshow("test5", (X_on_disk[a + 4] * 255.0).astype(np.uint8))
    # cv2.imshow("test6", (X_on_disk[a + 5] * 255.0).astype(np.uint8))
    # cv2.imshow("test7", (X_on_disk[a + 6] * 255.0).astype(np.uint8))
    # cv2.imshow("test8", (X_on_disk[a + 7] * 255.0).astype(np.uint8))
    # cv2.waitKey(0)

    train_spe = int(len(train_indices)/batch_size)/4.0
    val_spe = int(len(validation_indices)/batch_size)

    return model.fit_generator(image_segmentation_generator(X_on_disk, Y_on_disk, train_indices, batch_size), train_spe,
                     validation_data=image_segmentation_generator(X_val_on_disk, Y_val_on_disk, validation_indices, batch_size),
                     validation_steps=val_spe, epochs=100, callbacks=callbacks)


if __name__ == '__main__':

    experiment_name = "v3_small_epoch_f1"

    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 256, 3)))
    # model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.AveragePooling2D((2, 2)))
    # model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dense(3, activation='softmax'))

    back_bone = tf.keras.applications.VGG19(include_top=False, input_shape=(128, 256, 3), classes=3, pooling='avg', weights='imagenet')
    back_bone.trainable = True

    model.add(back_bone)

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(4096, activation='relu'))
    # model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    x_path = r"Z:/Tree_Direction_Images_v3.npy"
    x_val_path = r"Z:/Tree_Direction_Val_Images_v3.npy"
    y_path = r"Z:/Tree_Direction_Labels_v3.npy"
    y_val_path = r"Z:/Tree_Direction_Val_Labels_v3.npy"
    save_path = "training_results/VGG19/" + experiment_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path += "/"

    history = train_model(model, x_path, x_val_path, y_path, y_val_path, save_path, batch_size=64, w_path="")

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for dice
    plt.plot(history.history['dice'])
    plt.plot(history.history['val_dice'])
    plt.title('model dice')
    plt.ylabel('dice score')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

