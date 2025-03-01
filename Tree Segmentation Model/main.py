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
from typing import Callable, Union
import matplotlib.pyplot as plt
import keras_tuner as kt
import time
from tensorflow.keras.activations import softmax

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

def dice(targets, inputs, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(K.dot(targets, inputs))
    return (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)


def dice_loss_old(targets, inputs, smooth=1e-6):
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

def dice_loss(class_weights: Union[list, np.ndarray, tf.Tensor]) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Weighted Dice loss.
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Class weight coefficients (Union[list, np.ndarray, tf.Tensor], len=<N_CLASSES>)
    :return: Weighted Dice loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute weighted Dice loss.
        :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted Dice loss (tf.Tensor, shape=(None,))
        """
        axis_to_reduce = range(1, K.ndim(y_pred))  # Reduce all axis but first (batch)
        numerator = y_true * y_pred * class_weights  # Broadcasting
        numerator = 2. * K.sum(numerator, axis=axis_to_reduce)

        denominator = (y_true + y_pred) * class_weights # Broadcasting
        denominator = K.sum(denominator, axis=axis_to_reduce)

        return 1 - numerator / denominator

    return loss


def weighted_cross_entropy(class_weights: list, is_logits: bool = False) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
    """
    Multi-class weighted cross entropy.
        WCE(p, p̂) = −Σp*log(p̂)*class_weights
    Used as loss function for multi-class image segmentation with one-hot encoded masks.
    :param class_weights: Weight coefficients (list of floats)
    :param is_logits: If y_pred are logits (bool)
    :return: Weighted cross entropy loss function (Callable[[tf.Tensor, tf.Tensor], tf.Tensor])
    """
    if not isinstance(class_weights, tf.Tensor):
        class_weights = tf.constant(class_weights)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Computes the weighted cross entropy.
        :param y_true: Ground truth (tf.Tensor, shape=(None, None, None, None))
        :param y_pred: Predictions (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
        :return: Weighted cross entropy (tf.Tensor, shape=(<BATCH_SIZE>,))
        """
        assert len(class_weights) == y_pred.shape[-1], f"Number of class_weights ({len(class_weights)}) needs to be the same as number " \
                                                 f"of classes ({y_pred.shape[-1]})"

        if is_logits:
            y_pred = softmax(y_pred, axis=-1)

        y_pred = K.clip(y_pred, K.epsilon(), 1-K.epsilon())  # To avoid unwanted behaviour in K.log(y_pred)

        # p * log(p̂) * class_weights
        wce_loss = y_true * K.log(y_pred) * class_weights

        # Average over each data point/image in batch
        axis_to_reduce = range(1, K.ndim(wce_loss))
        wce_loss = K.mean(wce_loss, axis=axis_to_reduce)

        return -wce_loss

    return loss


def dice_wbce(true, pred, weight_zero=0.0667, weight_one=1):
    dice_score = dice_loss([1.0, 1.0])(true, pred)
    wbce_score = weighted_cross_entropy([weight_zero, weight_one])(true, pred)

    return wbce_score + dice_score


def dice_m(true, pred):
    dice_score = dice_loss([1.0, 1.0])(true, pred)
    return 1 - dice_score


def wbce_m(true, pred, weight_zero=0.0667, weight_one=1):
    wbce_score = weighted_cross_entropy([weight_zero, weight_one])(true, pred)
    return 1 - wbce_score


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


def wbce(true, pred, weight_zero=0.0667, weight_one=1):
    return 1 - weighted_bincrossentropy(true, pred, weight_zero, weight_one)


def focal_tversky_plus_wbce(true, pred):
    wbce = weighted_bincrossentropy(true, pred, weight_zero=0.05, weight_one=1)
    tversky = TverskyLoss(true, pred, alpha=0.5, beta=1.0)

    return K.pow(tversky, 0.75) + wbce

def get_model(type):
    #model = sm.Unet(type, input_shape=(256, 256, 3), classes=2, encoder_weights="imagenet")

    model = sm.Linknet(type, input_shape=(256, 256, 3), classes=2, encoder_weights="imagenet")
    # model = sm.FPN(type, input_shape=(256, 256, 3), classes=2, encoder_weights="imagenet")
    return model


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

            X.append(images[idx]) #*255
            #Y.append(np.reshape(segs[idx], (256*256, 2)))
            Y.append(np.float32(segs[idx]))

            if idx == last_idx:
                random.shuffle(indices)
                last_idx = indices[len(indices) - 1]
                zipped = itertools.cycle(indices)

        yield np.array(X), np.array(Y)


def train_model(model, x_path, y_path, x_val_path, y_val_path, save_path, batch_size=32, epochs=5, w_path=''):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.00001),
        loss=keras.losses.BinaryCrossentropy(),#dice_loss([1.0, 1.0]),
        metrics=['accuracy', keras.metrics.BinaryCrossentropy(), dice_m],
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
    Y_on_disk = np.load(y_path, mmap_mode='r')
    X_val_on_disk = np.load(x_val_path, mmap_mode='r')
    Y_val_on_disk = np.load(y_val_path, mmap_mode='r')

    print(Y_on_disk.shape)

    length = X_on_disk.shape[0]

    train_indices = list(range(X_on_disk.shape[0]))
    validation_indices = list(range(X_val_on_disk.shape[0]))

    random.seed(254)

    random.shuffle(train_indices)
    random.shuffle(validation_indices)

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

    train_spe = int(len(train_indices)/batch_size)
    val_spe = int(len(validation_indices)/batch_size)

    return model.fit_generator(image_segmentation_generator(X_on_disk, Y_on_disk, train_indices, batch_size), train_spe,
                     validation_data=image_segmentation_generator(X_val_on_disk, Y_val_on_disk, validation_indices, batch_size),
                     validation_steps=val_spe, epochs=100, callbacks=callbacks)


class CustomHyperModel(kt.HyperModel):
    def __init__(self, name=None, tunable=True):
        super().__init__(name, tunable)
        self.batch_size = None
        self.rand_seed = None

    def build(self, hp):
        # model = tf.keras.models.Sequential()
        # model.add(tf.keras.applications.ResNet50(include_top=False, input_shape=(128, 256, 3), classes=3,
        #                                          pooling=hp.Choice('pooling', ['avg', 'max']),
        #                                          weights='imagenet'))
        #
        # model.add(tf.keras.layers.Flatten())
        # model.add(tf.keras.layers.Dense(hp.Choice('units_1', [1024, 2048, 4096]), activation='relu'))
        #
        # dropoutr1 = hp.Choice('dropout_rate1', [0.1, 0.2, 0.3, 0.4])
        # dropoutr2 = hp.Choice('dropout_rate2', [0.1, 0.2, 0.3, 0.4])
        # do2 = hp.Boolean('dropout2')
        # u2 = hp.Choice('units_2', [256, 512, 1024])
        #
        # if hp.Boolean('dropout1'):
        #     model.add(tf.keras.layers.Dropout(dropoutr1))
        #
        # if hp.Boolean('dlayer2'):
        #     model.add(tf.keras.layers.Dense(u2, activation='relu'))
        #
        #     if do2:
        #         model.add(tf.keras.layers.Dropout(dropoutr2))
        #
        # model.add(tf.keras.layers.Dense(3, activation='softmax'))

        model = sm.Unet(hp.Choice('architecture', ['resnet50', 'vgg19', 'resnext50', 'seresnet50', 'seresnext50']),  , classes=2, encoder_weights="imagenet")

        learning_rate = hp.Float('lr', min_value=0.00001, max_value=0.001, sampling="log")

        losses = {"categorical_crossentropy": keras.losses.categorical_crossentropy, "dice": dice_loss([1.0, 1.0]), "wbce": wbce, "dice + wbce": dice_wbce}

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=losses.get(hp.Choice("loss_function", ["categorical_crossentropy", "dice", "wbce", "dice + wbce"])),
            metrics=['accuracy']
        )

        self.rand_seed = hp.Fixed('rseed', float(round(time.time()) % 86400))
        self.batch_size = hp.Choice('batch', [8, 16, 24])

        return model

    def fit(self, hp, model, *args, **kwargs):
        print("loading data")
        x_path = r"C:\Users\dbutt7\Documents\ml_datasets\training_images_v4.npy"
        y_path = r"C:\Users\dbutt7\Documents\ml_datasets\training_masks_v4.npy"
        x_val_path = r"C:\Users\dbutt7\Documents\ml_datasets\validation_images_v4.npy"
        y_val_path = r"C:\Users\dbutt7\Documents\ml_datasets\validation_masks_v4.npy"

        X_on_disk = np.load(x_path, mmap_mode='r')
        Y_on_disk = np.load(y_path, mmap_mode='r')
        X_val_on_disk = np.load(x_val_path, mmap_mode='r')
        Y_val_on_disk = np.load(y_val_path, mmap_mode='r')

        length = X_on_disk.shape[0]

        train_indices = list(range(X_on_disk.shape[0]))
        validation_indices = list(range(X_val_on_disk.shape[0]))

        random.seed(self.rand_seed)

        random.shuffle(train_indices)
        random.shuffle(validation_indices)

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

        train_spe = int(len(train_indices) / self.batch_size)
        val_spe = int(len(validation_indices) / self.batch_size)

        return model.fit_generator(image_segmentation_generator(X_on_disk, Y_on_disk, train_indices, self.batch_size), train_spe,
                     validation_data=image_segmentation_generator(X_val_on_disk, Y_val_on_disk, validation_indices, self.batch_size),
                     validation_steps=val_spe, **kwargs)


if __name__ == '__main__':

    # tuner = kt.BayesianOptimization(
    #     hypermodel=CustomHyperModel(),
    #     objective='val_accuracy',
    #     max_trials=45,
    #     num_initial_points=8,
    #     directory='training_results/tuner',
    #     project_name='tuner1'
    # )
    #
    # tuner.search(epochs=50, callbacks=[tf.keras.callbacks.EarlyStopping('val_accuracy', patience=5)])
    #
    # tuner.results_summary()

    experiment_name = "Link_resnet34_bcn"
    model = get_model('resnet34')

    x_path = r"C:\Users\dbutt7\Documents\ml_datasets\training_images_v4.npy"
    y_path = r"C:\Users\dbutt7\Documents\ml_datasets\training_masks_v4.npy"
    x_val_path = r"C:\Users\dbutt7\Documents\ml_datasets\validation_images_v4.npy"
    y_val_path = r"C:\Users\dbutt7\Documents\ml_datasets\validation_masks_v4.npy"
    save_path = "training_results/v4/" + experiment_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_path += "/"

    history = train_model(model, x_path, y_path, x_val_path, y_val_path, save_path, w_path="")
    #
    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    #
    # # summarize history for dice
    # plt.plot(history.history['dice'])
    # plt.plot(history.history['val_dice'])
    # plt.title('model dice')
    # plt.ylabel('dice score')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    #
    # # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
    #
