#import itertools
#import math
import random
#import sys
import os
from glob import glob

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
#os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import csv
#import segmentation_models as sm
from tqdm import tqdm
#import tensorflow as tf
#import keras


def create_mask(csv_path, image):
    f = open(csv_path)
    reader = csv.reader(f)
    reader.__next__()
    shapes = []
    poly_lines = []
    for row in reader:
        strs = row[5].split(':')
        x = strs[2][1:-16].split(',')
        y = strs[3][1:-2].split(',')

        shape = []
        for i in range(len(x)):
            shape.append([x[i], y[i]])

        if strs[1].find('polygon') != -1:
            shapes.append(shape)
        else:
            poly_lines.append(shape)

    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for shape in shapes:
        cv2.drawContours(mask, [np.array(shape, int)], -1, 255, -1)

    for line in poly_lines:
        cv2.polylines(mask, [np.array(line, int)], False, 255, 2)
    #kernel = np.ones((5, 5), np.uint8)
    #cv2.erode(mask, kernel, mask, iterations=2)
    cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY, mask)
    #cv2.imwrite("67-manual-mask_erode52.png", mask)

    return mask

def split_and_augment(image, mask, count, images_folder, masks_folder):

    for k in range(4):
        for y in range(int((image.shape[0] - k*64) / 256.0)):
            for x in range(int((image.shape[1] - k*64) / 256.0)):
                image_section = image[y*256 + k*64:y*256 + k*64 + 256, x*256 + k*64:x*256 + k*64 + 256]
                mask_section = mask[y*256 + k*64:y*256 + k*64 + 256, x*256 + k*64:x*256 + k*64 + 256]

                cv2.imwrite(images_folder + "img" + str(count) + ".png", image_section)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", mask_section)
                count += 1

                #flip vertical
                aug_img = cv2.flip(image_section, 0)
                aug_mask = cv2.flip(mask_section, 0)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

                # flip horizontal
                aug_img = cv2.flip(image_section, 1)
                aug_mask = cv2.flip(mask_section, 1)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

                #rotate 90 clockwise
                aug_img = cv2.rotate(image_section, cv2.ROTATE_90_CLOCKWISE)
                aug_mask = cv2.rotate(mask_section, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

                #rotate 180
                aug_img = cv2.rotate(image_section, cv2.ROTATE_180)
                aug_mask = cv2.rotate(mask_section, cv2.ROTATE_180)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

                # rotate 90 counterclockwise
                aug_img = cv2.rotate(image_section, cv2.ROTATE_90_COUNTERCLOCKWISE)
                aug_mask = cv2.rotate(mask_section, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

                # top left diagonal
                aug_img = cv2.flip(image_section, 0)
                aug_mask = cv2.flip(mask_section, 0)
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

                # top right diagonal
                aug_img = cv2.flip(image_section, 0)
                aug_mask = cv2.flip(mask_section, 0)
                aug_img = cv2.rotate(aug_img, cv2.ROTATE_90_CLOCKWISE)
                aug_mask = cv2.rotate(aug_mask, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_img)
                cv2.imwrite(masks_folder + "img" + str(count) + ".png", aug_mask)
                count += 1

    return count


def hue_augment(image_paths, mask_paths, count, images_folder, masks_folder):
    print("Hue Aug")
    for i in tqdm(range(len(image_paths))):
        image = cv2.imread(image_paths[i])
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
        aug_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)

        aug_image += np.array([h, 0, 0], np.uint8)
        aug_image = cv2.cvtColor(aug_image, cv2.COLOR_HSV2BGR)

        cv2.imwrite(images_folder + "img" + str(count) + ".png", aug_image)
        cv2.imwrite(masks_folder + "img" + str(count) + ".png", mask)
        count += 1


if __name__ == '__main__':

    # csv_paths = sorted(glob("validation_tagged_csvs/*.csv"))
    # image_paths = sorted(glob("validation_tagged_images/*.bmp"))
    #
    # print(csv_paths, image_paths)

    images_folder = "manual_dataset_images_v3/"
    masks_folder = "manual_dataset_masks_v3/"

    # image = cv2.imread("manually_tagged_images/7.bmp")
    # csv_path = "manually_tagged_csvs/7.csv"
    #
    # mask = create_mask(csv_path, image)
    # cv2.imwrite("mask7.png", mask)

    # images = glob(images_folder + "*.png")
    # masks = glob(masks_folder + "*.png")

    # # random.seed(420)
    # # a = random.randint(0, len(images)-1)
    # #
    # # image = cv2.imread(images[a])
    # # mask = cv2.imread(masks[a], cv2.IMREAD_GRAYSCALE)
    # #
    # # cv2.imshow("image", image)
    # # cv2.imshow("mask", mask)
    # # cv2.waitKey(0)
    #
    # count = 0
    # #
    # for i in tqdm(range(len(csv_paths))):
    #     image = cv2.imread(image_paths[i])
    #     mask = create_mask(csv_path=csv_paths[i], image=image)
    # # image = cv2.imread("manually_tagged_images/22_Lac_Ombon_283000_5221000.png")
    # # mask = np.zeros((image.shape[0], image.shape[1]))
    #
    #     cv2.imwrite("validation_masks/" + str(i) + ".png", mask)
    #
    #     count = split_and_augment(image=image, mask=mask, count=count,
    #                               images_folder=images_folder, masks_folder=masks_folder)
    #
    # image_paths = glob(images_folder + "*.png")
    # mask_paths = glob(masks_folder + "*.png")
    #
    # hue_augment(image_paths, mask_paths, count, images_folder=images_folder, masks_folder=masks_folder)

    image_paths = glob(images_folder + "*.png")
    mask_paths = glob(masks_folder + "*.png")

    images = []
    masks = []

    for path in tqdm(image_paths):
        image = cv2.imread(path)
        image = np.array(image, np.float32)
        image /= 255.0

        images.append(image)

    print("converting to np")
    images = np.array(images)

    print("saving images as npy")
    np.save("C:/Users/dbutt7/Documents/ml_datasets/training_images_v3.npy", images)

    for path in tqdm(mask_paths):
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask, np.float32)
        mask /= 255.0
        mask = np.array(mask, np.uint8)
        seg_mask = np.zeros((256, 256, 2), np.uint8)

        for c in range(2):
            seg_mask[:, :, c] = (mask == c).astype(np.uint8)

        masks.append(seg_mask)

    print("converting to np")
    masks = np.array(masks)

    print("saving masks as npy")
    np.save("C:/Users/dbutt7/Documents/ml_datasets/training_masks_v3.npy", masks)

