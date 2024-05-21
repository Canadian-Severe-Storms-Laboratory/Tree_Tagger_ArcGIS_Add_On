import cv2
import glob
import random
from sklearn.utils import shuffle
from tqdm import tqdm

image_files = glob.glob("F:/tree_directons/daniel_new/tree_direction_images/*.png")
full_files = glob.glob("F:/tree_directons/daniel_new/tree_direction_full_images/*.png")
left = glob.glob("F:/tree_directons/daniel_new/labelled_tree_directions/left/*.png")
right = glob.glob("F:/tree_directons/daniel_new/labelled_tree_directions/right/*.png")
inc = glob.glob("F:/tree_directons/daniel_new/labelled_tree_directions/inc/*.png")

previous = left + right + inc

for file in tqdm(previous):
    split = file.split('\\')
    name = split[len(split) - 1]
    image_files.remove("F:/tree_directons/daniel_new/tree_direction_images\\" + name)
    full_files.remove("F:/tree_directons/daniel_new/tree_direction_full_images\\" + name)

image_files, full_files = shuffle(image_files, full_files, random_state=0)

for i in range(len(image_files)):
    image_file = image_files[i]
    full_file = full_files[i]
    split = image_file.split('\\')
    name = split[len(split) - 1]
    image = cv2.imread(image_file)
    big_image = cv2.resize(image, (1024, 512), cv2.INTER_CUBIC)
    full_image = cv2.imread(full_file)
    full_image = cv2.resize(full_image, (full_image.shape[1]*2, full_image.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("image", big_image)
    cv2.imshow("full_image", full_image)

    while True:
        key = cv2.waitKeyEx()

        #left
        if key == 2424832:
            cv2.imwrite("F:/tree_directons/daniel_new/labelled_tree_directions/left/" + name, image)
            print("left - " + name)
            break

        #right
        if key == 2555904:
            cv2.imwrite("F:/tree_directons/daniel_new/labelled_tree_directions/right/" + name, image)
            print("right - " + name)
            break

        #up (inc)
        if key == 2490368:
            cv2.imwrite("F:/tree_directons/daniel_new/labelled_tree_directions/inc/" + name, image)
            print("inc - " + name)
            break


