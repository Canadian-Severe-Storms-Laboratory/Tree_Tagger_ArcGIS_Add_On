import cv2
import glob
import random
from sklearn.utils import shuffle
from tqdm import tqdm


image_files = glob.glob("Z:/tree_direction_images/*.png")
full_files = glob.glob("Z:/tree_direction_full_images/*.png")

image_files, full_files = shuffle(image_files, full_files, random_state=0)
x = 1000000
s = False

for i in tqdm(range(len(image_files))):
    image = cv2.imread(image_files[i])
    full_image = cv2.imread(full_files[i])

    if s:
        cv2.imwrite("Z:/tree_directions_split/daniel_new/tree_direction_images/img" + str(x) + ".png", image)
        cv2.imwrite("Z:/tree_directions_split/daniel_new/tree_direction_full_images/img" + str(x) + ".png", full_image)
    else:
        cv2.imwrite("Z:/tree_directions_split/emilio_new/tree_direction_images/img" + str(x) + ".png", image)
        cv2.imwrite("Z:/tree_directions_split/emilio_new/tree_direction_full_images/img" + str(x) + ".png", full_image)

    x += 1
    s = not s