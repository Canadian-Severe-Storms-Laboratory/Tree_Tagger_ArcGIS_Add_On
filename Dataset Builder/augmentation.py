import cv2
import glob
import random
import numpy as np
from tqdm.auto import tqdm

images = []
val_images = []
labels = []
val_labels = []

daniel_inc = glob.glob("F:/tree_directions/daniel/labelled_tree_directions/inc/*.png")
daniel_left = glob.glob("F:/tree_directions/daniel/labelled_tree_directions/left/*.png")
daniel_right = glob.glob("F:/tree_directions/daniel/labelled_tree_directions/right/*.png")

daniel_new_inc = glob.glob("F:/tree_directions/daniel_new/labelled_tree_directions/inc/*.png")
daniel_new_left = glob.glob("F:/tree_directions/daniel_new/labelled_tree_directions/left/*.png")
daniel_new_right = glob.glob("F:/tree_directions/daniel_new/labelled_tree_directions/right/*.png")

emilio_inc = glob.glob("F:/tree_directions/emilio/labelled_tree_directions/inc/*.png")
emilio_left = glob.glob("F:/tree_directions/emilio/labelled_tree_directions/left/*.png")
emilio_right = glob.glob("F:/tree_directions/emilio/labelled_tree_directions/right/*.png")

emilio_new_inc = glob.glob("F:/tree_directions/emilio_new/labelled_tree_directions/inc/*.png")
emilio_new_left = glob.glob("F:/tree_directions/emilio_new/labelled_tree_directions/left/*.png")
emilio_new_right = glob.glob("F:/tree_directions/emilio_new/labelled_tree_directions/right/*.png")

inc = daniel_inc + daniel_new_inc + emilio_inc + emilio_new_inc
left = daniel_left + daniel_new_left + emilio_left + emilio_new_left
right = daniel_right + daniel_new_right + emilio_right + emilio_new_right

for image_path in tqdm(right):
    image = cv2.imread(image_path)

    if(random.randint(1,5) == 1):
        val_images.append(image)
        val_labels.append(([0, 0, 1]))
    else:
        images.append(image)
        labels.append([0, 0, 1])

        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 0, 1])

        #flip horizontal
        image_fh = cv2.flip(image, 1)
        images.append(image_fh)
        labels.append([1, 0, 0])

        hue_image = cv2.cvtColor(image_fh, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([1, 0, 0])

        #flip vertical
        image_fv = cv2.flip(image, 0)
        images.append(image_fv)
        labels.append([0, 0, 1])

        hue_image = cv2.cvtColor(image_fv, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 0, 1])

        #flip both
        image_fvh = cv2.flip(image_fv, 1)
        images.append(image_fvh)
        labels.append([1, 0, 0])

        hue_image = cv2.cvtColor(image_fvh, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([1, 0, 0])

for image_path in tqdm(left):
    image = cv2.imread(image_path)

    if (random.randint(1, 5) == 1):
        val_images.append(image)
        val_labels.append(([1, 0, 0]))
    else:
        images.append(image)
        labels.append([1, 0, 0])

        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([1, 0, 0])

        #flip horizontal
        image_fh = cv2.flip(image, 1)
        images.append(image_fh)
        labels.append([0, 0, 1])

        hue_image = cv2.cvtColor(image_fh, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 0, 1])

        #flip vertical
        image_fv = cv2.flip(image, 0)
        images.append(image_fv)
        labels.append([1, 0, 0])

        hue_image = cv2.cvtColor(image_fv, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([1, 0, 0])

        #flip both
        image_fvh = cv2.flip(image_fv, 1)
        images.append(image_fvh)
        labels.append([0, 0, 1])

        hue_image = cv2.cvtColor(image_fvh, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 0, 1])

for image_path in tqdm(inc):
    image = cv2.imread(image_path)

    if (random.randint(1, 5) == 1):
        val_images.append(image)
        val_labels.append(([0, 1, 0]))
    else:
        images.append(image)
        labels.append([0, 1, 0])

        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 1, 0])

        #flip horizontal
        image_fh = cv2.flip(image, 1)
        images.append(image_fh)
        labels.append([0, 1, 0])

        hue_image = cv2.cvtColor(image_fh, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 1, 0])

        #flip vertical
        image_fv = cv2.flip(image, 0)
        images.append(image_fv)
        labels.append([0, 1, 0])

        hue_image = cv2.cvtColor(image_fv, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 1, 0])

        #flip both
        image_fvh = cv2.flip(image_fv, 1)
        images.append(image_fvh)
        labels.append([0, 1, 0])

        hue_image = cv2.cvtColor(image_fvh, cv2.COLOR_BGR2HSV)
        h = -1 * random.randint(10, 15) if random.randint(0, 2) == 0 else random.randint(10, 30)
        hue_image += np.array([h, 0, 0], np.uint8)
        hue_image = cv2.cvtColor(hue_image, cv2.COLOR_HSV2BGR)
        images.append(hue_image)
        labels.append([0, 1, 0])

print("normalizing")

for i in tqdm(range(len(images))):
    images[i] = images[i].astype(np.float32)
    images[i] /= 255.0

for i in tqdm(range(len(val_images))):
    val_images[i] = val_images[i].astype(np.float32)
    val_images[i] /= 255.0


print("converting to numpy")
images = np.array(images, np.float32)
val_images = np.array(val_images, np.float32)
labels = np.array(labels, np.float32)
val_labels = np.array(val_labels, np.float32)

print("saving as npy")

np.save("Z:/Tree_Direction_Images_v3.npy", images)
np.save("Z:/Tree_Direction_Val_Images_v3.npy", val_images)
np.save("Z:/Tree_Direction_Labels_v3.npy", labels)
np.save("Z:/Tree_Direction_Val_Labels_v3.npy", val_labels)