import cv2
import glob
import random
from tqdm.auto import tqdm
import numpy as np

# right = glob.glob("Z:/labelled_tree_directions/aug/right/*.bmp")
# left = glob.glob("Z:/labelled_tree_directions/aug/left/*.bmp")
# inc = glob.glob("Z:/labelled_tree_directions/aug/inc/*.bmp")
#
# X = []
# Y = []
#
# for file in tqdm(right):
#     Y.append([0, 0, 1])
#
#     image = cv2.imread(file)
#     image = image.astype(np.float32)
#     image /= 255.0
#     X.append(image)
#
# for file in tqdm(left):
#     Y.append([1, 0, 0])
#
#     image = cv2.imread(file)
#     image = image.astype(np.float32)
#     image /= 255.0
#     X.append(image)
#
# for file in tqdm(inc):
#     Y.append([0, 1, 0])
#
#     image = cv2.imread(file)
#     image = image.astype(np.float32)
#     image /= 255.0
#     X.append(image)
#
# print("creating numpy arrays")
# X = np.array(X)
# Y = np.array(Y)
#
# print("Saving")
# np.save("Z:/labelled_tree_directions/aug/Tree_Direction_Images_v2", X)
# np.save("Z:/labelled_tree_directions/aug/Tree_Direction_Labels_v2", Y)

labels = np.load("Z:/Tree_Direction_labels_v2.npy")
X = []

for label in tqdm(labels):
    label = label.astype(np.float32)

    X.append(label)

print("converting to numpy")
X = np.array(X)

print("saving as np")
np.save("Z:/Tree_Direction_Labels_v2_f.npy", X)

