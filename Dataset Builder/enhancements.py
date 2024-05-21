import random
from glob import glob
import cv2
from tqdm import tqdm



images_folder = "validation_dataset_images_v3/"
masks_folder = "validation_dataset_masks_v3/"

image_paths = glob(images_folder + "*.png")
mask_paths = glob(masks_folder + "*.png")
count = 34912

a = list(range(34912))

random.shuffle(a)
#
# for i in tqdm(range(2000)):
#     image = cv2.imread(image_paths[a[i]])
#     mask = cv2.imread(mask_paths[a[i]], cv2.IMREAD_GRAYSCALE)
#
#     cv2.imwrite("manual_dataset_images_v2/img" + str(count) + ".png", image)
#     cv2.imwrite("manual_dataset_masks_v2/img" + str(count) + ".png", mask)
#
#     count+=1

for i in tqdm(range(500)):
    image = cv2.imread(image_paths[a[i]])
    mask = cv2.imread(mask_paths[a[i]], cv2.IMREAD_GRAYSCALE)

    x1 = random.randint(20, 235)
    y1 = random.randint(20, 235)
    x2 = random.randint(20, 235)
    y2 = random.randint(20, 235)

    if x2 == x1:
        x2 += 1

    m = (y2 - y1) / float(x2 - x1)
    b = y1 - m*x1

    if random.randint(0, 1) == 0:
        for y in range(256):
            for x in range(256):
                if y > m*x+b:
                    image[y][x] = [0, 0, 0]
                    mask[y][x] = 0

    else:
        for y in range(256):
            for x in range(256):
                if y < m*x+b:
                    image[y][x] = [0, 0, 0]
                    mask[y][x] = 0

    cv2.imwrite("enh_img_v3/img" + str(count) + ".png", image)
    cv2.imwrite("enh_mask_v3/img" + str(count) + ".png", mask)

    count += 1