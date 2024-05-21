import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread(r"D:\arcGIS\TreeDirections\direction_images\22.bmp")

    img = cv2.resize(img, None, fx=1.0/6.0, fy=1.0/6.0)

    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    img = cv2.resize(img, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    img = cv2.filter2D(img, -1, kernel)

    cv2.imshow("test", img)
    cv2.imwrite("30cm_upscaled.png", img)
    cv2.waitKey(0)
