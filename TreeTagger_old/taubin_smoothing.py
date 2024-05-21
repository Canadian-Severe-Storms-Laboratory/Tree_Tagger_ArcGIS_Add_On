import numpy as np
import matplotlib.pyplot as plt
import cv2
from copy import deepcopy


def polygon_smooth(pts, k=0.5, u=-0.5, steps=5):
    N = len(pts)

    for s in range(steps):
        L = []

        for i in range(N):
            L.append([(pts[(i-1) % N][0] + pts[(i+1) % N][0])/2.0 - pts[i][0],
                      (pts[(i-1) % N][1] + pts[(i+1) % N][1])/2.0 - pts[i][1]])

        if s % 2 == 0:
            for i in range(N):
                pts[i][0] = pts[i][0] + k * L[i][0]
                pts[i][1] = pts[i][1] + k * L[i][1]

        else:
            for i in range(N):
                pts[i][0] = pts[i][0] + u * L[i][0]
                pts[i][1] = pts[i][1] + u * L[i][1]

    return pts


def remove_pixels(image):
    # Define a kernel to check 4 neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Iterate over each pixel and count its neighbors
    result_image = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j] > 0:
                neighbors_sum = np.sum((image[i-1:i+2, j-1:j+2]/255) * kernel)
                if neighbors_sum > 4:  # If the pixel has 3 or more neighbors, keep it
                    result_image[i-1, j-1] = 255

    return result_image


if __name__ == '__main__':

    img = cv2.imread("gridImage.png", cv2.IMREAD_GRAYSCALE)

    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    #img = cv2.erode(img, np.ones((3, 3), np.uint8))

    img = remove_pixels(img)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    plt.figure()

    for x in contours:
        contour = [[c[0][0], -c[0][1]] for c in x]

        if len(contour) < 3: continue

        contour = polygon_smooth(contour)

        contour.append(contour[0])

        xs, ys = zip(*contour)

        plt.plot(xs, ys)

    plt.show()

