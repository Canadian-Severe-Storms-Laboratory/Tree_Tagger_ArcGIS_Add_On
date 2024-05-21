import cv2
import numpy as np


def houghp_transform(image):
    # Apply Probabilistic Hough Transform
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=5)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the lines on the original image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image


def hough_transform(image):
    # Apply Hough Transform
    lines = cv2.HoughLines(image, 1, np.pi/180, 600)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw the lines on the original image
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1, cv2.LINE_AA)

    return image


if __name__ == '__main__':

    # Read the image in grayscale
    image = cv2.imread('simplified.png', cv2.IMREAD_GRAYSCALE)

    # Apply Hough Transform
    result = hough_transform(image)

    # Display the result
    cv2.imshow('Hough Transform Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()