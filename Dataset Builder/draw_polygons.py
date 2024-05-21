import numpy as np
import cv2
import csv


def draw_polygons(csv_path, image):
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

    for shape in shapes:
        cv2.drawContours(image, [np.array(shape, int)], -1, [0, 0, 255], 2)

    for line in poly_lines:
        cv2.polylines(image, [np.array(line, int)], False, [0, 0, 255], 2)

    return image


if __name__ == '__main__':
    image = cv2.imread("tree_polygons/images/2.bmp")

    image = draw_polygons("tree_polygons/polygon_csv/2.csv", image)

    cv2.imshow("polygons", image)
    cv2.imwrite("polygons2.png", image)
    cv2.waitKey(0)



