import cv2
from tqdm import tqdm
from glob import glob


if __name__ == '__main__':
    files = glob('images/*.bmp') + glob('direction images/*.bmp')

    for file in tqdm(files):
        img = cv2.imread(file)
        cv2.imwrite(file[:-4] + '.png', img)
