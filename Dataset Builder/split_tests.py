import time
import cv2
import numpy as np
from patchify import patchify, unpatchify


def blockgen(array, bpa):
    """Creates a generator that yields multidimensional blocks from the given
array(_like); bpa is an array_like consisting of the number of blocks per axis
(minimum of 1, must be a divisor of the corresponding axis size of array). As
the blocks are selected using normal numpy slicing, they will be views rather
than copies; this is good for very large multidimensional arrays that are being
blocked, and for very large blocks, but it also means that the result must be
copied if it is to be modified (unless modifying the original data as well is
intended)."""
    bpa = np.asarray(bpa) # in case bpa wasn't already an ndarray

    # parameter checking
    # if array.ndim != bpa.size:         # bpa doesn't match array dimensionality
    #     raise ValueError("Size of bpa must be equal to the array dimensionality.")
    # if (bpa.dtype != np.int            # bpa must be all integers
    #     or (bpa < 1).any()             # all values in bpa must be >= 1
    #     or (array.shape % bpa).any()): # % != 0 means not evenly divisible
    #     raise ValueError("bpa ({0}) must consist of nonzero positive integers "
    #                      "that evenly divide the corresponding array axis "
    #                      "size".format(bpa))


    # generate block edge indices
    rgen = (np.r_[:array.shape[i]+1:array.shape[i]//blk_n]
            for i, blk_n in enumerate(bpa))

    # build slice sequences for each axis (unfortunately broadcasting
    # can't be used to make the items easy to operate over
    c = [[np.s_[i:j] for i, j in zip(r[:-1], r[1:])] for r in rgen]

    # Now to get the blocks; this is slightly less efficient than it could be
    # because numpy doesn't like jagged arrays and I didn't feel like writing
    # a ufunc for it.
    for idxs in np.ndindex(*bpa):
        blockbounds = tuple(c[j][idxs[j]] for j in range(bpa.size))

        yield array[blockbounds]


def reshape_split(image: np.ndarray, kernel_size: tuple):

    img_height, img_width, channels = image.shape
    tile_height, tile_width = kernel_size

    tiled_array = image.reshape((img_height // tile_height,
                                tile_height,
                                img_width // tile_width,
                                tile_width,
                                channels))
    tiled_array = tiled_array.swapaxes(1, 2)
    return tiled_array


image = cv2.imread("Lac Flocon.tif")

image = cv2.resize(image, (19968, 19968), cv2.INTER_AREA)

# t = time.time()
# for i in range(10):
#     sections = reshape_split(image, (256, 256))
#     sections = sections.reshape((-1, 256, 256, 3)).astype(np.float32)
#     sections /= 255.0
# print((time.time() - t)/10)
# t = time.time()
# for i in range(10):
#     patches = patchify(image, (256, 256, 3), step=256)
#     patches = np.reshape(patches, (78*78, 256, 256, 3))
# print((time.time() - t)/10)
#
# print(patches.shape)

# l = []
#
# print("spliting")
# t = time.time()
# for i in range(10):
#     sections = blockgen(image, [78, 78, 1])
#     for section in sections:
#         l.append(section)
#
#     l_np = np.asarray(l, np.float32)
#     l_np /= 255.0
#
# print((time.time() - t)/10)
#
# cv2.imshow('test', l[0])
# cv2.imshow('test1', l[1])
# cv2.imshow('test78', l[78])
# cv2.waitKey(0)