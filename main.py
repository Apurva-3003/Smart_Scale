'''
Use a seam carving algorithm to reduce image width while
preserving important details by removing low-contour regions.
'''

import sys
import numba
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
from tqdm import trange

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])

    # np.transpose() switches indices of rows and columns
    filter_dv = np.transpose(filter_du)

    # Convert from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis = 2)
    filter_dv = np.stack([filter_dv] * 3, axis = 2)

    # Find each pixel's energy value
    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # Sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis = 2)

    return energy_map

# numba.jit helps speed up calculations
@numba.jit
def minimum_seam(img):
    
    row, col, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, row):
        for j in range(0, col):
            # Handle the left edge of the image, to ensure we don't index -1
            if j == 0:
                idx = np.argmin(M[i-1, 0:2])
                backtrack[i, j] = idx
                min_energy = M[i-1, idx]
            else:
                idx = np.argmin(M[i-1, j-1: j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

@numba.jit
def carve_col(img):
    row, col, _ = img.shape

    M, backtrack = minimum_seam(img)

    # Create a (row, col) matrix filled with the value True,
    # We'll be removing all pixels which have False later
    mask = np.ones((row,col), dtype = np.bool)

    # Find the position of the smallest element in the
    # last row of M
    j = np.argmin(M[-1])

    for i in reversed(range(row)):
        # Mark the pixels for deletion
        mask[i, j] = False
        j = backtrack[i,j]

    # Since the image has 3 channels, we convert our
    # mask to 3D
    mask = np.stack([mask] * 3, axis=2)

    # Delete all the pixels marked False in the mask,
    # and resize it to the new image dimensions
    img = img[mask].reshape((row, col - 1, 3))

    return img

def crop_col(img, scale_col):
    row, col, _ = img.shape
    new_col = int(scale_col * col)

    for i in trange(col-new_col):
        img = carve_col(img)   

    return img

def main():
    scale = float(sys.argv[1])
    in_filename = sys.argv[2]
    out_filename = sys.argv[3]

    img = imread(in_filename)
    out = crop_col(img, scale)
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()