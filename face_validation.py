import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct


def get_histogram(image):
    h, b, _ = plt.hist(image.ravel(), 256, [0, 256])
    return h


def get_dft(image):
    f = np.fft.fft2(image)
    return f


def get_dct(image):
    c = dct(image)
    return c


def get_gradient(image):
    gr = np.gradient(image)
    return gr


if __name__ == "__main__":
    name = 'photos/valid_40_5.pgm'
    img = cv2.imread(name, 0)
    # example_hist, _ = get_histogram(img)

    example_dft = get_dft(img)[0]


    names = ['1_1', '2_2', '5_3', '21_8', '26_4', '31_8', '32_5', '32_10', '40_2', '40_10']
    dists = []
    for name in names:
        name = 'photos/photos_for_valid/' + name + '.pgm'
        img = cv2.imread(name, 0)
        cv2.imshow('name', img)
        dft = get_dft(img)[0]
        dist = np.linalg.norm(example_dft - dft, ord=2)
        dists.append(dist)
    print(names[np.argmin(dists)], ':', min(dists))
