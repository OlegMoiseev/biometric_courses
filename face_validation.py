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


name = 'photos/faces/s34/8.pgm'
img = cv2.imread(name, 0)
# example_hist, _ = get_histogram(img)

example_dft = get_dft(img)[0]


means = []
for i in range(1, 41):
    dists = []
    for j in range(1, 11):
        name = 'photos/faces/s' + str(i) + '/' + str(j) + '.pgm'
        img = cv2.imread(name, 0)
        dft = get_dft(img)[0]
        dist = np.linalg.norm(example_dft - dft, ord=2)
        dists.append(dist)
    means.append(np.mean(dists))
print(np.argmin(means) + 1, ':', min(means))
