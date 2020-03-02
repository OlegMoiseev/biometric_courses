import cv2
import numpy as np
from matplotlib import pyplot as plt

for i in range(1, 11):
    name = 'photos/' + str(i) + '.jpg'

    img = cv2.imread(name, 0)
    img2 = img.copy()
    template = cv2.imread('photos/me_template.jpg', 0)
    w, h = template.shape[::-1]

    meth = cv2.TM_SQDIFF_NORMED
    img = img2.copy()

    # Apply template Matching
    res = cv2.matchTemplate(img,template, meth)
    print(res)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 20)

    plt.subplot(121), plt.imshow(res, cmap='gray')
    plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)

    plt.show()
