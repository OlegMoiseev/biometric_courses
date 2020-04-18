import cv2
from matplotlib import pyplot as plt

# for i in range(1, 21):
#     dists = []
#     for j in range(1, 11):
# name = 'photos/faces/s' + str(i) + '/' + str(j) + '.pgm'

name = 'photos/me.jpg'

img = cv2.imread(name, 0)

for i in range(100, 310, 10):
    template = cv2.imread('photos/crop2.png', 0)  # read template of me

    scale_percent = i  # percent of original size
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)

    w, h = template.shape[::-1]

    meth = cv2.TM_SQDIFF_NORMED

    # Apply template Matching
    res = cv2.matchTemplate(img, template, meth)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 5)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.show()
