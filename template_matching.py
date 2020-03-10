import cv2
from matplotlib import pyplot as plt

for i in range(1, 9):
    name = 'photos/' + str(i) + '.jpg'

    img = cv2.imread(name, 0)
    template = cv2.imread('photos/crop_template.jpg', 0)  # read template of me
    w, h = template.shape[::-1]

    meth = cv2.TM_SQDIFF_NORMED

    # Apply template Matching
    res = cv2.matchTemplate(img, template, meth)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = min_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(img, top_left, bottom_right, 255, 20)
    plt.imshow(img, cmap='gray')
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])

    plt.show()
