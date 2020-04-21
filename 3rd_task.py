import numpy as np
from numpy import random
import random
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces


def get_histogram(image: np.ndarray):
    hist, _ = np.histogram(image, bins=np.linspace(0, 1))
    return hist


def get_trains_tests():
    data_images = fetch_olivetti_faces()
    data_faces = data_images.images
    data_target = data_images.target

    images_all = 400
    images_per_person = 10
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, images_all, images_per_person):
        num_of_samples = 9
        indices = list(range(i, i + num_of_samples))

        indices = random.sample(indices, num_of_samples)  # shuffle samples
        x_train.extend(data_faces[index] for index in indices)
        y_train.extend(data_target[index] for index in indices)

        remaining_indices = set(range(i, i + images_per_person)) - set(indices)
        x_test.extend(data_faces[index] for index in remaining_indices)
        y_test.extend(data_target[index] for index in remaining_indices)

    return x_train, x_test, y_train, y_test


def predict_method(method, new_face, train):
    hist = method(new_face)
    dists = [np.linalg.norm(hist - res, ord=2) for res, _, _ in train]
    win = dists.index(min(dists))
    return train[win][1], train[win][2]  # return class_face, image


def test_method(method, arg_x_tr, arg_x, arg_y_tr, arg_y):
    train_res = []
    for image, class_face in zip(arg_x_tr, arg_y_tr):
        train_res.append((method(image), class_face, image))

    right = 0
    fig = plt.figure()

    for image, answer in zip(arg_x, arg_y):

        res, img_closest = predict_method(method, image, train_res)

        if res == answer:
            right += 1
            fig.patch.set_facecolor('xkcd:light green')
        else:
            # print(res, answer)
            fig.patch.set_facecolor('xkcd:salmon')

        plt.title("Closest images")

        ax2 = plt.subplot(421)
        ax2.imshow(img_closest, cmap='gray')  # Values >0.0 zoom out
        ax2.set_title(f'test ({answer})')
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = plt.subplot(422)
        ax3.imshow(image, cmap='gray')
        ax3.set_title(f'closest ({res})')
        ax3.set_xticks([])
        ax3.set_yticks([])

        plt.draw()
        plt.pause(1)
        plt.clf()


    accuracy = right / len(face_test)
    print(accuracy)


if __name__ == '__main__':
    face_train, face_test, class_train, class_test = get_trains_tests()
    test_method(get_histogram, face_train, face_test, class_train, class_test)

# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.
