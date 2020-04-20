import numpy as np
from numpy import random
import random
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces


# from face_validation import get_histogram

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


def predict_hist(new_face, train):
    hist = get_histogram(new_face)
    dists = [np.linalg.norm(hist - res, ord=2) for res, _, _ in train]
    win = dists.index(min(dists))
    return train[win][1]  # return class_face


if __name__ == '__main__':
    face_train, face_test, class_train, class_test = get_trains_tests()
    train_res = []
    for image, class_face in zip(face_train, class_train):
        train_res.append((get_histogram(image), class_face, image))

    for image in face_test:
        print(predict_hist(image, train_res))
# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.
