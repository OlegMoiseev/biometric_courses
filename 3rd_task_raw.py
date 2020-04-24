import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
from face_validation import get_dct, get_dft, get_gradient, get_scale, get_histogram, get_random_points
from collections import Counter


def vote(image, trains):
    results = []
    for method, train in zip(methods, trains):
        meth_name = method.__name__[4:]

        if meth_name != 'random_points':
            res, img_closest, method_res = predict_method(method, image, train)

        else:
            res, img_closest, method_res = predict_method(method, image, train, points)

        results.append(res)
    win = Counter(results).most_common()[0][0]
    return win


def test_vote(trained, arg_x, arg_y):
    x_acc, y_acc = [], []
    counter, right = 0, 0
    accuracy = 0
    for image, answer in zip(arg_x, arg_y):
        res = vote(image, trained)
        if res == answer:
            right += 1
        counter += 1
        accuracy = right / counter
        x_acc.append(counter)
        y_acc.append(accuracy)
        plt.clf()
        plt.plot(x_acc, y_acc)
        plt.pause(0.01)
    plt.show()
    return accuracy


def get_trains_tests(num_of_samples=5):
    data_images = fetch_olivetti_faces()
    data_faces = data_images.images
    data_target = data_images.target

    images_all = 400
    images_per_person = 10
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, images_all, images_per_person):
        indices = list(range(i, i + num_of_samples))

        indices = rnd.sample(indices, num_of_samples)  # shuffle samples
        x_train.extend(data_faces[index] for index in indices)
        y_train.extend(data_target[index] for index in indices)

        remaining_indices = set(range(i, i + images_per_person)) - set(indices)
        x_test.extend(data_faces[index] for index in remaining_indices)
        y_test.extend(data_target[index] for index in remaining_indices)

    return x_train, x_test, y_train, y_test


def predict_method(method, new_face, train, points=None):
    meth_name = method.__name__[4:]
    if meth_name != 'random_points':
        hist = method(new_face)
    else:
        hist = method(new_face, points)

    dists = [np.linalg.norm(hist - res, ord=2) for res, _, _ in train]
    win = dists.index(min(dists))
    return train[win][1], train[win][2], train[win][0]  # return class_face, image, result of method


def test_method(method, arg_x_tr, arg_x, arg_y_tr, arg_y):
    meth_name = method.__name__[4:]
    print(meth_name)

    train_res = []
    if meth_name != 'random_points':
        for image, class_face in zip(arg_x_tr, arg_y_tr):
            train_res.append((method(image), class_face, image))
    else:
        num_points = 145
        points = np.array([random.randint(0, 64, (1, 2)) for _ in range(num_points)])
        for image, class_face in zip(arg_x_tr, arg_y_tr):
            train_res.append((method(image, points), class_face, image))

    right, counter = 0, 0
    fig = plt.figure(num=meth_name)
    x_acc, y_acc = [], []
    for image, answer in zip(arg_x, arg_y):
        if meth_name != 'random_points':
            res, img_closest, method_res = predict_method(method, image, train_res)
        else:
            res, img_closest, method_res = predict_method(method, image, train_res, points)

        if res == answer:
            right += 1

        counter += 1
        x_acc.append(counter)
        curr_accuracy = right / counter
        y_acc.append(curr_accuracy)
        plt.plot(x_acc, y_acc)

        plt.draw()
        plt.pause(0.01)
        plt.clf()
    plt.plot(x_acc, y_acc)

    accuracy = right / len(arg_x)
    print(accuracy)
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    methods = [get_random_points, get_scale, get_gradient, get_histogram, get_dct, get_dft]

    face_train, face_test, class_train, class_test = get_trains_tests()

    trains = []
    for method in methods:
        meth_name = method.__name__[4:]

        train_res = []
        if meth_name != 'random_points':
            for image, class_face in zip(face_train, class_train):
                train_res.append((method(image), class_face, image))
        else:
            num_points = 155
            points = np.array([random.randint(0, 64, (1, 2)) for _ in range(num_points)])
            for image, class_face in zip(face_train, class_train):
                train_res.append((method(image, points), class_face, image))

        trains.append(train_res)
    acc_vote = test_vote(trains, face_test, class_test)

# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.
