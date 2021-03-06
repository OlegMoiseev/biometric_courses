import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
from face_validation import get_dct, get_dft, get_gradient, get_scale, get_histogram, get_random_points
from collections import Counter


def get_trains_tests(num_of_samples=9):
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
        num_points = 155
        points = np.array([random.randint(0, 64, (1, 2)) for _ in range(num_points)])
        for image, class_face in zip(arg_x_tr, arg_y_tr):
            train_res.append((method(image, points), class_face, image))

    right, counter = 0, 0
    fig = plt.figure(num=meth_name)
    x_acc, y_acc = [], []
    for image, answer in zip(arg_x, arg_y):
        if meth_name != 'random_points':
            test_res = method(image)
            res, img_closest, method_res = predict_method(method, image, train_res)
        else:
            test_res = method(image, points)
            res, img_closest, method_res = predict_method(method, image, train_res, points)

        is_chart = meth_name != 'random_points' and len(test_res.shape) == 1  # if chart -> len 1, if image -> len 2
        if res == answer:
            right += 1

        counter += 1
        x_acc.append(counter)
        curr_accuracy = right / counter
        y_acc.append(curr_accuracy)

    accuracy = right / len(arg_x)
    print(accuracy)
    return accuracy


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

    return accuracy


if __name__ == '__main__':
    methods = [get_random_points, get_scale, get_gradient, get_histogram, get_dct, get_dft]

    x_plot = []
    y_plot = [[] for _ in range(len(methods) + 1)]

    for num in range(1, 10):
        face_train, face_test, class_train, class_test = get_trains_tests(num)
        for j in range(len(methods)):
            tmp = test_method(methods[j], face_train, face_test, class_train, class_test)
            y_plot[j].append(tmp)

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
        y_plot[-1].append(acc_vote)
        x_plot.append(num)
        plt.clf()

        for k in range(len(y_plot)):
            plt.plot(x_plot, y_plot[k])
        plt.legend(labels=('Random points', 'Scale', 'Gradient', 'Histogram', 'DCT', 'DFT', 'Vote'))
        plt.draw()
    plt.show()
# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.

