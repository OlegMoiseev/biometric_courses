import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
from face_validation import get_dct, get_dft, get_gradient, get_scale, get_histogram, get_random_points
from collections import Counter


def get_trains_tests():
    data_images = fetch_olivetti_faces()
    data_faces = data_images.images
    data_target = data_images.target

    images_all = 400
    images_per_person = 10
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(0, images_all, images_per_person):
        num_of_samples = 8
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


def vote(image, trains):
    results = []

    plots = [ax1, ax2, ax3, ax4, ax5, ax6]
    for method, train, plot in zip(methods, trains, plots):
        meth_name = method.__name__[4:]

        if meth_name != 'random_points':
            res, img_closest, method_res = predict_method(method, image, train)

            is_chart = len(method_res.shape) == 1  # if chart -> len 1, if image -> len 2
            plot.cla()
            plot.plot(method_res) if is_chart else plot.imshow(method_res, cmap='gray')

        else:
            res, img_closest, method_res = predict_method(method, image, train, points)
            plot.cla()
            plot.imshow(img_closest, cmap='gray')
            plot.scatter(x=[point[0][0] for point in points], y=[point[0][1] for point in points], c='r')

        results.append(res)
    win = Counter(results).most_common()[0][0]
    return win


def test_vote(trained, arg_x, arg_y):
    x_acc, y_acc = [], []
    counter, right = 0, 0

    for image, answer in zip(arg_x, arg_y):
        res = vote(image, trained)

        ax0.cla()
        ax0.imshow(image, cmap='gray')

        if res == answer:
            right += 1
            fig.patch.set_facecolor('xkcd:light green')
        else:
            fig.patch.set_facecolor('xkcd:salmon')
        counter += 1
        accuracy = right / counter
        print(accuracy)
        x_acc.append(counter)
        y_acc.append(accuracy)

        ax7.cla()
        ax7.set_title(f'Current VOTE accuracy: {accuracy:.2f}')

        ax7.plot(x_acc, y_acc)
        plt.draw()
        plt.pause(0.1)


if __name__ == '__main__':
    methods = [get_random_points, get_scale, get_gradient, get_histogram, get_dct, get_dft]
    # methods = [get_scale, get_gradient, get_histogram, get_dct, get_dft]

    face_train, face_test, class_train, class_test = get_trains_tests()

    trains = []
    num_points = 145
    points = np.array([random.randint(0, 64, (1, 2)) for _ in range(num_points)])

    for method in methods:
        meth_name = method.__name__[4:]
        print(meth_name)
        train_res = []
        if meth_name != 'random_points':
            for image, class_face in zip(face_train, class_train):
                train_res.append((method(image), class_face, image))
        else:

            for image, class_face in zip(face_train, class_train):
                train_res.append((method(image, points), class_face, image))
        trains.append(train_res)

    fig = plt.figure(num='VOTE')

    s_plots_code = [434, 435, 436, 437, 438, 439]   # num strings, cols
    ax0 = plt.subplot(432)

    ax1 = plt.subplot(434)
    ax2 = plt.subplot(435)
    ax3 = plt.subplot(436)
    ax4 = plt.subplot(437)
    ax5 = plt.subplot(438)
    ax6 = plt.subplot(439)

    ax7 = plt.subplot(414)

    plt.subplots_adjust(hspace=.5)
    test_vote(trains, face_test, class_test)
# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.

