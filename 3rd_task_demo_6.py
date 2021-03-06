import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import random as rnd
from sklearn.datasets import fetch_olivetti_faces
from face_validation import get_dct, get_dft, get_gradient, get_scale, get_histogram, get_random_points


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
            test_res = method(image)
            res, img_closest, method_res = predict_method(method, image, train_res)
        else:
            test_res = method(image, points)
            res, img_closest, method_res = predict_method(method, image, train_res, points)

        is_chart = meth_name != 'random_points' and len(test_res.shape) == 1  # if chart -> len 1, if image -> len 2
        if res == answer:
            right += 1
            fig.patch.set_facecolor('xkcd:light green')
        else:
            fig.patch.set_facecolor('xkcd:salmon')

        plt.title("Closest images")

        ax2 = plt.subplot(421)
        ax2.imshow(img_closest, cmap='gray')
        ax2.set_title(f'test ({answer})')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax4 = plt.subplot(423)

        if meth_name == 'random_points':
            ax4.imshow(img_closest, cmap='gray')
            plt.scatter(x=[point[0][0] for point in points], y=[point[0][1] for point in points], c='r')
        else:
            ax4.plot(test_res) if is_chart else ax4.imshow(test_res, cmap='gray')

        ax3 = plt.subplot(422)
        ax3.imshow(image, cmap='gray')
        ax3.set_title(f'closest ({res})')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax5 = plt.subplot(424)
        if meth_name == 'random_points':
            ax5.imshow(image, cmap='gray')
            plt.scatter(x=[point[0][0] for point in points], y=[point[0][1] for point in points], c='r')
        else:
            ax5.plot(method_res) if is_chart else ax5.imshow(method_res, cmap='gray')

        counter += 1
        x_acc.append(counter)
        curr_accuracy = right / counter
        y_acc.append(curr_accuracy)
        ax1 = plt.subplot(414)
        ax1.plot(x_acc, y_acc)
        ax1.set_title(f'Current {meth_name} accuracy: {curr_accuracy:.2f}')

        plt.draw()
        plt.pause(0.01)
        plt.clf()

    accuracy = right / len(arg_x)
    print(accuracy)
    plt.close(fig)


if __name__ == '__main__':
    methods = [get_scale, get_histogram, get_random_points, get_gradient, get_dct, get_dft]

    face_train, face_test, class_train, class_test = get_trains_tests()
    for method in methods:
        test_method(method, face_train, face_test, class_train, class_test)

# тренируемся - то есть записываем метрики для каждого изображения. Потом делаем тест, кидая на вход лицо из уже
# тренированных - так получаем точность в 100%. Потом кидаем что-то новое - тогда должны получить около 100%.

