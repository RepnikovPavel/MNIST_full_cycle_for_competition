"""
    1)разархивируются idx файлы и
    2)ко всем функция добавляется название __make_dataset__ в качетсве сишной реализации namespace
    3)создаются записи примеров (картинка, метка)  для датасета MNIST в виде np.array
    4)Список использованных функций:
        1)__make_dataset__get_images_from_idx()
        2)__make_dataset__get_labels_from_idx()
        3)__make_dataset__build_mnist_dataset()
"""

import struct as st
import numpy as np


def __make_dataset__get_images_from_idx(path):
    """
        возращает np array shape=(num_of_image,height,width)
        1)работает конкретно под конкретный формат файла .idx3-ubyte
        2)читается все разом
    """
    # "https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1"
    file = open(path, 'rb')
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    num_of_images = st.unpack('>I', file.read(4))[0]
    num_of_rows = st.unpack('>I', file.read(4))[0]
    num_of_column = st.unpack('>I', file.read(4))[0]
    images_array = np.zeros((num_of_images, num_of_rows, num_of_column))

    n_bytes_total = num_of_images * num_of_rows * num_of_column * 1
    images_array = np.asarray(st.unpack('>' + 'B' * n_bytes_total, file.read(n_bytes_total)), dtype=np.uint8).reshape(
        (num_of_images, num_of_rows, num_of_column))

    return images_array


def __make_dataset__get_labels_from_idx(path):
    """
        возращает np array shape=(num_of_label)
        1)работает конкретно под конкретный формат файла .idx1-ubyte
        2)читается все разом
    """
    # "https://medium.com/the-owl/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1"
    file = open(path, 'rb')
    file.seek(0)
    magic = st.unpack('>4B', file.read(4))
    num_of_labels = st.unpack('>I', file.read(4))[0]
    images_array = np.zeros((num_of_labels))

    n_bytes_total = num_of_labels * 1
    labels_array = np.asarray(st.unpack('>' + 'B' * n_bytes_total, file.read(n_bytes_total)), dtype=np.uint8).reshape(
        (num_of_labels))

    return labels_array


def __make_dataset__build_mnist_dataset(train_data_path,
                        train_labels_path,
                        test_data_path,
                        test_labels_path,
                        record_train_data_path,
                        record_train_labels_path,
                        record_test_data_path,
                        record_test_labels_path
                        ):
    """
        результат работы: в указанные пути будут записаны датасет
        train и датасет test
        состоящие из np.array = из сэмплов (картинка, метка)  shape = (( image  ),( label ))

        1)разделение на train и test происходит на уровне датасета непосредственно из сырых данных,
        не относящегося к обучению(никаих разделений для кросс валидации, никаих
        аугментаций)

        2)в случае конкретно MNIST под train_path,test_path имеется в виду
         непосредственно имя файла(потому что MNIST лежит целиком в 4х бинарниках)
         на практике обычно дается именно директория и имена файлов нужно будет
         доставать автоматически, поэтому я выбрал именно такие названия с заделом на будущее
    """

    train_images = __make_dataset__get_images_from_idx(train_data_path)
    train_labels = __make_dataset__get_labels_from_idx(train_labels_path)
    test_images = __make_dataset__get_images_from_idx(test_data_path)
    test_labels = __make_dataset__get_labels_from_idx(test_labels_path)

    train_data_filename = record_train_data_path + '\\train_data_record'
    train_labels_filename = record_train_labels_path + '\\train_labels_record'
    test_data_filename = record_test_data_path + '\\test_data_record'
    test_labels_filename = record_test_labels_path + '\\test_labels_record'

    np.save(train_data_filename, train_images)
    np.save(train_labels_filename, train_labels)
    np.save(test_data_filename, test_images)
    np.save(test_labels_filename, test_labels)


if __name__ == '__main__':
    __make_dataset__build_mnist_dataset(
        train_data_path='F:\\unpacked_datasets\\mnist\\train-images.idx3-ubyte',
        train_labels_path='F:\\unpacked_datasets\\mnist\\train-labels.idx1-ubyte',
        test_data_path='F:\\unpacked_datasets\\mnist\\t10k-images.idx3-ubyte',
        test_labels_path='F:\\unpacked_datasets\\mnist\\t10k-labels.idx1-ubyte',

        record_train_data_path = 'F:\\builded_datasets\\mnist\\train',
        record_train_labels_path='F:\\builded_datasets\\mnist\\train',
        record_test_data_path='F:\\builded_datasets\\mnist\\test',
        record_test_labels_path='F:\\builded_datasets\\mnist\\test'
    )
