"""
    1)ко всем функциям добавляется название __training_loop__ в качетсве си-реализации namespace
    2)с помощью флагов можно тренировать отдельно взятый алгоритм или несколько алгоритомв.
    то же самое относится и к воспроизведению отчетов о тренировке алгоритма/ алгоритмов( в виде графиков )

"""

from dense_FCN import build_and_train_dense_FCN, load_dense_FCN_and_train_more
from linear_classifier import build_and_train_linear_classifier, load_linear_classifier_and_train_more
from VGG import build_and_train_VGG, load_VGG_and_train_more

import os
import glob
import math

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def __training_loop__load_all_public_images_and_labels(
        train_images_filename,
        train_labels_filename
):
    """
        на вход: имена файлов с расширением .npy в которых лежат картинки и метки
        на выход: список из np.array (картинки, метки)
    """
    all_train_images = np.load(train_images_filename)
    all_train_labels = np.load(train_labels_filename)
    return all_train_images, all_train_labels


def __training_loop__save_trained_model_and_training_history(training_loop_path, model, history, description_for_txt,
                                                             algorithm_id):
    """
        записывает Model и history на диск
        training_loop_path: путь к папке, где будут сохраняться обученные модели
        model: обученная модель
        history: информация о тренировке || инфорация об алгоритме  в виде pandas_dataframe
        algorithm_id: имя алгоритма
    """
    if training_loop_path[-1] == '\\':
        filename = training_loop_path + str(algorithm_id)
    else:
        filename = training_loop_path + '\\' + str(algorithm_id)
    tf.keras.models.save_model(model, filename)
    history.to_csv(filename + '_history.csv')
    text_file_title_for_plot = open(filename + '_title.txt', "w")
    text_file_title_for_plot.write(str(description_for_txt))
    text_file_title_for_plot.close()


def __training_loop__get_the_aspect_ratio(real_num_of_images):
    """
        на вход: число картинок для отрисовки
        на выход: соотношения сторон и max_index для проверки на out of range для
        массива картинок

        суть:
            нужно получить соотношение 2:1 для отрисовки графиков на одном figure
    """
    a = real_num_of_images

    b = 0
    m = 0
    n = 0
    if a == 1:
        m = 1
        n = 1
        max_index_in_plot_cycle = 1
        return m, n, max_index_in_plot_cycle
    else:
        i = 0
        j = 0
        while (True):
            if (j > 100):
                print('infinitive cycle')
                return
            else:
                b = a + i
                m = int(math.sqrt(b / 2))
                n = int(b / m)
                if m * n == b:
                    break
                else:
                    i += 1
                    j += 1
                    continue
        max_index_in_plot_cycle = b
        return m, n, max_index_in_plot_cycle


def __training_loop__load_and_show_training_summary(training_loop_path, load_and_show_dict):
    """
        по заданным шаблонам в папке training_loop_path находятся информация для отрисовки в указанной папке,
        после чего из единого формата данных для все алогоритмов, указанных
        в  load_and_show_dict рисуются картинки

        load_and_show_dict={
            'alg_id': True
            'alg_id': False
            ...
        }

        покачто формат частного характера, а именно в csv файле лежит model.fit().history()
        а в txt лежит title для картинки

        картинка рисуется только если такой формат существует в данной директории
    """

    ###################
    ######
    #
    #   !!!ВАЖНО!!!    нужно проверить в каком порядке у нас создается список (может быть такое, что данные могут записываться
    #   не последовательно ТОГДА ВСЕ СЛОМАЕТСЯ лучше сделать словарь для надежности
    #
    ######
    ##################

    if os.listdir(training_loop_path):

        # загрузка путей для всех существующих алгоритмов
        if training_loop_path[-1] == '\\':
            parsed_txt_title_list_of_paths = glob.glob(training_loop_path + '*.txt')
            parsed_train_history_list_of_paths = glob.glob(training_loop_path + '*.csv')
        else:
            parsed_txt_title_list_of_paths = glob.glob(training_loop_path + '\\*.txt')
            parsed_train_history_list_of_paths = glob.glob(training_loop_path + '\\*.csv')

        if len(parsed_txt_title_list_of_paths) != len(parsed_train_history_list_of_paths):
            print('число титульников для графиков и табличек для графиков разное')
            return

        # нужно отобрать только те алгоритмы, которые просят отрисовать
        txt_title_list_of_paths_to_process = []
        train_history_list_of_paths_to_process = []

        for key, value in load_and_show_dict.items():
            for path in parsed_txt_title_list_of_paths:
                if key in path:
                    if value == True:
                        txt_title_list_of_paths_to_process.append(path)
            for path in parsed_train_history_list_of_paths:
                if key in path:
                    if value == True:
                        train_history_list_of_paths_to_process.append(path)

        # отрисовка отобранных алгоритмов
        number_of_graphs = len(txt_title_list_of_paths_to_process)

        m, n, max_index = __training_loop__get_the_aspect_ratio(number_of_graphs)

        fig, axarr = plt.subplots(m, n, figsize=(19, 9), squeeze=False)

        k = 0
        for i in range(m):
            for j in range(n):
                if i * j <= max_index:
                    dt_table = pd.read_csv(train_history_list_of_paths_to_process[k])
                    file = open(txt_title_list_of_paths_to_process[k], "r")
                    title = str(file.read())
                    file.close()
                    k += 1

                    #   сделаем проверку и отрисуем только в случае нахождения нужных колонок
                    if 'x' in dt_table and 'y1' in dt_table and 'y2' in dt_table:
                        axarr[i, j].plot(dt_table['x'], dt_table['y1'], label='train')
                        axarr[i, j].plot(dt_table['x'], dt_table['y2'], label='val')
                        axarr[i, j].legend()
                        axarr[i, j].set_title(title)
        plt.show()
    else:
        print("забыли потренировать модели")
        return


def execution_of_existing_functions(config):
    pub_images, pub_labels = __training_loop__load_all_public_images_and_labels(config['recorded_train_data_filename'],
                                                                                config[
                                                                                    'recorded_train_labels_filename'])
    algorithms_config = config['algorithms']

    # training pass
    # 1st model
    if algorithms_config['handmade_VGG']['processing']:
        alg_id = 'handmade_VGG'
        alg_config = algorithms_config['handmade_VGG']
        if alg_config['training']:
            training_config = alg_config['training_config']
            if training_config['training_from_zero']:
                model, df_history = build_and_train_VGG(pub_images, pub_labels)
                __training_loop__save_trained_model_and_training_history(
                    config['train_loop_path'],
                    model, df_history,
                    description_for_txt=alg_id + 'categorial_accuracy',
                    algorithm_id=alg_id
                )
            if training_config['train_an_under-trained_model']:
                model, df_history = load_VGG_and_train_more(config['train_loop_path'], alg_id, pub_images, pub_labels)
                __training_loop__save_trained_model_and_training_history(
                    config['train_loop_path'],
                    model, df_history,
                    description_for_txt=alg_id + ' categorial_accuracy',
                    algorithm_id=alg_id
                )

    if algorithms_config['dense_FCN']['processing']:
        alg_id = 'dense_FCN'
        alg_config = algorithms_config[alg_id]
        if alg_config['training']:
            training_config = alg_config['training_config']
            if training_config['training_from_zero']:
                model, df_history = build_and_train_dense_FCN(pub_images, pub_labels)
                __training_loop__save_trained_model_and_training_history(
                    config['train_loop_path'],
                    model, df_history,
                    description_for_txt=alg_id + ' categorial_accuracy',
                    algorithm_id=alg_id
                )
            if training_config['train_an_under-trained_model']:
                model, df_history = load_dense_FCN_and_train_more(config['train_loop_path'], alg_id, pub_images,
                                                                  pub_labels)
                __training_loop__save_trained_model_and_training_history(
                    config['train_loop_path'],
                    model, df_history,
                    description_for_txt=alg_id + ' categorial_accuracy',
                    algorithm_id=alg_id
                )

    if algorithms_config['linear_classifier']['processing']:
        alg_id = 'linear_classifier'
        alg_config = algorithms_config[alg_id]
        if alg_config['training']:
            training_config = alg_config['training_config']
            if training_config['training_from_zero']:
                model, df_history = build_and_train_linear_classifier(pub_images, pub_labels)
                __training_loop__save_trained_model_and_training_history(
                    config['train_loop_path'],
                    model, df_history,
                    description_for_txt=alg_id + ' categorial_accuracy',
                    algorithm_id=alg_id
                )
            if training_config['train_an_under-trained_model']:
                model, df_history = load_linear_classifier_and_train_more(config['train_loop_path'], alg_id, pub_images,
                                                                          pub_labels)
                __training_loop__save_trained_model_and_training_history(
                    config['train_loop_path'],
                    model, df_history,
                    description_for_txt=alg_id + ' categorial_accuracy',
                    algorithm_id=alg_id
                )

    # summary pass
    load_and_show_dict = {}
    for alg_id, alg_config in algorithms_config.items():
        if alg_config['processing']:
            load_and_show_dict[alg_id] = alg_config['show_training_summary']

    __training_loop__load_and_show_training_summary(config['train_loop_path'], load_and_show_dict)


if __name__ == '__main__':
    execution_of_existing_functions(
        config={
            'recorded_train_data_filename': 'F:\\builded_datasets\\mnist\\train\\train_data_record.npy',
            'recorded_train_labels_filename': 'F:\\builded_datasets\\mnist\\train\\train_labels_record.npy',
            'train_loop_path': 'F:\\mnist_competition\\training_loop',
            'algorithms': {
                'dense_FCN': {'processing': False,
                              'training': False,
                              'training_config': {
                                  'training_from_zero': False,
                                  'train_an_under-trained_model': True
                              },
                              'show_training_summary': True},
                'linear_classifier': {'processing': False,
                                      'training': False,
                                      'training_config': {
                                          'training_from_zero': False,
                                          'train_an_under-trained_model': True
                                      },
                                      'show_training_summary': True},
                'handmade_VGG': {'processing': True,
                                 'training': False,
                                 'training_config': {
                                     'training_from_zero': False,
                                     'train_an_under-trained_model': True
                                 },
                                 'show_training_summary': True}}}
    )
