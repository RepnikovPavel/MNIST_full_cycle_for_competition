import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_dense_FCN():
    model = keras.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(100, activation=None, use_bias=True,
                     bias_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     kernel_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     activity_regularizer=tf.keras.regularizers.L1L2(0, 0.01)
                     ),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.3),
        layers.Dense(100, activation=None, use_bias=True,
                     bias_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     kernel_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     activity_regularizer = tf.keras.regularizers.L1L2(0, 0.01)
                     ),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.3),
        layers.Dense(100, activation=None, use_bias=True,
                     bias_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     kernel_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     activity_regularizer = tf.keras.regularizers.L1L2(0, 0.01)
                     ),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.3),
        layers.Dense(100, activation=None, use_bias=True,
                     bias_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     kernel_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     activity_regularizer = tf.keras.regularizers.L1L2(0, 0.01)
                    ),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        keras.layers.Dropout(rate=0.3),
        layers.Dense(100, activation=None, use_bias=True,
                     bias_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     kernel_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                     activity_regularizer=tf.keras.regularizers.L1L2(0, 0.01)
                    ),
        tf.keras.layers.LeakyReLU(alpha=0.3),
        tf.keras.layers.BatchNormalization(),
        layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                         amsgrad=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[keras.metrics.categorical_accuracy])
    return model


def build_and_train_dense_FCN(images, labels):
    """
        вход: images, lables - np.array
        выход: обченная модель keras.sequential и информация о тренировке в виде pd_dataframe

        производится reshape=(60 000,784)  и нормализация данных = (x-mean(x) / std(x)
        метки переводятся в onehot representation
        на выходе FCN считается кроссэнтропийная функция ошибки
        обученная сеть записывается на диск

        название final_save_path мотивируется тем, что могут быть промежуточные save path (если нужно будет сохранять
        check points )

        в конченой df_table столбик 'x' - это данные для отрисовки(history.epoch)
                            столбик 'y1'- это метрика на train
                            столбик 'y2'- это метрика на validation
    """

    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    images = np.reshape(images, newshape=(np.shape(images)[0], 784))
    images = images / 255.0
    images = (images - np.mean(images)) / np.std(images)

    model = build_dense_FCN()

    EPOCHS = 300

    print("\n")
    print("###########################\n")
    print("dense_FCN starting to train\n")
    print("###########################\n")

    def scheduler(epoch, lr):
        if epoch % 100 == 0:
            return lr * tf.math.exp(-0.69)
        else:
            return lr

    change_learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(images, one_hot_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=1, batch_size=1024,
                        callbacks=[change_learning_rate]
                        )

    pandas_df_history = pd.DataFrame(history.history)
    pandas_df_history['epoch'] = history.epoch

    pandas_df_history.rename(columns={'epoch': 'x', 'categorical_accuracy': 'y1', 'val_categorical_accuracy': 'y2'},
                             inplace=True)

    return model, pandas_df_history


def load_dense_FCN_and_train_more(training_loop_path, alg_id, images, labels):
    """
        загружается уже обучаемая раннее модель, тренируестя, потом информация о тренировке дозапиисывется в
        файл с иторией о тренировке, дообученная модель перезаписывается на место старой модели
    """
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    images = np.reshape(images, newshape=(np.shape(images)[0], 784))
    images = images / 255.0
    images = (images - np.mean(images)) / np.std(images)

    model = keras.models.load_model(training_loop_path +'\\'+ alg_id)

    EPOCHS = 50

    print("\n")
    print("###########################\n")
    print("dense_FCN starting to train\n")
    print("###########################\n")

    def scheduler(epoch, lr):
        if epoch % 100 == 0:
            return lr * tf.math.exp(-0.69)
        else:
            return lr

    change_learning_rate = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(images, one_hot_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=1, batch_size=1024,
                        callbacks=[change_learning_rate]
                        )

    pandas_df_history = pd.DataFrame(history.history)
    pandas_df_history['epoch'] = history.epoch

    pandas_df_history.rename(columns={'epoch': 'x', 'categorical_accuracy': 'y1', 'val_categorical_accuracy': 'y2'},
                             inplace=True)
    # теперь сконкатинируем pd_dataframe
    # попутно добавив к только что полученной dataframe смещение по эпохам относительно старой
    # dataframe, чтобы можно было построить график за весь период

    ################
    # если в будущем будет делаться cross валидация то это тоже надо будет учитывать
    # пока это не обрабатывается
    # ################

    df_filename = training_loop_path + '\\' + alg_id + '_history.csv'
    print(df_filename)
    previous_df = pd.read_csv(df_filename)
    # номер эпохи начинается от нуля , поэтому добавлять нужно будет x + 1
    epoch_offset = previous_df['x'].iloc[-1] + 1
    pandas_df_history['x'] += epoch_offset

    new_df_history = pd.concat([previous_df, pandas_df_history])

    return model, new_df_history
