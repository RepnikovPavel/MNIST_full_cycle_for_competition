import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras


def build_VGG():
    # Input Layer
    input_height, input_width = 28, 28
    input_channels = 1
    input_layer = tf.keras.Input(shape=(input_channels, input_height, input_width))

    # Convolutional Layer #1 and Pooling Layer #1
    conv1_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(input_layer)
    conv1_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(conv1_1)
    pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(conv1_2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2_1 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(pool1)
    conv2_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(conv2_1)
    pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(conv2_2)

    # Convolutional Layer #3 and Pooling Layer #3
    conv3_1 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(pool2)
    conv3_2 = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(conv3_1)
    pool3 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(conv3_2)

    # Convolutional Layer #4 and Pooling Layer #4
    conv4_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(pool3)
    conv4_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(conv4_1)
    pool4 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(conv4_2)

    # Convolutional Layer #5 and Pooling Layer #5
    conv5_1 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(pool4)
    conv5_2 = tf.keras.layers.Conv2D(filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)(conv5_1)
    pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same")(conv5_2)

    # FC Layers
    pool5_flat = tf.keras.layers.Flatten()(pool5)
    FC1 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu
                                )(pool5_flat)
    b_n1 = tf.keras.layers.BatchNormalization()(FC1)
    drop_out1 = keras.layers.Dropout(0.4)(b_n1)

    FC2 = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu

                                )(drop_out1)
    b_n2 = tf.keras.layers.BatchNormalization()(FC2)
    drop_out2 = keras.layers.Dropout(0.4)(b_n2)


    FC3 = tf.keras.layers.Dense(units=1000, activation=tf.nn.relu
                                )(drop_out2)
    b_n3 = tf.keras.layers.BatchNormalization()(FC3)
    drop_out3 = keras.layers.Dropout(0.3)(b_n3)

    FC4 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu
                                # bias_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                                # kernel_regularizer=tf.keras.regularizers.L1L2(0, 0.01),
                                # activity_regularizer=tf.keras.regularizers.L1L2(0, 0.01)
                                )(drop_out3)

    b_n4 = tf.keras.layers.BatchNormalization()(FC4)

    FC5 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(b_n4)

    model = tf.keras.Model(inputs=input_layer, outputs=FC5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                         amsgrad=False)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=[keras.metrics.categorical_accuracy])

    return model


def build_and_train_VGG(images, labels):
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
    images = images[:, np.newaxis]
    images = images / 255.0
    images = (images - np.mean(images)) / np.std(images)

    model = build_VGG()
    EPOCHS = 30

    print("\n")
    print("#####################\n")
    print("VGG starting to train\n")
    print("#####################\n")

    history = model.fit(images, one_hot_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=1, batch_size=512)

    pandas_df_history = pd.DataFrame(history.history)
    pandas_df_history['epoch'] = history.epoch

    pandas_df_history.rename(columns={'epoch': 'x', 'categorical_accuracy': 'y1', 'val_categorical_accuracy': 'y2'},
                             inplace=True)

    return model, pandas_df_history


def load_VGG_and_train_more(training_loop_path, alg_id, images, labels):
    """
        загружается уже обучаемая раннее модель, тренируестя, потом информация о тренировке дозапиисывется в
        файл с иторией о тренировке, дообученная модель перезаписывается на место старой модели
    """
    one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=10)
    images = images[:, np.newaxis]
    images = images / 255.0
    images = (images - np.mean(images)) / np.std(images)

    model = keras.models.load_model(training_loop_path +'\\'+ alg_id)

    EPOCHS = 10

    print("\n")
    print("#####################\n")
    print("VGG starting to train\n")
    print("#####################\n")

    history = model.fit(images, one_hot_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=1, batch_size=512)

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
