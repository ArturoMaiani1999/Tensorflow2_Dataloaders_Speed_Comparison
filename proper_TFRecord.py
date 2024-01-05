import os
import logging
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, activations, datasets
from tensorflow.train import Example, Features, Feature, BytesList, Int64List

import time

DATA_PATH = os.path.join('data', 'mnist')
AUTOTUNE = tf.data.AUTOTUNE
def fetch_data():
    mnist = datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_valid = x_train[:55900], x_train[55900:]
    y_train, y_valid = y_train[:55900], y_train[55900:]

    train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    valid_set = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    print('The dataset has been fetched.')
    return train_set, valid_set, test_set


def create_protobuf(image, label):
    image = tf.io.serialize_tensor(image).numpy()
    label = label.numpy()

    image_example = Example(
        features=Features(
            feature={
                'image': Feature(bytes_list=BytesList(value=[image])),
                'label': Feature(int64_list=Int64List(value=[label]))
            })).SerializeToString()

    return image_example


def save_datasets(n_files=15):
    train_set, valid_set, test_set = fetch_data()
    datasets = [train_set, valid_set, test_set]

    data_types = ['train', 'valid', 'test']
    for data_type in data_types:
        data_path = os.path.join(DATA_PATH, data_type)
        os.makedirs(data_path, exist_ok=True)

    train_paths = [os.path.join(DATA_PATH, 'train', f'train_{number_file}.tfrecord') for number_file in range(n_files)]
    valid_paths = [os.path.join(DATA_PATH, 'valid', f'valid_{number_file}.tfrecord') for number_file in range(n_files)]
    test_paths = [os.path.join(DATA_PATH, 'test', f'test_{number_file}.tfrecord') for number_file in range(n_files)]
    filepaths = [train_paths, valid_paths, test_paths]

    for filepath, dataset in zip(filepaths, datasets):
        writers = [tf.io.TFRecordWriter(path) for path in filepath]
        for index, (image, label) in dataset.enumerate():
            n_file = index % n_files
            example = create_protobuf(image, label)
            writers[n_file].write(example)
        print(f'The {data_type} dataset has been saved as TFRecord files.')

    return train_paths, valid_paths, test_paths


def get_filepaths():
    if os.path.exists(DATA_PATH):
        data_types = ['train', 'valid', 'test']
        filepaths = []
        for data_type in data_types:
            list_files = os.listdir(os.path.join(DATA_PATH, data_type))
            filepath = [os.path.join(DATA_PATH, data_type, file) for file in list_files]
            filepaths.append(filepath)

        train_paths = filepaths[0]
        valid_paths = filepaths[1]
        test_paths = filepaths[2]

    else:
        train_paths, valid_paths, test_paths = save_datasets()

    print('The paths for the files of the dataset have been retrieved.')

    return train_paths, valid_paths, test_paths


def preprocess(serialized_image):
    feature_descriptions = {
        'image': tf.io.FixedLenFeature([], tf.string, default_value=b''),
        'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
    }
    example = tf.io.parse_single_example(serialized_image,
                                         feature_descriptions)
    image = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=(28, 28, 1))
    return image, example['label']


def get_data(filepaths, shuffle_buffer_size=None, batch_size=32):
    list_files = tf.data.Dataset.list_files(filepaths)
    dataset = tf.data.TFRecordDataset(list_files, num_parallel_reads=AUTOTUNE)
    dataset = dataset.map(preprocess, num_parallel_calls=AUTOTUNE).cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)

    return dataset.batch(batch_size, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


class ResidualUnit(layers.Layer):

    def __init__(self, n_filters, size_filters=3, strides=2, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.size_filters = size_filters
        self.strides = strides
        self.activation = activations.get(activation)
        self.main_layers = [
            layers.Conv2D(n_filters, size_filters, strides=strides, padding='same', use_bias=False),
            layers.BatchNormalization(),
            self.activation,
            layers.Conv2D(n_filters, size_filters, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
        ]
        self.skip_layers = [
            layers.Conv2D(n_filters, 1, strides=strides, padding='same', use_bias=False),
            layers.BatchNormalization()
        ]

    def call(self, inputs):

        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)

        return self.activation(z + skip_z)

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_filters': self.n_filters,
            'size_filters': self.size_filters,
            'strides': self.strides,
            'activation': self.activation,
            'main_layers': self.main_layers,
            'skip_layers': self.skip_layers
        })
        return config

def new_model(train_set):

    tf.keras.backend.clear_session()
    tf.random.set_seed(15)

    normalizer = layers.Normalization(input_shape=[28, 28, 1])
    sample_data = train_set.take(1000).map(lambda x, y: x)
    normalizer.adapt(sample_data)

    model = models.Sequential([
        normalizer,
        layers.Conv2D(16, 5, padding='same', activation='relu', input_shape=[28, 28, 1]),
        layers.BatchNormalization(),
        ResidualUnit(32),
        ResidualUnit(64),
        layers.GlobalAvgPool2D(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])

    logging.info('The model has been created.')

    optimizer = optimizers.Nadam()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print('The model has been compiled.')

    return model

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])

    logging.info('The model has been created.')

    optimizer = optimizers.Nadam()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    print('The model has been compiled.')
    return model


def train_model(model, train_set, valid_set, test_set):

    # early_stopping_cb = callbacks.EarlyStopping(patience=15, restore_best_weights=True)
    #
    # callbacks_list = [early_stopping_cb]

    print('Training the model...')

    model.fit(train_set,
              validation_data=valid_set,
              epochs=5)
              # callbacks=callbacks_list)

    print('The model has been trained')

    accuracy_model = round(model.evaluate(test_set, verbose=0)[1], 6) * 100
    print(f'The accuracy of the model is {accuracy_model}%')

if __name__ == '__main__':
    train_filepaths, valid_filepaths, test_filepaths = get_filepaths()
    train_set = get_data(train_filepaths, shuffle_buffer_size=10000)
    valid_set = get_data(valid_filepaths)
    test_set = get_data(test_filepaths)

    # model = new_model(train_set)
    model = create_model()
    start = time.time()
    train_model(model, train_set, valid_set, test_set)
    end = time.time()
    print(f'The training took {end - start} seconds.')
    model.summary()

    x_test = test_set.take(1).map(lambda image, label: image)
    predictions = model.predict(x_test, use_multiprocessing=True)
    predictions = np.argmax(predictions, axis=1)

    plt.figure(figsize=(20, 10))

    index = 0
    for batch in x_test:
        for image in batch:
            plt.subplot(4, 8, index + 1)
            plt.imshow(image)
            plt.title(f'Class predicted: {predictions[index]}')
            plt.axis('off')
            index += 1

    plt.show()