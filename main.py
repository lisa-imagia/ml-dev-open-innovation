import argparse
from scipy.misc import imread
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K


def create_model(num_classes, input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def train(model, img_rows, img_cols, num_classes, batch_size=128, epochs=3):
    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    # input_shape = (1, img_rows, img_cols)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Handwritten digit classification')
    parser.add_argument('--train', help='root path to nlst', action='store_true')
    parser.add_argument('--predict', type=str, help='classify given image (png)')
    args = parser.parse_args()

    batch_size = 128
    num_classes = 10
    epochs = 3

    # input image dimensions
    img_rows, img_cols = 28, 28
    input_shape = (1, img_rows, img_cols)

    if args.train:  # create and train model
        model = create_model(num_classes, input_shape)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        score = train(model, img_rows, img_cols, num_classes, batch_size, epochs)

        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        model.save_weights('weights.h5')

    elif args.predict:
        model = create_model(num_classes, input_shape)
        model.load_weights('weights.h5')
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        img = imread(args.predict)
        img = img.reshape((1, img_rows, img_cols, 1))
        pred = model.predict(img, batch_size=1)
        print("Predicted: ", np.argmax(pred))
