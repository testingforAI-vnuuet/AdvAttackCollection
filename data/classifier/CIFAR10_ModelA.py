import tensorflow

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD
from tensorflow.python.keras.utils.np_utils import to_categorical


def define_model():
    '''
    https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def load_CIFAR10_dataset():
    # load dataset
    (trainX, trainY_label), (testX, testY_label) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY_label)
    testY = to_categorical(testY_label)
    return trainX, trainY, testX, testY, testY_label.reshape(-1), trainY_label.reshape(-1)

# scale pixels
def prep_pixels(train, test):
    train_norm = train.astype('float32')
    # convert from integers to floats
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

if __name__ == '__main__':
    model = define_model()
    trainX, trainY, testX, testY, _, _ = load_CIFAR10_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    history = model.fit(trainX, trainY, epochs=100, batch_size=512, validation_data=(testX, testY))
    _, acc = model.evaluate(testX, testY, verbose=0)
    model.save('/Users/ducanhnguyen/Documents/HPBA/src/classifier/CIFAR10_ModelA')