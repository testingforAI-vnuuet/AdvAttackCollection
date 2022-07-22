from __future__ import absolute_import

from numpy.core.multiarray import ndarray
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Conv2DTranspose, ReLU, Input, Concatenate
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, History

from src.hpba_v2_2_3.src.attacker.losses import *
from src.hpba_v2_2_3.src.data_preprocessing.mnist import MnistPreprocessing
from src.hpba_v2_2_3.src.utility.config import *
from src.hpba_v2_2_3.src.utility.statistics import *
from src.utils.attack_logger import AttackLogger

logger = AttackLogger.get_logger()


class MnistAutoEncoder:
    def __init__(self, input_range=None):
        """
              if input_range is None or (0, 1), ignore and set activation function in last layers as sigmoid
              else input_range 2-element tuple, support adaptive wide range activation function.
                  min_value = input_range[0]
                  max_value = input_range[1]

              """
        # self.input_range = input_range
        self.last_activation_function = 'sigmoid'
        if input_range is not None and input_range != (0, 1):
            self.last_activation_function = wrap_range_custom_activation(min_value=input_range[0],
                                                                         max_value=input_range[1])
        # pass

    def train(self,
              auto_encoder: keras.models.Model,
              attacked_classifier: keras.models.Model,
              loss,
              epochs: int,
              batch_size: int,
              epsilon: float,
              target_label: int,
              training_set: np.ndarray,
              output_model_path: str,
              is_fit=True):
        # save the best model during training
        adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
        earlyStopping = EarlyStopping(monitor='loss', patience=30, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(output_model_path, save_best_only=True, monitor='loss', mode='min')
        target_label_one_hot = keras.utils.to_categorical(target_label, MNIST_NUM_CLASSES, dtype='float32')
        auto_encoder.compile(optimizer=adam,
                             loss=loss(
                                 classifier=attacked_classifier,
                                 epsilon=epsilon,
                                 target_label=target_label_one_hot)
                             )
        if is_fit:
            auto_encoder.fit(x=training_set,
                             y=training_set,
                             epochs=epochs,
                             batch_size=batch_size,
                             callbacks=[earlyStopping, mcp_save])
        return auto_encoder

    def get_architecture(self):
        input_img = keras.layers.Input(shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(1, (3, 3), activation=self.last_activation_function, padding='same')(x)
        return keras.models.Model(input_img, decoded)

    def apdative_architecture(self, input_shape):
        input_img = keras.layers.Input(shape=(input_shape))
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        encoded = keras.layers.MaxPooling2D((2, 2), padding='same')(x)

        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        decoded = keras.layers.Conv2D(input_shape[-1], (3, 3), activation=self.last_activation_function,
                                      padding='same')(x)
        return keras.models.Model(input_img, decoded)

    def get_3d_atchitecture(self, input_shape):
        dae_inputs = Input(shape=input_shape, name='dae_input')
        conv_block1 = conv_block(dae_inputs, 32, 3)
        conv_block2 = conv_block(conv_block1, 64, 3)
        conv_block3 = conv_block(conv_block2, 128, 3)
        conv_block4 = conv_block(conv_block3, 256, 3)
        conv_block5 = conv_block(conv_block4, 256, 3, 1)

        deconv_block1 = deconv_block(conv_block5, 256, 3)
        merge1 = Concatenate()([deconv_block1, conv_block3])
        deconv_block2 = deconv_block(merge1, 128, 3)
        merge2 = Concatenate()([deconv_block2, conv_block2])
        deconv_block3 = deconv_block(merge2, 64, 3)
        merge3 = Concatenate()([deconv_block3, conv_block1])
        deconv_block4 = deconv_block(merge3, 32, 3)

        final_deconv = Conv2DTranspose(filters=3,
                                       kernel_size=3,
                                       padding='same')(deconv_block4)

        dae_outputs = Activation(self.last_activation_function, name='dae_output')(final_deconv)
        return tf.keras.Model(dae_inputs, dae_outputs, name='dae')

    def compute_balanced_point(self,
                               auto_encoder: Model,
                               attacked_classifier: Sequential,
                               loss,
                               train_data: ndarray,
                               target_label: int):
        target_label_one_hot = keras.utils.to_categorical(target_label, MNIST_NUM_CLASSES, dtype='float32')
        # compute the distance term
        auto_encoder.compile(loss=loss(
            classifier=attacked_classifier,
            epsilon=0,
            target_label=target_label_one_hot)
        )
        auto_encoder.fit(x=train_data,
                         y=train_data,
                         epochs=1,
                         batch_size=len(train_data))
        loss_distance = auto_encoder.history.history['loss'][0]

        # compute the probability term
        auto_encoder.compile(loss=loss(
            classifier=attacked_classifier,
            epsilon=1,
            target_label=target_label_one_hot)
        )
        auto_encoder.fit(x=train_data,
                         y=train_data,
                         epochs=1,
                         batch_size=len(train_data))
        loss_probability = auto_encoder.history.history['loss'][0]

        return loss_distance / (loss_distance + loss_probability)

    def plot(self, history: History, path: str):
        plt.plot(history.history['loss'])
        plt.title('Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.legend(['train', '_test'], loc='upper left')
        # plt.show()
        plt.savefig(path)
        plt.clf()


def conv_block(x, filters, kernel_size, strides=2):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def deconv_block(x, filters, kernel_size):
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        strides=2,
                        padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


if __name__ == '__main__':
    START_SEED, END_SEED = 0, 1000
    TARGET = 7
    AE_LOSS = AE_LOSSES.cross_entropy_loss
    CNN_MODEL = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')
    AE_MODEL = CLASSIFIER_PATH + '/xxxx.h5'
    FIG_PATH = CLASSIFIER_PATH + '/xxxx.png'

    # load dataset
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    pre_mnist = MnistPreprocessing(train_X, train_Y, test_X, test_Y, START_SEED, END_SEED, TARGET)
    train_X, train_Y, test_X, test_Y = pre_mnist.preprocess_data()
    countSamples(probability_vector=train_Y, n_class=MNIST_NUM_CLASSES)

    # train an autoencoder
    ae_trainer = MnistAutoEncoder()
    ae = ae_trainer.get_architecture()
    ae.summary()
    ae_trainer.train(
        auto_encoder=ae,
        attacked_classifier=CNN_MODEL,
        loss=AE_LOSS,
        epochs=2,
        batch_size=256,
        training_set=train_X,
        epsilon=0.01,
        output_model_path=AE_MODEL,
        target_label=TARGET)

    # compute the balance point
    balanced_point = ae_trainer.compute_balanced_point(auto_encoder=ae,
                                                       attacked_classifier=CNN_MODEL,
                                                       loss=AE_LOSS,
                                                       train_data=train_X,
                                                       target_label=TARGET)
    print(balanced_point)
