import enum
import random

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras import Sequential
import numpy as np
from attacker.constants import CLASSIFIER_PATH
from data_preprocessing.mnist import MnistPreprocessing
from utility.mylogger import MyLogger

logger = MyLogger.getLog()
from utility.config import *


class RANKING_ALGORITHM(enum.Enum):
    ABS = 1
    COI = 2
    CO = 3


class feature_ranker:
    def __init__(self):
        return

    @staticmethod
    def compute_gradient_wrt_features(input: tf.Tensor,
                                      target_neuron: int,
                                      classifier: tf.keras.Sequential):
        """Compute gradient wrt features.
        Args:
            input: a tensor (shape = `[1, height, width, channel`])
            target_neuron: the index of the neuron on the output layer needed to be differentiated
            classifier: a sequential model
        Returns:
            gradient: ndarray (shape = `[height, width, channel`])
        """
        with tf.GradientTape() as tape:
            tape.watch(input)
            prediction_at_target_neuron = classifier(input)[0][target_neuron]
        gradient = tape.gradient(prediction_at_target_neuron, input)
        gradient = gradient.numpy()[0]
        return gradient

    @staticmethod
    def compute_gradient_batch(inputs: tf.Tensor, target_neuron: int, classifier):
        '''
        :param inputs:
        :type inputs:
        :param target_neuron:
        :type target_neuron:
        :param classifier:
        :type classifier:
        :return:
        :rtype:
        '''
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            predictions_at_target_neuron = classifier(inputs)[:, target_neuron]
        gradient = tape.gradient(predictions_at_target_neuron, inputs)
        return gradient.numpy()

    @staticmethod
    def sequence_ranking(generated_adv, origin_image, border_index, target_label, classifier, num_expected_features=1,
                         num_classes=10):
        return np.array([*range(np.prod(generated_adv.shape, dtype=np.int64))])

    @staticmethod
    def sequence_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        batch_size = generated_advs.shape[0]
        single_adv = np.array(generated_advs[0])
        single_indexes = np.array([*range(np.prod(single_adv.shape, dtype=np.int64))])
        return np.repeat([single_indexes], repeats=batch_size, axis=0), np.array(generated_advs)

    @staticmethod
    def jsma_KA_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        dF_t = []
        dF_rest = []
        num_elements = np.prod(attack_config.input_shape)
        for i in range(num_class):
            dF_i = feature_ranker.compute_gradient_batch(
                inputs=tf.convert_to_tensor(generated_advs.reshape((-1, *attack_config.input_shape))),
                classifier=classifier, target_neuron=target_label)
            if i != target_label:
                dF_rest.append(dF_i.reshape((-1, num_elements)))
            else:
                dF_t = dF_i.reshape((-1, num_elements))
        dF_rest = np.asarray(dF_rest)
        dF_rest = np.rollaxis(dF_rest, axis=1, start=0)
        dF_t = np.asarray(dF_t)
        advs_flatten = generated_advs.reshape((-1, num_elements))
        oris_flatten = origin_images.reshape((-1, num_elements))
        SXs = []

        for index in range(np.prod(attack_config.input_shape)):
            dF_t_i = dF_t[:, index]
            sum_dF_rest_i = np.sum(dF_rest[:, :, index], axis=1)
            compare = advs_flatten[:, index] > oris_flatten[:, index]
            compare_opposite = ~ compare
            compare = np.array(compare, dtype=int)

            positive_rank = abs(dF_t_i) * abs(sum_dF_rest_i)
            negative_rank = -1 * 1 / (positive_rank + 0.1)

            init = np.array(positive_rank)
            init[compare is True and (dF_t_i < 0 or sum_dF_rest_i > 0)] = negative_rank[
                compare is True and dF_t_i < 0 and sum_dF_rest_i > 0]
            init[compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)] = negative_rank[
                compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)]
            SXs.append(init)

        SXs = np.asarray(SXs).T
        ranking_results = []
        value_results = []
        for index in range(len(diff_pixels)):
            SX = SXs[index][diff_pixels[index]]
            a_argsort = np.argsort(SX)
            ranking_results.append(np.array(diff_pixels[index])[a_argsort])
            value_results.append(SX[a_argsort])
        return np.asarray(ranking_results), np.asarray(value_results)

    @staticmethod
    def jsma_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        dF_t = []
        dF_rest = []
        num_elements = np.prod(attack_config.input_shape)
        for i in range(num_class):
            dF_i = feature_ranker.compute_gradient_batch(
                inputs=tf.convert_to_tensor(generated_advs.reshape((-1, *attack_config.input_shape))),
                classifier=classifier, target_neuron=target_label)
            if i != target_label:
                dF_rest.append(dF_i.reshape((-1, num_elements)))
            else:
                dF_t = dF_i.reshape((-1, num_elements))
        dF_rest = np.asarray(dF_rest)
        dF_rest = np.rollaxis(dF_rest, axis=1, start=0)
        dF_t = np.asarray(dF_t)
        advs_flatten = generated_advs.reshape((-1, num_elements))
        oris_flatten = origin_images.reshape((-1, num_elements))
        SXs = []

        for index in range(np.prod(attack_config.input_shape)):
            dF_t_i = dF_t[:, index]
            sum_dF_rest_i = np.sum(dF_rest[:, :, index], axis=1)
            compare = advs_flatten[:, index] > oris_flatten[:, index]
            compare_opposite = ~ compare
            compare = np.array(compare, dtype=int)

            positive_rank = abs(dF_t_i) * abs(sum_dF_rest_i)
            negative_rank = np.zeros_like(positive_rank)

            init = np.array(positive_rank)
            init[compare is True and (dF_t_i < 0 or sum_dF_rest_i > 0)] = negative_rank[
                compare is True and dF_t_i < 0 and sum_dF_rest_i > 0]
            init[compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)] = negative_rank[
                compare is False and (dF_t_i > 0 or sum_dF_rest_i < 0)]
            SXs.append(init)

        SXs = np.asarray(SXs).T
        ranking_results = []
        value_results = []
        for index in range(len(diff_pixels)):
            SX = SXs[index][diff_pixels[index]]
            a_argsort = np.argsort(SX)
            ranking_results.append(np.array(diff_pixels[index])[a_argsort])
            value_results.append(SX[a_argsort])
        return np.asarray(ranking_results), np.asarray(value_results)

    @staticmethod
    def coi_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        dF_t = feature_ranker.compute_gradient_batch(
            inputs=tf.convert_to_tensor(generated_advs.reshape((-1, *attack_config.input_shape))),
            classifier=classifier, target_neuron=target_label)
        dF_t = dF_t.reshape(-1, np.prod(generated_advs[0].shape))
        score_matrices = dF_t * generated_advs
        score_matrices = score_matrices

        ranking_results = []
        value_results = []
        for index in range(len(diff_pixels)):
            SX = score_matrices[index][diff_pixels[index]]
            a_argsort = np.argsort(SX)
            ranking_results.append(np.array(diff_pixels[index])[a_argsort])
            value_results.append(SX[a_argsort])
        return np.asarray(ranking_results), np.asarray(value_results)

    @staticmethod
    def random_ranking_batch(generated_advs, origin_images, target_label, classifier, diff_pixels, num_class):
        for index in range(len(diff_pixels)):
            random.shuffle(diff_pixels[index])
        return diff_pixels, None

    @staticmethod
    def get_important_pixel_vetcan(image, classifier):

        important_pixels = []
        score = []
        changed_pixel_values = [2., 3.]
        tmp_image = np.array([image])
        predict_true = classifier.predict(tmp_image)[0]
        y_true = np.argmax(predict_true)
        confident_true = np.max(predict_true)

        for index in range(np.prod(image.shape)):
            row, col = int(index // 28), int(index % 28)
            tmp_pixel_value = tmp_image[0][row, col][0]
            for changed_pixel_value in changed_pixel_values:
                tmp_image[0][row, col] = changed_pixel_value
                predict = classifier.predict(tmp_image)[0]
                y_pred = np.argmax(predict)
                if y_pred != y_true:
                    print(f'pred: {y_pred}, true: {y_true}')
                    important_pixels.append(index)
                    score.append(abs(np.max(predict) - confident_true))
                    break
            tmp_image[0][row, col] = tmp_pixel_value

        return np.array(important_pixels), np.array(score)


if __name__ == '__main__':
    classifier = keras.models.load_model(CLASSIFIER_PATH + '/pretrained_mnist_cnn1.h5')
    LABEL = 4
    if isinstance(classifier, Sequential):
        # get a seed
        (trainX, trainY), (testX, testY) = mnist.load_data()
        pre_mnist = MnistPreprocessing(
            trainX=trainX,
            trainY=trainY,
            testX=testX,
            testY=testY,
            start=0,
            end=100,
            removed_labels=None)
        trainX, trainY, _, _ = pre_mnist.preprocess_data()

        # consider an input vector
        important_features = feature_ranker.find_important_features_of_a_sample(
            input_image=trainX[0],
            n_rows=MNIST_IMG_ROWS,
            n_cols=MNIST_IMG_COLS,
            n_channels=MNIST_IMG_CHL,
            n_important_features=50,
            algorithm=RANKING_ALGORITHM.ABS,
            gradient_label=3,
            classifier=classifier
        )
        logger.debug(important_features.shape)
        feature_ranker.highlight_important_features(
            important_features=important_features,
            input_image=trainX[0]
        )

        # consider input vectors
        important_features = feature_ranker.find_important_features_of_samples(
            input_images=trainX[0:100],
            n_rows=MNIST_IMG_ROWS,
            n_cols=MNIST_IMG_COLS,
            n_channels=MNIST_IMG_CHL,
            n_important_features=3,
            algorithm=RANKING_ALGORITHM.COI,
            gradient_label=1,
            classifier=classifier
        )
        logger.debug(important_features.shape)
        feature_ranker.highlight_important_features(
            important_features=important_features,
            input_image=trainX[1]
        )
