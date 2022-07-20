import numpy as np
import tensorflow as tf
from tensorflow import keras

from attacker.constants import *


class AE_LOSSES:
    """
    Provide some loss functions for autoencoder attacker
    """
    CROSS_ENTROPY = 'identity'
    RE_RANK = 're_rank'

    # @staticmethod
    # def cross_entropy_loss_untargeted(classifier, target_vector, beta,
    #                                   input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
    #     """
    #
    #     :param input_shape: shape of an individual input image
    #     :param target_vector: target vector - one-hot vector of target label.
    #                         In untargeted attack, target_vector is viewed as original vector
    #     :param classifier: classification model
    #     :param target_label: target_label in one-hot coding
    #     :param beta: balance between target and L2 distance
    #     :return: loss
    #     """
    #
    #     def loss(origin_images, generated_images):
    #         batch_size = origin_images.shape[0]
    #         target_vectors = np.repeat(np.array([target_vector]), batch_size, axis=0)
    #         return (1 - beta) * tf.keras.losses.mean_squared_error(
    #             tf.reshape(origin_images, (batch_size, np.prod(input_shape))),
    #             tf.reshape(generated_images, (batch_size, np.prod(input_shape)))) - \
    #                beta * tf.keras.losses.categorical_crossentropy(classifier(generated_images), target_vectors)
    #
    #     return loss

    @staticmethod
    def cross_entropy_loss(classifier, target_vector, beta,
                           input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL), isuntargeted=False):
        """

        :param input_shape: shape of an individual input image
        :param target_vector: target vector - one-hot vector of target label
        :param classifier: classification model
        :param target_label: target_label in one-hot coding
        :param beta: balance between target and L2 distance
        :return: loss
        """
        sign_in_2nd_loss = 1 if isuntargeted is False else -1
        beta = sign_in_2nd_loss * beta

        def loss(origin_image, generated_image):
            batch_size = origin_image.shape[0]
            target_vectors = np.repeat(np.array([target_vector]), batch_size, axis=0)
            return (1 - beta) * tf.keras.losses.mean_squared_error(
                tf.reshape(origin_image, (batch_size, np.prod(input_shape))),
                tf.reshape(generated_image, (batch_size, np.prod(input_shape)))) + \
                   beta * tf.keras.losses.categorical_crossentropy(classifier(generated_image), target_vectors)

        return loss

    @staticmethod
    def binary_classifier_loss_untargeted(classifier, target_vector, beta,
                                          input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
        """

        :param classifier: target classifier
        :param target_vector: in untargeted attack, target_vector is viewed as original vector
        :param beta: balance between target and L2 distance
        :param input_shape: shape of an individual input image
        :return: loss
        """

        def loss(origin_images, generated_images):
            batch_size = origin_images.shape[0]
            target_vectors = np.repeat(target_vector, repeats=batch_size)

            return (1 - beta) * tf.keras.losses.mean_squared_error(
                tf.reshape(origin_images, (batch_size, np.prod(input_shape))),
                tf.reshape(generated_images,
                           (batch_size, np.prod(input_shape)))) - beta * tf.keras.losses.binary_crossentropy(
                classifier(generated_images), target_vectors)

        return loss

    @staticmethod
    def binary_classifier_loss(classifier, target_vector, beta,
                               input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL), isuntargeted=False):
        """

        :param classifier: target classifier
        :param target_vector: target vector is presented as [0] or [1]
        :param beta: balance between target and L2 distance
        :param input_shape: shape of an individual input image
        :return: loss
        """

        def loss(origin_images, generated_images):
            batch_size = origin_images.shape[0]
            target_vectors = np.repeat(target_vector, repeats=batch_size)

            return (1 - beta) * tf.keras.losses.mean_squared_error(
                (1 - beta) * tf.keras.losses.mean_squared_error(
                    tf.reshape(origin_images, (batch_size, np.prod(input_shape))),
                    tf.reshape(generated_images,
                               (batch_size, np.prod(input_shape))))) + beta * tf.keras.losses.binary_crossentropy(
                classifier(generated_images), target_vectors)

        return loss

    @staticmethod
    def ssim_loss(y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=2.0))

    @staticmethod
    def ssim_multiscale(y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim_multiscale(y_true, y_pred, max_val=2))

    @staticmethod
    def all_feature_ssim_loss(classifier, target_vector, beta,
                              input_shape=(MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)):
        def loss(origin_image, generated_image):
            batch_size = origin_image.shape[0]
            target_vectors = np.repeat(np.array([target_vector]), batch_size, axis=0)
            return (1 - beta) * (1 - tf.reduce_mean(
                tf.image.ssim(origin_image, generated_image,
                              max_val=2.0))) + beta * tf.keras.losses.categorical_crossentropy(
                classifier(generated_image), target_vectors)
            #
            #    AE_LOSSES.ssim_loss(
            # tf.reshape(origin_image, (batch_size, input_shape)),
            # tf.reshape(generated_image,
            #            (batch_size, input_shape)))

        return loss

    @staticmethod
    def re_rank_loss(classifier, target_vector, weight, alpha=1.5):
        """

        :param classifier: classification model
        :param target_label: target_label in one-hot coding
        :param weight: balance between target and L2 distance
        :param alpha: // TO_DO
        :return: loss
        """
        target_class = tf.argmax(target_vector)

        def loss(true_image, generated_image):
            re_rank = classifier(true_image).numpy()
            predicted_class = [np.argmax(re_rank_i) for re_rank_i in re_rank]

            for index, predicted_class_i in enumerate(predicted_class):
                re_rank[index, target_class] = alpha * re_rank[index, predicted_class_i]

            re_rank = np.array([(re_rank_i - np.mean(re_rank_i)) / np.std(re_rank_i) for re_rank_i in re_rank])

            batch_size = true_image.shape[0]
            true_image1 = tf.reshape(true_image, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))
            generated_image1 = tf.reshape(generated_image,
                                          (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))

            # print(a)
            return weight * tf.keras.losses.mean_squared_error(true_image1, generated_image1) + \
                   tf.keras.losses.mean_squared_error(classifier(generated_image), re_rank)

        return loss

    @staticmethod
    def border_loss(model, target_vector, beta, shape=(28, 28, 1)):
        def loss(combined_labels, generated_borders):
            borders = combined_labels[:, 0]
            borders = tf.reshape(borders, (borders.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
            internal_images = combined_labels[:, 1]
            internal_images = tf.reshape(internal_images,
                                         (internal_images.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL))
            batch_size = borders.shape[0]
            target_vectors = np.repeat(np.array([target_vector]), borders.shape[0], axis=0)

            generated_borders = generated_borders * borders

            border_pixels = combined_labels[:, 2]
            border_pixels = tf.reshape(border_pixels, (
                border_pixels.shape[0], MNIST_IMG_ROWS, MNIST_IMG_COLS, MNIST_IMG_CHL)) * borders

            return (1 - beta) * keras.losses.mean_squared_error(
                tf.reshape(border_pixels, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL)),
                tf.reshape(generated_borders, (batch_size, MNIST_IMG_ROWS * MNIST_IMG_COLS * MNIST_IMG_CHL))) + \
                   beta * keras.losses.categorical_crossentropy(target_vectors,
                                                                model(internal_images + generated_borders))

        return loss

    @staticmethod
    def general_loss(classifier, target_vector, beta, input_shape, is_untargeted, quality_loss_str, num_class):
        """

        :param classifier:  target white-box classifier
        :param target_vector: in untargeted attack, target_vector is viewed as original vector
        :param beta:
        :param input_shape: balance between target and L2 distance
        :param is_untargeted: True or False
        :param quality_loss: quality loss function as string type
        :return:
        """
        sign_in_2nd_loss = 1 if is_untargeted is False else -1
        weight_in_2nd_loss = sign_in_2nd_loss * beta
        quality_loss_function = None
        if quality_loss_str == LOSS_MSE:
            quality_loss_function = tf.keras.losses.mean_squared_error
        elif quality_loss_str == LOSS_SSIM:
            quality_loss_function = AE_LOSSES.ssim_loss
        elif quality_loss_str == LOSS_SSIM_MULTISCALE:
            quality_loss_function = AE_LOSSES.ssim_multiscale
        else:
            return None

        def loss(origin_images, generated_images):
            batch_size = origin_images.shape[0]
            if num_class > 1:
                target_vectors = np.repeat(np.array([target_vector]), batch_size, axis=0)
                adv_loss = tf.keras.losses.categorical_crossentropy
            else:
                target_vectors = np.repeat(target_vector, repeats=batch_size)
                adv_loss = tf.keras.losses.binary_crossentropy

            quality_loss = quality_loss_function(origin_images,
                                                 generated_images) if quality_loss_str != LOSS_MSE else quality_loss_function(
                tf.reshape(origin_images, (batch_size, np.prod(input_shape))),
                tf.reshape(generated_images, (batch_size, np.prod(input_shape))))
            return (1 - beta) * quality_loss + weight_in_2nd_loss * adv_loss(classifier(generated_images),
                                                                             target_vectors)

        return loss




# register custom objects
custom_losses = {
    'cross_entropy_loss': AE_LOSSES.cross_entropy_loss,
    'binary_classifier_loss_untargeted': AE_LOSSES.binary_classifier_loss_untargeted,
    'binary_classifier_loss': AE_LOSSES.binary_classifier_loss,
    'ssim_loss': AE_LOSSES.ssim_loss,
    'ssim_multiscale': AE_LOSSES.ssim_multiscale,
    're_rank_loss': AE_LOSSES.re_rank_loss,
    'border_loss': AE_LOSSES.border_loss,
    'general_loss': AE_LOSSES.general_loss,
}
