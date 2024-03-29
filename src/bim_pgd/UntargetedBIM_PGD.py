'''
Untargeted attack
'''
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow
from tqdm import tqdm

from src.utils import utils
from src.utils.attack_logger import AttackLogger

logger = AttackLogger.get_logger()


class UntargetedBIM_PGD:
    def __init__(self
                 , X  # (batch, width, height, channel), must be recognized correctly by the target classifier
                 , Y  # 1D, just contain labels
                 , target_classifier
                 , epsilon=1 / 255  # a positive number, from 0 to 1
                 , batch_size=1000  # a positive number, integer
                 , max_iteration=10  # a positive number, integer
                 , lower_pixel_value=0  # the valid lower range of a pixel
                 , upper_pixel_value=1  # the valid upper range of a pixel
                 , max_ball=1 / 255
                 ):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.target_classifier = target_classifier

        self.lower_pixel_value = lower_pixel_value
        self.upper_pixel_value = upper_pixel_value
        self.max_norm = max_ball

        self.final_advs = None
        self.final_true_labels = None
        self.final_origin = None

    def attack(self):
        '''
        Attack
        '''
        # logger.debug(f'ep={self.epsilon}, max_iter={self.max_iteration}')
        X = self.X
        ep = self.epsilon
        batch_size = self.batch_size
        max_iteration = self.max_iteration
        Y = self.Y
        target_classifier = self.target_classifier
        upper_pixel_value = self.upper_pixel_value
        lower_pixel_value = self.lower_pixel_value
        max_norm = self.max_norm
        n_batchs = int(np.ceil(len(X) / batch_size))
        for idx in tqdm(range(n_batchs), desc=f'Attacking batch of {self.batch_size} images:...'):
        # for idx in range(n_batchs):
            '''
            Computing batch range
            '''
            start = idx * batch_size
            end = start + batch_size
            if end > len(X):
                end = len(X)
            #logger.debug(f'[{idx + 1} / {n_batchs}] Attacking from {start} to {end}')

            '''
            Attack
            '''
            iter = max_iteration
            advs = X[start: end].copy()
            tensor_advs = tensorflow.convert_to_tensor(advs)
            min = advs - max_norm
            max = advs + max_norm

            while iter >= 1:
                #logger.debug(f'\tIteration {max_iteration - iter + 1}')
                # compute gradient
                gradient, tape = utils.compute_gradient_batch(inputs=tensor_advs,
                                                              target_neurons=Y[start: end],
                                                              target_classifier=target_classifier)
                grad = tape.gradient(gradient, tensor_advs)
                grad = np.asarray(grad)

                # find not-satisfied indexes
                batch_adv_label = np.argmax(target_classifier.predict(advs), axis=1)
                isDiffLabels = batch_adv_label != Y[start: end]
                not_satisfied = np.where(isDiffLabels == False)[0]

                advs[not_satisfied] = advs[not_satisfied] - ep * np.sign(grad[not_satisfied])
                '''
                clip to max_norm
                '''
                con1 = min[not_satisfied] <= advs[not_satisfied]
                con2 = advs[not_satisfied] <= max[not_satisfied]
                in_bound_matrix = con1 & con2
                # print(f'before {np.sum(np.invert(con1 & con2))}')
                advs[not_satisfied] = advs[not_satisfied] * in_bound_matrix + \
                                      X[start:end][not_satisfied] * np.invert(in_bound_matrix) - \
                                      np.sign(grad[not_satisfied]) * (max_norm) * np.invert(in_bound_matrix)
                '''
                clip to valid range
                '''
                advs[not_satisfied] = np.clip(advs[not_satisfied], lower_pixel_value, upper_pixel_value)

                not_satisfied_adv_labels = np.argmax(target_classifier.predict(advs[not_satisfied]), axis=1)
                batch_adv_label[not_satisfied] = not_satisfied_adv_labels
                isDiffLabels = batch_adv_label != Y[start: end]

                iter -= 1
                sr = np.sum(isDiffLabels) / len(Y[start: end])

                if sr == 1 or iter <= 0:
                    #logger.debug(f'\tAttacking this batch done. Success rate = {sr * 100}%')

                    satisfied = np.where(isDiffLabels)[0]
                    if self.final_advs is None:
                        self.final_advs = advs[satisfied]
                        self.final_origin = X[start:end][satisfied]
                        self.final_true_labels = Y[start:end][satisfied]
                    else:
                        self.final_advs = np.concatenate((self.final_advs, advs[satisfied]), axis=0)
                        self.final_origin = np.concatenate((self.final_origin, X[start:end][satisfied]), axis=0)
                        self.final_true_labels = np.concatenate((self.final_true_labels, Y[start:end][satisfied]),
                                                                axis=0)
                    break
                else:
                    tensor_advs = tensorflow.convert_to_tensor(advs)

        utils.confirm_adv_attack(self.target_classifier, self.final_advs, self.final_origin, self.final_true_labels,
                                 self.X)
        return self.final_origin, self.final_advs, self.final_true_labels


if __name__ == '__main__':
    TARGET_CLASSIFIER_PATH = 'D:\Things\PyProject\AdvDefense\data\CIFAR10\cifar10_classifier_I.h5'
    target_classifier = keras.models.load_model(TARGET_CLASSIFIER_PATH)

    trainingsetX = np.load(
        'D:\Things\PyProject\AdvDefense\data\CIFAR10\cifar10_test_data.npy')
    trainingsetY = np.load(
        'D:\Things\PyProject\AdvDefense\data\CIFAR10\cifar10_sparse_test_label.npy').reshape(-1)

    '''
    Just attack on correctly predicted images
    '''
    pred = target_classifier.predict(trainingsetX)
    pred = np.argmax(pred, axis=1)
    true_indexes = np.where(pred == trainingsetY)[0][:1000]
    print(f'Number of correctly predicted images = {len(true_indexes)}')

    attacker = UntargetedBIM_PGD(X=trainingsetX[true_indexes][:100],
                                 Y=trainingsetY[true_indexes][:100],
                                 target_classifier=target_classifier)
    final_origin, final_advs, final_true_labels = attacker.attack()
    utils.exportAttackResult(
        output_folder='D:\Things\PyProject\AdvDefense\data',
        name='bim',
        target_classifier=target_classifier,
        final_advs=final_advs,
        final_origin=final_origin,
        final_true_labels=final_true_labels,
        logger=logger
    )

    a = np.argmax(target_classifier.predict(final_origin), axis=1).reshape(-1)
    print((a == final_true_labels).sum() / len(a))

    a = np.argmax(target_classifier.predict(final_advs), axis=1).reshape(-1)
    print((a == final_true_labels).sum() / len(a))