'''
Untargeted attack
Reference: Boosting Adversarial Attacks with Momentum, 2018
'''

import numpy as np
import tensorflow
from tensorflow import keras
from tqdm import tqdm
from src.attack_config.config import logger
from src.utils import utils


class UntargetedMI_FGSM:
    def __init__(self
                 , X  # (batch, width, height, channel), must be recognized correctly by the target classifier
                 , Y  # 1D, just contain labels
                 , target_classifier
                 , alpha=1 / 255  # a positive number, from 0 to 1
                 , batch_size=200  # a positive number, integer
                 , max_iteration=10  # a positive number, integer
                 , lower_pixel_value=0  # the valid lower range of a pixel
                 , upper_pixel_value=1  # the valid upper range of a pixel
                 , decay_factor = 1
                 ):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.target_classifier = target_classifier

        self.lower_pixel_value = lower_pixel_value
        self.upper_pixel_value = upper_pixel_value

        self.final_advs = None
        self.final_true_labels = None
        self.final_origin = None

        self.decay_factor = decay_factor

    def attack(self):
        '''
        Attack
        '''
        X = self.X
        alpha = self.alpha
        batch_size = self.batch_size
        max_iteration = self.max_iteration
        Y = self.Y
        target_classifier = self.target_classifier
        upper_pixel_value = self.upper_pixel_value
        lower_pixel_value = self.lower_pixel_value
        decay_factor = self.decay_factor

        n_batchs = int(np.ceil(len(X) / batch_size))
        # for i in tqdm(range(0, len(x), self.batch_size), desc='Attacking:...'):
        for idx in tqdm(range(n_batchs), desc=f'Attacking batch of {self.batch_size} images:...'):
            '''
            Computing batch range
            '''
            start = idx * batch_size
            end = start + batch_size
            if end > len(X):
                end = len(X)
            # logger.debug(f'[{idx + 1} / {n_batchs}] Attacking from {start} to {end}')

            '''
            Attack
            '''
            iter = max_iteration
            advs = X[start: end].copy()
            velocity = np.zeros(shape=(end - start, advs.shape[1], advs.shape[2], advs.shape[3])) # (batch, width, height, channel)
            tensor_advs = tensorflow.convert_to_tensor(advs)

            while iter >= 1:
                # logger.debug(f'\tIteration {max_iteration - iter + 1}')

                # compute gradient
                gradient, tape = utils.compute_gradient_batch(inputs=tensor_advs,
                                                             target_neurons=Y[start: end],
                                                             target_classifier=target_classifier)
                grad = tape.gradient(gradient, tensor_advs)
                grad = np.asarray(grad)
                #

                # find not-satisfied indexes
                batch_adv_label = np.argmax(target_classifier.predict(advs), axis=1)
                isDiffLabels = batch_adv_label != Y[start: end]
                not_satisfied = np.where(isDiffLabels == False)[0]

                '''
                '''
                for kdx in range(len(not_satisfied)):
                    tmp = alpha * np.sign(grad[not_satisfied]) / np.sum(grad[not_satisfied][kdx])

                velocity[not_satisfied] = decay_factor * velocity[not_satisfied] + tmp
                advs[not_satisfied] = advs[not_satisfied] + alpha * np.sign(velocity[not_satisfied])

                '''
                '''
                advs[not_satisfied] = np.clip(advs[not_satisfied], lower_pixel_value, upper_pixel_value)

                not_satisfied_adv_labels = np.argmax(target_classifier.predict(advs[not_satisfied]), axis=1)
                batch_adv_label[not_satisfied] = not_satisfied_adv_labels
                isDiffLabels = batch_adv_label != Y[start: end]

                iter -= 1
                sr = np.sum(isDiffLabels) / len(Y[start: end])

                if sr == 1 or iter <= 0:
                    # logger.debug(f'\tAttacking this batch done. Success rate = {sr * 100}%')

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

        utils.confirm_adv_attack(self.target_classifier, self.final_advs, self.final_origin, self.final_true_labels, self.X)

        return self.final_origin, self.final_advs, self.final_true_labels



if __name__ == '__main__':
    TARGET_CLASSIFIER_PATH = '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/data/classifier/CIFAR-10/ModelA/model'
    target_classifier = keras.models.load_model(TARGET_CLASSIFIER_PATH)

    trainingsetX = np.load(
        '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/data/dataset/CIFAR-10/50ktrainingset.npy')
    trainingsetY = np.load(
        '/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/data/dataset/CIFAR-10/50ktrainingset_labels.npy')

    '''
    Just attack on correctly predicted images
    '''
    pred = target_classifier.predict(trainingsetX)
    pred = np.argmax(pred, axis=1)
    true_indexes = np.where(pred == trainingsetY)[0][:200]
    print(f'Number of correctly predicted images = {len(true_indexes)}')

    attacker = UntargetedMI_FGSM(X=trainingsetX[true_indexes],
                                 Y=trainingsetY[true_indexes],
                                 target_classifier=target_classifier)
    final_origin, final_advs, final_true_labels = attacker.attack()
    utils.exportAttackResult(
        output_folder='/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/untargeted mi_fgsm',
        target_classifier=target_classifier, final_advs=final_advs, final_origin=final_origin,
        final_true_labels=final_true_labels)

    l = np.argmax(target_classifier.predict(final_origin), axis=1)
    print(np.sum(l == final_true_labels) / len(final_true_labels))

    l = np.argmax(target_classifier.predict(final_advs), axis=1)
    print(np.sum(l == final_true_labels) / len(final_true_labels))