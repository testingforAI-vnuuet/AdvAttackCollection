import numpy as np
from tensorflow import keras

from src.utils import utils
from src.utils.attack_logger import AttackLogger

logger = AttackLogger.get_logger()

class UntargetedGaussian:
    def __init__(self
                 , X  # (batch, width, height, channel), must be recognized correctly by the target classifier
                 , Y  # 1D, just contain labels
                 , target_classifier
                 , epsilon=5 / 255
                 , batch_size=1000  # a positive number, integer
                 , max_iteration=20  # a positive number, integer
                 , lower_pixel_value=0  # the valid lower range of a pixel
                 , upper_pixel_value=1  # the valid upper range of a pixel
                 ):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.target_classifier = target_classifier
        self.epsilon = epsilon
        self.lower_pixel_value = lower_pixel_value
        self.upper_pixel_value = upper_pixel_value

        self.final_advs = None
        self.final_true_labels = None
        self.final_origin = None

    def attack(self):
        X = self.X
        batch_size = self.batch_size
        max_iteration = self.max_iteration
        Y = self.Y
        target_classifier = self.target_classifier
        upper_pixel_value = self.upper_pixel_value
        lower_pixel_value = self.lower_pixel_value
        epsilon = self.epsilon

        n_batchs = int(np.ceil(len(X) / batch_size))

        for idx in range(n_batchs):
            '''
            Computing batch range
            '''
            start = idx * batch_size
            end = start + batch_size
            if end > len(X):
                end = len(X)
            logger.debug(f'[{idx} / {n_batchs}] Attacking from {start} to {end}')

            '''
            Attack
            '''
            iter = max_iteration
            advs = X[start: end].copy()
            not_satisfied = np.arange(start, end)
            adv_labels = Y[start:end].copy()
            while iter >= 0:
                logger.debug(f'\tIteration {iter}')
                noise = epsilon * np.random.normal(size=(len(not_satisfied), X.shape[1], X.shape[2], X.shape[3]))
                advs[not_satisfied] = np.clip(advs[not_satisfied] + noise, lower_pixel_value, upper_pixel_value)

                pred = target_classifier.predict(advs[not_satisfied])
                pred = np.argmax(pred, axis=1)
                adv_labels[not_satisfied] = pred
                not_satisfied = np.where(adv_labels == Y[start:end])[0]
                satisfied = np.where(adv_labels != Y[start:end])[0]
                logger.debug(f'\t\t Success rate of this batch = {np.round(len(satisfied) / len(adv_labels) * 100, 2)}%')

                if len(not_satisfied) == 0:
                    break
                else:
                    iter -= 1

            '''
            Save
            '''
            if self.final_advs is None:
                self.final_advs = advs[satisfied]
                self.final_origin = X[start:end][satisfied]
                self.final_true_labels = Y[start:end][satisfied]
            else:
                self.final_advs = np.concatenate((self.final_advs, advs[satisfied]), axis=0)
                self.final_origin = np.concatenate((self.final_origin, X[start:end][satisfied]), axis=0)
                self.final_true_labels = np.concatenate((self.final_true_labels, Y[start:end][satisfied]),
                                                        axis=0)
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
    true_indexes = np.where(pred == trainingsetY)[0][:1000]
    print(f'Number of correctly predicted images = {len(true_indexes)}')

    attacker = UntargetedGaussian(X=trainingsetX[true_indexes],
                                  Y=trainingsetY[true_indexes],
                                  target_classifier=target_classifier)
    final_origin, final_advs, final_true_labels = attacker.attack()
    utils.exportAttackResult(
        output_folder='/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/untargeted gaussian',
        target_classifier=target_classifier, final_advs=final_advs, final_origin=final_origin,
        final_true_labels=final_true_labels)
