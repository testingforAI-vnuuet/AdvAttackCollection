'''
Untargeted attack
'''
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow

from src.utils import utils


class UntargetedBIS:
    def __init__(self
                 , X  # (batch, width, height, channel), pixel in [0..1]
                 , Y  # 1D, just contain labels
                 , target_classifier
                 , output_folder=None
                 , epsilon=1 / 255  # a positive number, from 0 to 1
                 , batch_size=100  # a positive number, integer
                 , max_iteration=1  # a positive number, integer
                 , lower_pixel_value = 0 # the valid lower raneg of a pixel
                 , upper_pixel_value= 1 # the valid upper raneg of a pixel
                 ):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.target_classifier = target_classifier
        self.output_folder = output_folder
        self.lower_pixel_value = lower_pixel_value
        self.upper_pixel_value = upper_pixel_value

        self.final_advs = None
        self.final_true_labels = None
        self.final_origin = None

    def compute_gradient_batch(self, inputs: tf.Tensor, target_neuron, target_classifier):
        # print(f'compute_gradient_batch - begin')
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            outputs = target_classifier(inputs)
            gradient = []
            # print(f'compute_gradient_batch - middle')
            for idx in range(outputs.shape[0]):
                # print(idx)
                gradient.append(outputs[idx, target_neuron[idx]])
        # print(f'compute_gradient_batch - end')
        return gradient, tape

    def attack(self):
        '''
        Attack
        '''
        if os.path.exists(self.output_folder):
            print(f'Folder {self.output_folder} exists. Stop attacking.')
            return None, None, None

        X = self.X
        ep = self.epsilon
        batch_size = self.batch_size
        max_iteration = self.max_iteration
        Y = self.Y
        target_classifier = self.target_classifier
        upper_pixel_value = self.upper_pixel_value
        lower_pixel_value = self.lower_pixel_value

        n_batchs = int(np.ceil(len(X) / batch_size))
        adv_label = None
        for idx in range(n_batchs):
            '''
            Computing batch range
            '''
            start = idx * batch_size
            end = start + batch_size
            if end > len(X):
                end = len(X)
            print(f'[{idx} / {n_batchs}] Attacking from {start} to {end}')

            '''
            Attack
            '''
            iter = max_iteration
            advs = X[start: end].copy()
            tensor_advs = tensorflow.convert_to_tensor(advs)

            while iter >= 0:
                print(f'\tIteration {iter}')
                # compute gradient
                gradient, tape = self.compute_gradient_batch(inputs=tensor_advs,
                                                             target_neuron=Y[start: end],
                                                             target_classifier=target_classifier)
                grad = tape.gradient(gradient, tensor_advs)
                grad = np.asarray(grad)

                # find not-satisfied indexes
                adv_label = np.argmax(target_classifier.predict(advs), axis=1)
                isDiffLabels = adv_label != Y[start: end]
                not_satisfied = np.where(isDiffLabels == False)[0]

                advs[not_satisfied] = advs[not_satisfied] - ep * np.sign(grad[not_satisfied])
                advs[not_satisfied] = np.clip(advs[not_satisfied], lower_pixel_value, upper_pixel_value)

                not_satisfied_adv_labels = np.argmax(target_classifier.predict(advs[not_satisfied]), axis=1)
                adv_label[not_satisfied] = not_satisfied_adv_labels
                isDiffLabels = adv_label != Y[start: end]

                iter -= 1
                sr = np.sum(isDiffLabels) / len(Y[start: end])

                if sr == 1 or iter < 0:
                    print(f'\tAttacking this batch done. Success rate = {sr * 100}%')

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

                    # print(f'\toutput shape = {self.final_advs.shape}; {self.final_origin.shape}')
                    break
                else:
                    tensor_advs = tensorflow.convert_to_tensor(advs)

        print('----------------------')
        print('DONE ATTACK. It is time to export the results.')
        if self.output_folder is not None:
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)

            print(f'\tExporting original images to \'{self.output_folder}/origins\'')
            np.save(f'{self.output_folder}/origins', self.final_origin)

            print(f'\tExporting adversarial images to \'{self.output_folder}/advs\'')
            np.save(f'{self.output_folder}/advs', self.final_advs)

            print(f'\tExporting ground-truth labels of images to \'{self.output_folder}/true_labels\'')
            np.save(f'{self.output_folder}/true_labels', self.final_true_labels)

            img_folder = f'{self.output_folder}/examples'
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

            print(f'\tExporting some images to \'{img_folder}\'')
            for idx in range(0, 10):  # just plot some images for visualization
                if idx >= len(self.final_origin):
                    break
                utils.show_two_images_3D(self.final_origin[idx],
                                         self.final_advs[idx],
                                         left_title=f'origin\n(label {self.final_true_labels[idx]})',
                                         right_title=f'adv\n(label {adv_label[satisfied][idx]})',
                                         display=False,
                                         path=f'{img_folder}/img {idx}')
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
    true_indexes = np.where(pred == trainingsetY)[0][:1]
    print(f'Number of correctly predicted images = {len(true_indexes)}')

    attacker = UntargetedBIS(X=trainingsetX[true_indexes],
                             Y=trainingsetY[true_indexes],
                             target_classifier=target_classifier,
                             output_folder='/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/untargeted bis')
    attacker.attack()
