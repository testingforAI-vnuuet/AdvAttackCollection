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
                 , X  # (batch, width, height, channel)
                 , Y  # 1D, just contain labels
                 , target_classifier
                 , epsilon=1 / 255  # a positive number, from 0 to 1
                 , batch_size=100  # a positive number, integer
                 , max_iteration=20  # a positive number, integer
                 , lower_pixel_value = 0 # the valid lower range of a pixel
                 , upper_pixel_value= 1 # the valid upper range of a pixel
                 ):
        self.X = X
        self.Y = Y
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.max_iteration = max_iteration
        self.target_classifier = target_classifier

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
        X = self.X
        ep = self.epsilon
        batch_size = self.batch_size
        max_iteration = self.max_iteration
        Y = self.Y
        target_classifier = self.target_classifier
        upper_pixel_value = self.upper_pixel_value
        lower_pixel_value = self.lower_pixel_value

        n_batchs = int(np.ceil(len(X) / batch_size))
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
                batch_adv_label = np.argmax(target_classifier.predict(advs), axis=1)
                isDiffLabels = batch_adv_label != Y[start: end]
                not_satisfied = np.where(isDiffLabels == False)[0]

                advs[not_satisfied] = advs[not_satisfied] - ep * np.sign(grad[not_satisfied])
                advs[not_satisfied] = np.clip(advs[not_satisfied], lower_pixel_value, upper_pixel_value)

                not_satisfied_adv_labels = np.argmax(target_classifier.predict(advs[not_satisfied]), axis=1)
                batch_adv_label[not_satisfied] = not_satisfied_adv_labels
                isDiffLabels = batch_adv_label != Y[start: end]

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

        return self.final_origin, self.final_advs, self.final_true_labels

    def export(self, output_folder):
        print('----------------------')
        print('DONE ATTACK. It is time to export the results.')

        if os.path.exists(output_folder):
            print(f'Folder {output_folder} exists. Stop attacking.')
            return
        else:
            os.makedirs(output_folder)

        print(f'\tExporting original images to \'{output_folder}/origins\'')
        np.save(f'{output_folder}/origins', self.final_origin)

        print(f'\tExporting adversarial images to \'{output_folder}/advs\'')
        np.save(f'{output_folder}/advs', self.final_advs)

        print(f'\tExporting ground-truth labels of images to \'{output_folder}/true_labels\'')
        np.save(f'{output_folder}/true_labels', self.final_true_labels)

        img_folder = f'{output_folder}/examples'
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        print(f'\tExporting some images to \'{img_folder}\'')
        for idx in range(0, 10):  # just plot some images for visualization
            advLabel = target_classifier.predict(self.final_advs[idx][np.newaxis, ...])
            advLabel = np.argmax(advLabel, axis=1)[0]

            utils.show_two_images_3D(self.final_origin[idx],
                                     self.final_advs[idx],
                                     left_title=f'origin\n(label {self.final_true_labels[idx]})',
                                     right_title=f'adv\n(label {advLabel})',
                                     display=False,
                                     path=f'{img_folder}/img {idx}')

if __name__ == '__main__':
    TARGET_CLASSIFIER_PATH = 'D:\Things\PyProject\AdvDefense\data\CIFAR10\cifar10_classifier_I.h5'
    target_classifier = keras.models.load_model(TARGET_CLASSIFIER_PATH)

    trainingsetX = np.load(
        'D:\Things\PyProject\AdvDefense\data\CIFAR10\cifar10_train_data.npy')
    trainingsetY = np.load(
        'D:\Things\PyProject\AdvDefense\data\CIFAR10\cifar10_sparse_train_label.npy')

    '''
    Just attack on correctly predicted images
    '''
    pred = target_classifier.predict(trainingsetX)
    pred = np.argmax(pred, axis=1)
    true_indexes = np.where(pred == trainingsetY)[0][:1000]
    print(f'Number of correctly predicted images = {len(true_indexes)}')

    attacker = UntargetedBIS(X=trainingsetX[true_indexes],
                             Y=trainingsetY[true_indexes],
                             target_classifier=target_classifier)
    final_origin, final_advs, final_true_labels = attacker.attack()
    attacker.export(output_folder='data')
