'''
Untargeted attack
'''
import numpy as np
import tensorflow as tf
from tensorflow import keras

import tensorflow

from data.classifier.CIFAR10_ModelA import load_CIFAR10_dataset, prep_pixels
from src.utils import utils

LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def compute_gradient_batch(inputs: tf.Tensor, target_neuron: int, classifier):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        predictions_at_target_neuron = classifier(inputs)[:, target_neuron]
    gradient = tape.gradient(predictions_at_target_neuron, inputs)
    return gradient.numpy()


if __name__ == '__main__':
    trainX, trainY, testX, testY, testY_label, trainY_label = load_CIFAR10_dataset()
    trainX, testX = prep_pixels(trainX, testX)  # pixel value in range of [0,1]
    model = keras.models.load_model(
        '../../data/classifier/CIFAR10_ModelA')  # Accuracy on test set = 0.6612, Accuracy on training set = 0.8404
    model.summary()

    '''
    Prediction
    '''
    pred = model.predict(testX)
    pred = np.argmax(pred, axis=1)
    print(f'Accuracy on test set = {np.sum(pred == testY_label) / len(testX)}')

    pred = model.predict(trainX)
    pred = np.argmax(pred, axis=1)

    print(f'Accuracy on training set = {np.sum(pred == trainY_label) / len(trainX)}')
    ori_indexes = np.asarray(np.where(pred == trainY_label)).reshape(-1)

    '''
    Attack
    '''
    target_label = 1  # all modified images are classified as a specific label
    for i in range(0, len(ori_indexes)):
        index = ori_indexes[i]
        ori_label = trainY_label[index]
        if LABELS[ori_label] == target_label:
            # no need to attack
            continue

        found = False
        ep = 1 / 255
        ori = np.copy(np.expand_dims(trainX[index], axis=0))
        adv = ori

        max_iter = 10
        while True and max_iter >= 0:
            print("Modifying")
            adv = tensorflow.convert_to_tensor(adv)
            grad = compute_gradient_batch(inputs=adv,
                                          target_neuron=target_label,
                                          classifier=model)
            adv = adv + ep * np.sign(grad)
            adv = np.clip(adv, 0, 1)
            adv_label = np.argmax(model.predict(adv), axis=1)[0]
            l2 = utils.compute_l2(adv[0], ori[0])
            print(f"l2 = {l2}")
            if adv_label == target_label:
                found = True
                break
            max_iter -= 1

        if found:
            utils.show_two_images_3D(x_28_28_left=np.squeeze(ori),
                                     x_28_28_right=np.squeeze(adv),
                                     left_title=f"ori label = {ori_label} ({LABELS[ori_label]})",
                                     right_title=f"adv label = {adv_label} ({LABELS[adv_label]})",
                                     display=True)
        else:
            print("Adv not found")
