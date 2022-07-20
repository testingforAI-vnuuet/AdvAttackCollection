"""
Created At: 14/07/2021 14:28
"""
import os
from datetime import datetime

import cv2
import numpy as np
import sys
import tensorflow as tf
# from utility.config import *
from matplotlib import pyplot as plt


def compute_l0(adv: np.ndarray,
               ori: np.ndarray):
    # if np.max(adv) <= 1:
    #     adv = np.round(adv * 255)
    # if np.max(ori) <= 1:
    #     ori = np.round(ori * 255)
    adv = np.round(adv, decimals=2)
    ori = np.round(ori, decimals=2)

    adv = adv.reshape(-1)
    ori = ori.reshape(-1)
    l0_dist = 0
    for idx in range(len(adv)):
        if adv[idx] != ori[idx]:
            l0_dist += 1
    return l0_dist


def compute_l0s(advs: np.ndarray,
                oris: np.ndarray,
                n_features: int):
    # if np.max(advs) <= 1:
    #     advs = np.round(advs * 255)
    # if np.max(oris) <= 1:
    #     oris = np.round(oris * 255)
    advs = np.round(advs, decimals=2)
    oris = np.round(oris, decimals=2)

    advs = advs.reshape(-1, n_features)
    oris = oris.reshape(-1, n_features)
    l0_dist = np.sum(advs != oris, axis=1)
    return l0_dist


def compute_distance(data_1: np.ndarray, data_2: np.ndarray):
    if len(data_1.shape) == 4:  # (#samples, width, height, channel)
        n_features = data_1.shape[1] * data_1.shape[2]
    elif len(data_1.shape) == 3:  # (width, height, channel)
        n_features = data_1.shape[0] * data_1.shape[1]
    elif len(data_1.shape) == 2:  # (#samples, width * height)
        n_features = data_1.shape[1]

    result_l0 = compute_l0s(data_1, data_2, n_features)
    result_l2 = compute_l2s(data_1, data_2, n_features)
    result_ssim = compute_ssim(data_1, data_2)

    return np.asarray(result_l0), np.asarray(result_l2), np.asarray(result_ssim)


def compute_l2(adv: np.ndarray,
               ori: np.ndarray):
    # if np.max(adv) > 1:
    #     adv = adv / 255.
    # if np.max(ori) > 1:
    #     ori = ori / 255.
    adv = np.round(adv, decimals=2)
    ori = np.round(ori, decimals=2)
    return np.linalg.norm(adv.reshape(-1) - ori.reshape(-1))


def compute_l2s(advs: np.ndarray,
                oris: np.ndarray,
                n_features: int):
    # if np.max(advs) > 1:
    #     advs = advs / 255.
    # if np.max(oris) > 1:
    #     oris = oris / 255.
    advs = np.round(advs, decimals=2)
    oris = np.round(oris, decimals=2)
    l2_dist = np.linalg.norm(advs.reshape(-1, n_features) - oris.reshape(-1, n_features), axis=1)
    return l2_dist


def compute_ssim(advs: np.ndarray,
                 oris: np.ndarray):
    '''
    SSIM distance between a set of two images (adv, ori)
    :param advs: (size, width, height, channel). If size = 1, we have one adversarial example.
    :param oris: (size, width, height, channel). If size = 1, we have one original image.
    :return:
    '''
    # if np.max(advs) > 1:
    #     advs = advs / 255.
    # if np.max(oris) > 1:
    #     oris = oris / 255.
    advs = np.round(advs, decimals=2)
    oris = np.round(oris, decimals=2)
    advs = tf.image.convert_image_dtype(advs, tf.float32)
    oris = tf.image.convert_image_dtype(oris, tf.float32)
    ssim = tf.image.ssim(advs, oris, max_val=2.0)
    return ssim.numpy().reshape(-1)


def compute_l2_v2(adv: np.ndarray,
                  ori: np.ndarray):
    # if np.max(adv) > 1:
    #     adv = adv / 255.
    # if np.max(ori) > 1:
    #     ori = ori / 255.
    adv = np.round(adv, decimals=2)
    ori = np.round(ori, decimals=2)
    return np.linalg.norm(adv.reshape(-1) - ori.reshape(-1))


def compute_l2s_v2(advs: np.ndarray,
                   oris: np.ndarray,
                   n_features: int):
    # if np.max(advs) > 1:
    #     advs = advs / 255.
    # if np.max(oris) > 1:
    #     oris = oris / 255.
    advs = np.round(advs, decimals=2)
    oris = np.round(oris, decimals=2)
    l2_dist = np.linalg.norm(advs.reshape(-1, n_features) - oris.reshape(-1, n_features), axis=1)
    return l2_dist


def show_three_images_3D(x_28_28_left, x_28_28_mid, x_28_28_right, left_title="", mid_title="", right_title="",
                         path=None, display=False):
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 3, 1)
    fig1.title.set_text(left_title)
    plt.imshow(x_28_28_left)
    # plt.imshow(x_28_28_left)

    fig2 = fig.add_subplot(1, 3, 2)
    fig2.title.set_text(mid_title)
    plt.imshow(x_28_28_mid)
    # plt.imshow(x_28_28_right)

    fig3 = fig.add_subplot(1, 3, 3)
    fig3.title.set_text(right_title)
    plt.imshow(x_28_28_right)
    # plt.imshow(x_28_28_right)

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight', dpi=600)

    if display:
        plt.show()


def show_two_images_3D(x_28_28_left, x_28_28_right, left_title="", right_title="", path=None, display=False):
    fig = plt.figure()
    fig1 = fig.add_subplot(1, 2, 1)
    fig1.title.set_text(left_title)
    plt.imshow(x_28_28_left)
    # plt.imshow(x_28_28_left)

    fig2 = fig.add_subplot(1, 2, 2)
    fig2.title.set_text(right_title)
    plt.imshow(x_28_28_right)
    # plt.imshow(x_28_28_right)

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight', dpi=600)

    if display:
        plt.show()

def exportAttackResult(output_folder, target_classifier, final_origin, final_advs, final_true_labels):
    print('----------------------')
    print('DONE ATTACK. It is time to export the results.')

    if os.path.exists(output_folder):
        print(f'Folder {output_folder} exists. Stop attacking.')
        return
    else:
        os.makedirs(output_folder)

    print(f'\tExporting original images to \'{output_folder}/origins\'')
    np.save(f'{output_folder}/origins', final_origin)

    print(f'\tExporting adversarial images to \'{output_folder}/advs\'')
    np.save(f'{output_folder}/advs', final_advs)

    print(f'\tExporting ground-truth labels of images to \'{output_folder}/true_labels\'')
    np.save(f'{output_folder}/true_labels', final_true_labels)

    img_folder = f'{output_folder}/examples'
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    print(f'\tExporting some images to \'{img_folder}\'')
    for idx in range(0, 10):  # just plot some images for visualization
        advLabel = target_classifier.predict(final_advs[idx][np.newaxis, ...])
        advLabel = np.argmax(advLabel, axis=1)[0]

        show_two_images_3D(final_origin[idx],
                                 final_advs[idx],
                                 left_title=f'origin\n(label {final_true_labels[idx]})',
                                 right_title=f'adv\n(label {advLabel})',
                                 display=False,
                                 path=f'{img_folder}/img {idx}')


def compute_gradient_batch(inputs: tf.Tensor, target_neurons, target_classifier):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        outputs = target_classifier(inputs)
        gradient = []
        n = outputs.shape[0]
        for idx in range(n):
            gradient.append(outputs[idx, target_neurons[idx]])
    return gradient, tape
