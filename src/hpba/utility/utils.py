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

from utility.constants import DATA_NP_EXTENSION, ALLOWED_IMAGE_EXTENSIONS, shared_exit_msg


def get_timestamp(timestamp_format="%d%m%d-%H%M%S"):
    return datetime.now().strftime(timestamp_format)


def check_path_exists(path):
    return os.path.exists(path)


def mkdir(path):
    if not check_path_exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def clear_dir(dir_path):
    if not any(os.scandir(dir_path)):
        return
    for f in os.listdir(dir_path):
        os.remove(os.path.join(dir_path, f))


def check_file_extension(file_path: str, extension_type: str):
    return file_path.endswith(extension_type)


def get_file_name(file_path: str):
    full_file_name = os.path.basename(file_path)
    return os.path.splitext(full_file_name)[0]


# image

def get_border(image: np.ndarray) -> np.ndarray:
    min = np.min(image)
    max = np.max(image)
    img_0_255 = image.astype(np.unit8)
    if min >= 0. and max <= 1.:
        img_0_255 = (image * 255.).astype(np.unit8)
    border_img = np.array(cv2.Canny(img_0_255, 100, 200)).reshape((image.shape[0], image.shape[1], 1))
    return border_img


def get_borders(images: np.ndarray) -> np.ndarray:
    border_results = []
    for image in images:
        border_results.append(get_border(image))
    return np.array(border_results, dtype=np.float32) / 255.


def write_to_file(content, path: str, mode='w'):
    file_writer = open(file=path, mode=mode)
    file_writer.write(content)
    file_writer.close()


def read_data_from_npy(config_parser, attack_config, shared_exit_msg, logger):
    training_data_path = os.path.abspath(config_parser['DATA']['trainingDataPath'])
    if not check_path_exists(training_data_path):
        logger.error(f'not found training data path: {training_data_path}')
        exit_execution(shared_exit_msg)

    if not check_file_extension(training_data_path, DATA_NP_EXTENSION):
        logger.error(
            f'file type does not match: {training_data_path}\n Please choose the file with extension: {DATA_NP_EXTENSION}')
        exit_execution(shared_exit_msg)

    attack_config.training_path = training_data_path
    attack_config.training_data = np.load(attack_config.training_path)

    label_data_path = os.path.abspath(config_parser['DATA']['labelDataPath'])
    if not check_path_exists(label_data_path):
        logger.error(f'not found label data path: {label_data_path}')
        exit_execution(shared_exit_msg)

    if not check_file_extension(label_data_path, DATA_NP_EXTENSION):
        logger.error(
            f'file type does not match: {label_data_path}\n Please choose file with extension: {DATA_NP_EXTENSION}')
        exit_execution(shared_exit_msg)

    attack_config.label_data = label_data_path
    attack_config.label_data = np.load(label_data_path)

    if attack_config.training_data.shape[0] != attack_config.label_data.shape[0]:
        logger.error(f'training data and label data are not matched.')
        exit_execution(shared_exit_msg)

    # analyze
    if len(attack_config.training_data.shape) == 3:
        attack_config.training_data = attack_config.training_data.reshape((*attack_config.training_data.shape, 1))

    attack_config.input_size = attack_config.training_data.shape[0]
    data_example = attack_config.training_data[:1]
    attack_config.input_shape = data_example[0].shape
    attack_config.total_element_a_data = np.prod(attack_config.input_shape)
    attack_config.num_class = len(attack_config.classifier.predict(data_example)[0])

    if len(attack_config.label_data.shape) == 1 and attack_config.num_class != 1:
        attack_config.label_data = tf.keras.utils.to_categorical(attack_config.label_data, attack_config.num_class)


def read_image_from_folder(config_parser, attack_config, shared_exit_msg, logger):
    image_folder_path = config_parser['DATA']['dataFolder']
    ok, msg = validate_image_folder(attack_config=attack_config, image_folder_path=image_folder_path)
    if not ok:
        logger.error(msg)
        exit_execution(shared_exit_msg)

    sub_data_folder = os.path.join(image_folder_path, str(attack_config.original_class))
    logger.debug('Found training data autoencoder: ' + str(sub_data_folder))
    # datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255.)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=None)

    data_train = datagen.flow_from_directory(image_folder_path, target_size=attack_config.input_shape[:-1],
                                             color_mode='rgb', classes=[str(attack_config.original_class)],
                                             batch_size=200)
    logger.debug('Reading images ...')
    x = []
    y = []
    for i in range(data_train.__len__()):
        logger.debug(f'Image batch number: {i + 1}/{data_train.__len__()}')
        x.append(data_train.next()[0])
        y.append(data_train.next()[1])

    # x = np.concatenate([data_train.next()[0] for _ in range(data_train.__len__())])
    # y = np.concatenate([data_train.next()[1] for _ in range(data_train.__len__())])
    x = np.concatenate(x)
    y = np.concatenate(y)

    attack_config.training_data = x
    attack_config.input_size = len(x)
    attack_config.label_data = tf.keras.utils.to_categorical(
        np.array([attack_config.original_class] * attack_config.input_size),
        attack_config.num_class)
    logger.debug('Reading images DONE!')
    return None


def save_images_to_folder(folder_path, images, prefix_file_name='', extension=ALLOWED_IMAGE_EXTENSIONS[0],
                          logger=None):
    if not check_path_exists(folder_path):
        mkdir(folder_path)
    elif any(os.scandir(folder_path)):
        logger.debug(f'folder {folder_path} is not empty!')
    if np.max(images) < 1.1:
        images = np.round(images * 255)
    for index, image in enumerate(images):
        file_name = prefix_file_name + str(index) + '.' + extension
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, image)


def validate_image_folder(attack_config, image_folder_path: str):
    if not check_path_exists(image_folder_path):
        return False, 'Data folder does not exist! ' + image_folder_path
    for path in os.listdir(image_folder_path):
        if os.path.isdir(os.path.join(image_folder_path, path)) and path.isnumeric():

            # check if original_class folder is found
            if int(path) == attack_config.original_class:
                return True, 'ok'

    return False, 'Not found dataset in folder corresponding to origin label: ' + str(
        attack_config.original_class) + \
           '. Please set the dataset directory named as "' + str(attack_config.original_class) + '"'


# def pre_process_from_folder_wrapper(attack_config):
#     def pre_process_folder(filename):
#         img = cv2.imread(filename)


# metric

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


def custom_evaluate_classifier(pred_vector, true_pred):
    pred_vector = np.argmax(pred_vector, axis=-1)
    if len(true_pred.shape) == 2:
        true_pred = np.argmax(true_pred, axis=-1)
    if len(pred_vector.shape) == 2:
        pred_vector = np.argmax(pred_vector, axis=-1)
    return sum(true_pred == pred_vector) / len(pred_vector)


def cut_off_generated_data(origin, generated, ori_pred, gen_pred, L2_percent_thresold=0.7, L2_value_thresold=None):
    """
    replace top 30% of L2 distance from generated by origin ones
    return:
        generated_new: generated advs after cutting off
        L2_threshold: L2 value threshold to cut-off
    """
    # print(origin.shape[1:])
    if len(ori_pred.shape) == 2:
        ori_pred = np.argmax(ori_pred, axis=-1)
    if len(gen_pred.shape) == 2:
        gen_pred = np.argmax(gen_pred, axis=-1)
    l2s = compute_l2s_v2(origin, generated, np.prod(origin.shape[1:]))

    if L2_value_thresold is None:
        idx = np.argsort(l2s)
        length = len(l2s)
        cut_index = int(L2_percent_thresold * length)
        cut_off_idxs = np.array(idx[cut_index:])
        L2_value_thresold = l2s[cut_index]
    else:
        cut_off_idxs = np.where(l2s < L2_value_thresold)[0]
    diff_idxs = np.where(ori_pred != gen_pred)[0]
    final_cut_off_idxs = np.array([i for i in diff_idxs if i in cut_off_idxs])

    generated_new = np.array(generated)
    if final_cut_off_idxs is not None and len(final_cut_off_idxs) != 0:
        generated_new[final_cut_off_idxs] = np.array(origin[final_cut_off_idxs])
    return generated_new, L2_value_thresold


def wrap_range_custom_activation(min_value, max_value):
    def range_custom_activation(x):
        x_tanh = tf.keras.activations.tanh(x) + 1  # x in range(0,2)
        scale = (max_value - min_value) / 2.
        return x_tanh * scale + min_value

    return range_custom_activation


def get_range_of_input(data: np.ndarray):
    return np.min(data), np.max(data)


def check_inside_range(data, checked_range):
    """
    checked_range = (min_value, max_value)
    """
    min_value, max_value = get_range_of_input(data)
    return min_value >= checked_range[0] and max_value <= checked_range[1]


def exit_execution(msg: str = shared_exit_msg):
    sys.exit(msg)
