"""
Created At: 14/07/2021 14:28
"""
import os
from datetime import datetime

import cv2
import numpy as np
import sys
import tensorflow as tf

from src.utils.attack_logger import AttackLogger

logger = AttackLogger.get_logger()
from src.utils.constants import *
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

    fig2 = fig.add_subplot(1, 2, 2)
    fig2.title.set_text(right_title)
    plt.imshow(x_28_28_right)

    if path is not None:
        plt.savefig(path, pad_inches=0, bbox_inches='tight', dpi=600)

    if display:
        plt.show()
    plt.close()


def exportAttackResult(output_folder, name, target_classifier, final_origin, final_advs, final_true_labels,
                       logger=None):
    logger.debug('----------------------')
    logger.debug('DONE ATTACK. It is time to export the results.')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    origin_path = f'{output_folder}/origins'
    if not os.path.exists(origin_path):
        os.makedirs(origin_path)
    origin_file_path = f'{origin_path}/{name}_origins'
    logger.debug(f'\t\tExporting original images to \'{origin_file_path}')
    np.save(origin_file_path, final_origin)

    advs_path = f'{output_folder}/advs'
    if not os.path.exists(advs_path):
        os.makedirs(advs_path)
    advs_file_path = f'{advs_path}/{name}_advs'
    logger.debug(f'\t\tExporting adversarial images to \'{advs_file_path}')
    np.save(advs_file_path, final_advs)

    label_path = f'{output_folder}/labels'
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    label_file_path = f'{label_path}/{name}_labels'
    logger.debug(f'\t\tExporting ground-truth labels of images to \'{label_file_path}')
    np.save(label_file_path, final_true_labels)

    n_images = 10 if len(final_origin) > 10 else len(final_origin)

    images_path = f'{output_folder}/examples/{name}'
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    logger.debug(f'\t\tExporting some examples to \'{images_path}')

    for idx in range(0, n_images):  # just plot some images for visualization
        if idx >= len(final_advs):
            break
        advLabel = target_classifier.predict(final_advs[idx][np.newaxis, ...])
        advLabel = np.argmax(advLabel, axis=1)[0]

        show_two_images_3D(final_origin[idx],
                           final_advs[idx],
                           left_title=f'origin\n(label {final_true_labels[idx]})',
                           right_title=f'adv\n(label {advLabel})',
                           display=False,
                           path=f'{images_path}/img {idx}')


def compute_gradient_batch(inputs: tf.Tensor, target_neurons, target_classifier):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        outputs = target_classifier(inputs)
        gradient = []
        n = outputs.shape[0]
        for idx in range(n):
            gradient.append(outputs[idx, target_neurons[idx]])
    return gradient, tape


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


def save_images_to_folder(
        folder_path,
        images,
        prefix_file_name='',
        extension=IMAGE_EXTENSION[0],
        logger=None
):
    if not check_path_exists(folder_path):
        mkdir(folder_path)
    elif any(os.scandir(folder_path)):
        logger.debug(f'Folder {folder_path} is not empty!')

    if np.max(images) < 1.1:
        images = np.round(images * 255)

    for index, image in enumerate(images):
        file_name = prefix_file_name + str(index) + '.' + extension
        file_path = os.path.join(folder_path, file_name)
        cv2.imwrite(file_path, image)


def validate_input_value(data, value_range):
    min_value = np.min(data)
    max_value = np.max(data)
    return min_value >= value_range[0] and max_value <= value_range[1]


def load_model(model_path, logger=None):
    """
    Read model using multiple model-reading methods
    Parameters:
        model_path: path to the model
        logger: logger
    Returns:
        model: tensorflow model
        model_name: name of model
        success: if classifier_path is valid
        message: returned message
    """
    if not check_path_exists(model_path):
        return None, None, False, 'Path not found: ' + str(model_path)

    # list of functions to read classifier model by supported formats
    read_model_functions_list = [load_model_from_h5, load_model_from_folder]
    success = False
    model = None
    model_name = None

    for read_model_function in read_model_functions_list:
        model, model_name, success, message = read_model_function(model_path)
        logger.debug(message) if logger is not None else None
        if success is True:
            break

    if success is True:
        message = 'Successfully loaded model: '
    else:
        message = 'Failed to load model from: '
    message += str(model_path)

    return model, model_name, success, message


def load_model_from_h5(model_file_path: str):
    if not check_file_extension(model_file_path, TF_MODEL_H5_EXTENSION):
        return None, None, False, f'Model format does not match to {TF_MODEL_H5_EXTENSION}'
    try:
        model = tf.keras.models.load_model(model_file_path, compile=False)
        model_name = get_file_name(model_file_path)
    except (ImportError, IOError) as e:
        return None, None, False, e

    return model, model_name, True, f'Loaded model with format {TF_MODEL_H5_EXTENSION}'


def load_model_from_folder(model_folder_path: str):
    """
    read model from folder. Learn more: https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading
    :param model_folder_path: path to the model folder
    :return:
        model: tensorflow model
        model_name: name of model
        success: if classifier_path is valid
        message: returned message
    """
    try:
        model = tf.keras.models.load_model(model_folder_path, compile=False)
        model_name = get_file_name(model_folder_path)
    except (ImportError, IOError) as e:
        return None, None, False, e

    return model, model_name, True, 'Loaded model saved in folder as tf2.x version'


def load_data_from_npy(data_file_path, logger):
    """
    read data from numpy binary file
    :param
        data_file_path: file path to the data file
        logger: logger
    :return:
        data: returned loaded data
    """
    if not check_path_exists(data_file_path):
        logger.error(f'Not found data path: {data_file_path}')
        sys.exit()

    if not check_file_extension(data_file_path, NP_BINARY_EXTENSION):
        logger.error(
            f'File type does not match: {data_file_path}'
            f'Please choose the file with extension: {NP_BINARY_EXTENSION}'
        )
        sys.exit()

    try:
        data = np.load(data_file_path)
    except OSError as e:
        logger.error(f'Failed to load data from {data_file_path}')
        sys.exit()

    logger.info(f'Successfully loaded data from {data_file_path}')
    logger.info(f'Data shape {data.shape}')
    return data


def load_image_from_folder(images_folder_path, image_shape, logger):
    """
    read data from folder
    :param:
        images_folder_path: folder path to the data folder
        image_shape: shape of input images
        logger: logger
    :return:
        data: returned loaded data from data folder
        label: label of data referring to data folder
    """
    valid, message = validate_images_folder(images_folder_path)
    if not valid:
        logger.error(message)
        sys.exit()
    logger.info(message)

    data = []
    label = []
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=None)

    data_train = datagen.flow_from_directory(
        images_folder_path, color_mode='rgb',
        batch_size=200,
        class_mode='categorical',
        target_size=image_shape,
    )

    for i in range(data_train.__len__()):
        logger.debug('Reading_image {:d}%'.format(int((i + 1) / data_train.__len__() * 100)))
        data.append(data_train.next()[0])
        label.append(data_train.next()[1])

    data = np.concatenate(data)
    label = np.concatenate(label)

    logger.info('Reading images DONE!')
    logger.info(f'Data size {data.shape[0]}')
    logger.info(f'Image shape {data.shape[1:]}')
    logger.info(f'Label shape {label.shape[1:]}')

    return data, label


def validate_images_folder(images_folder_path: str):
    if not check_path_exists(images_folder_path):
        return False, f'Data folder does not exist in: {images_folder_path}'

    for path in os.listdir(images_folder_path):
        if not path.isnumeric():
            return False, f'Sub data folder is not valid: {path}'

    return True, f'Data folder is valid: {images_folder_path}'

def confirm_adv_attack(target_classifier,
                       final_advs,
                       final_origin,
                       final_true_labels,
                       X # correctly predicted images
                       ):
    '''
    Confirm again
    '''
    try:
        logger.debug('Confirming the result:')
        if final_advs is None or len(final_advs) == 0:
            logger.debug("\t\tThere is no generated adversarial examples. Overall success rate = 0%")
            return

        pred_adv = target_classifier.predict(final_advs)
        pred_adv = np.argmax(pred_adv, axis=1)

        pred_ori = target_classifier.predict(final_origin)
        pred_ori = np.argmax(pred_ori, axis=1)
        logger.debug(f'\t\t#Origins = {len(X)}')

        tmp = np.sum(pred_ori != final_true_labels)
        logger.debug(f'\t\t#Invalid true labels = {tmp}')

        tmp = np.sum(pred_adv == pred_ori)
        logger.debug(f'\t\t#Invalid adversarial examples = {tmp}')

        tmp = np.sum(pred_adv != pred_ori)
        logger.debug(f'\t\t#Valid adversarial examples = {tmp}')

        logger.debug(f'The overall success rate = {np.round(len(pred_adv) / len(X) * 100, 2)}%.')

        l0s = compute_l0s(final_advs, final_origin, n_features=final_advs.shape[1] * final_advs.shape[2] * final_advs.shape[3])
        min = np.min(l0s)
        max = np.max(l0s)
        avg = np.round(np.average(l0s), 2)
        logger.debug(f'L0 min/max/avg = {min} / {max} / {avg: 0.2f} (close to 0 is better)')

        l2s = compute_l2s(final_advs, final_origin, n_features=final_advs.shape[1] * final_advs.shape[2] * final_advs.shape[3])
        min = np.round(np.min(l2s), 2)
        max = np.round(np.max(l2s), 2)
        avg = np.round(np.average(l2s), 2)
        logger.debug(f'L2 min/max/avg = {min: 0.2f} / {max: 0.2f} / {avg: 0.2f} (close to 0 is better)')

        ssims = compute_ssim(final_advs, final_origin)
        min = np.round(np.min(ssims), 2)
        max = np.round(np.max(ssims), 2)
        avg = np.round(np.average(ssims), 2)
        logger.debug(f'SSIM min/max/avg = {min: 0.2f} / {max: 0.2f} / {avg: 0.2f} (larger value is better)')
    except Exception:
        return