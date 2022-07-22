"""
Created At: 29/07/2021 22:32
"""
from itertools import accumulate

import numpy as np
import tensorflow as tf

from src.hpba_v2_2_3.src.attacker.constants import *
from src.hpba_v2_2_3.src.utility.config import attack_config
from src.hpba_v2_2_3.src.utility.feature_ranker import feature_ranker
# progbar = tf.keras.utils.Progbar(len(train_data))
from src.hpba_v2_2_3.src.utility.utils import compute_l0, compute_distance


def optimize_advs(classifier, generated_advs, origin_images, target_label, origin_label, origin_labels, step,
                  return_recover_speed=False, num_class=10, ranking_type=None, epoch_to_optimize=10, batch_size=None,
                  is_untargeted=False):
    input_shape = generated_advs[0].shape
    total_element = np.prod(input_shape)

    advs_len = len(generated_advs)
    generated_advs = generated_advs.reshape((-1, total_element))
    origin_images = origin_images.reshape((-1, total_element))
    # origin_images_0_255 = np.round(origin_images * 255)
    # smooth_advs_0_255s = np.round(generated_advs * 255).reshape((-1, total_element))
    origin_images_0_255 = np.array(origin_images)
    smooth_advs_0_255s = np.array(generated_advs).reshape(-1, total_element)

    new_smooth_advs = []
    if batch_size is None:
        batch_size = len(generated_advs)
    latest_recover_speed = None

    if epoch_to_optimize > np.ceil(np.log2(step)):
        epoch_to_optimize = int(np.ceil(np.log2(step)))

    for epoch in range(epoch_to_optimize):
        print(
            f'[{epoch + 1}/{epoch_to_optimize}] Optimizing {len(generated_advs)} adversarial examples with step = {step}')
        if len(new_smooth_advs) != 0:
            smooth_advs_0_255s = np.array(new_smooth_advs)
        new_smooth_advs = np.array([])
        progbar = tf.keras.utils.Progbar(advs_len)
        recover_speed = None
        for batch_index in range(0, advs_len, batch_size):
            # print(f'batch: {batch_index}')
            result = optimize_batch(classifier,
                                    smooth_advs_0_255s[
                                    batch_index: batch_index + batch_size],
                                    origin_images[batch_index: batch_index + batch_size],
                                    smooth_advs_0_255s[batch_index: batch_index + batch_size],
                                    origin_images_0_255[
                                    batch_index: batch_index + batch_size],
                                    target_label,
                                    origin_label,
                                    origin_labels,
                                    step, num_class, ranking_type, return_recover_speed, is_untargeted)
            if return_recover_speed is False:
                smooth_advs_0_255 = result
            else:
                smooth_advs_0_255, recover_speed = result
            if len(new_smooth_advs) == 0:
                new_smooth_advs = smooth_advs_0_255
            else:
                new_smooth_advs = np.concatenate((new_smooth_advs, smooth_advs_0_255))
            progbar.update(batch_index + batch_size if batch_index + batch_size < advs_len else advs_len)

        # display
        L0, L2, SSIM = compute_distance(smooth_advs_0_255s.reshape(-1, input_shape[0], input_shape[1], input_shape[2]),
                                        origin_images.reshape(-1, input_shape[0], input_shape[1], input_shape[2]))
        print(f'\t| L0 min/max/avg: {np.min(L0)}/ {np.max(L0)}/ {np.average(L0):.2f}')
        print(
            f'\t| L2 min/max/avg: {np.round(np.min(L2), 2):.2f}/ {np.round(np.max(L2), 2):.2f}/ {np.round(np.average(L2), 2):.2f}')
        print(
            f'\t| SSIM min/max/avg: {np.round(np.min(SSIM), 2):.2f}/ {np.round(np.max(SSIM), 2):.2f}/ {np.round(np.average(SSIM), 2):.2f}')

        if return_recover_speed is True:
            if latest_recover_speed is None:
                latest_recover_speed = list(recover_speed)
            else:
                for index in range(len(recover_speed)):
                    for index_j in range(len(recover_speed[index])):
                        latest_recover_speed[index].append(recover_speed[index][index_j])
        step = step // 2
        if step == 0:
            break
    if return_recover_speed is True:
        latest_recover_speed = list(map(lambda x: list(accumulate(x)), latest_recover_speed))
        for index in range(len(latest_recover_speed)):
            latest_recover_speed[index] = latest_recover_speed[index] + [latest_recover_speed[index][-1]] * (
                    total_element - len(latest_recover_speed[index]))
        latest_recover_speed = np.array(latest_recover_speed, dtype=float)
        for index in range(len(latest_recover_speed)):
            latest_recover_speed[index] = latest_recover_speed[index] / float(
                compute_l0(generated_advs[index], origin_images[index]))
        average_recover_speed = np.average(latest_recover_speed, axis=0)
        latest_values = average_recover_speed[-1]
        if len(average_recover_speed) < total_element:
            average_recover_speed = np.concatenate(
                (average_recover_speed, [latest_values] * (total_element - len(average_recover_speed))), axis=0)
        return smooth_advs_0_255s.reshape(-1, *input_shape), average_recover_speed
    return smooth_advs_0_255s.reshape(-1, *input_shape)


def optimize_batch(classifier, generated_advs, origin_images, generated_advs_0_255, origin_images_0_255,
                   target_label, origin_label, origin_labels,
                   step, num_class=10, ranking_type=None, return_recover_speed=False, is_untargeted=False):
    diff_pixels = []
    batch_size = len(generated_advs)
    # smooth_advs_0_255_arrs = np.array(generated_advs_0_255)
    # diff_pixel_arrs = None
    for index in range(batch_size):
        compare = generated_advs_0_255[index] == origin_images_0_255[index]
        diff_pixels.append(np.where(compare == False)[0])
    target_label_tmp = target_label if num_class > 1 else 0
    if ranking_type == JSMA_RANKING:
        diff_pixel_arrs, diff_value_arrs = feature_ranker.jsma_ranking_batch(generated_advs=generated_advs,
                                                                             origin_images=origin_images,
                                                                             target_label=target_label_tmp,
                                                                             classifier=classifier,
                                                                             diff_pixels=diff_pixels,
                                                                             num_class=num_class)
    elif ranking_type == COI_RANKING:
        diff_pixel_arrs, diff_value_arrs = feature_ranker.coi_ranking_batch(generated_advs=generated_advs,
                                                                            origin_images=origin_images,
                                                                            target_label=target_label_tmp,
                                                                            classifier=classifier,
                                                                            diff_pixels=diff_pixels,
                                                                            num_class=num_class)
    elif ranking_type == RANDOM_RANKING:
        diff_pixel_arrs, diff_value_arrs = feature_ranker.random_ranking_batch(generated_advs=generated_advs,
                                                                               origin_images=origin_images,
                                                                               target_label=target_label_tmp,
                                                                               classifier=classifier,
                                                                               diff_pixels=diff_pixels,
                                                                               num_class=num_class)
    elif ranking_type == SEQUENTIAL_RANKING:
        diff_pixel_arrs, diff_value_arrs = feature_ranker.sequence_ranking_batch(generated_advs=generated_advs,
                                                                                 origin_images=origin_images,
                                                                                 target_label=target_label_tmp,
                                                                                 classifier=classifier,
                                                                                 diff_pixels=diff_pixels,
                                                                                 num_class=num_class)

    else:
        raise Exception('not found ranking type')

    return recover_batch(classifier=classifier, generated_advs=generated_advs,
                         origin_labels=origin_labels,
                         generated_advs_0_255=generated_advs_0_255,
                         origin_images_0_255=origin_images_0_255, target_label=target_label,
                         origin_label=origin_label,
                         step=step, diff_pixel_arrs=diff_pixel_arrs,
                         return_recover_speed=return_recover_speed, is_untargeted=is_untargeted, num_class=num_class
                         )


def recover_batch(classifier, generated_advs, origin_labels, generated_advs_0_255, origin_images_0_255, target_label,
                  origin_label,
                  step, diff_pixel_arrs=None, return_recover_speed=False, is_untargeted=False, num_class=10):
    total_elems = np.prod(generated_advs[0].shape)
    # make all diff_pixel arr to same length
    padded_diff_pixels = padd_to_arrs(diff_pixel_arrs, padded_value=total_elems, d_type=int)

    # adding to new pixel for each image
    padded_generated_advs_0_255 = padd_to_arrs(generated_advs_0_255, max_length=total_elems + 1)
    padded_origin_images_0_255 = padd_to_arrs(origin_images_0_255, max_length=total_elems + 1)
    old_padded_generated_advs_0_255 = np.array(padded_generated_advs_0_255)
    recover_speed = [[] for _ in range(len(padded_diff_pixels))]
    max_diff_lenth = max(map(len, diff_pixel_arrs))
    for step_i in range(0, max_diff_lenth, step):
        for index in range(len(generated_advs)):
            indexes = padded_diff_pixels[index][step_i:step_i + step].astype(int)

            padded_generated_advs_0_255[index, indexes] = \
                padded_origin_images_0_255[index, indexes]
            recover_speed[index].append(step)

        predictions = classifier.predict(
            padded_generated_advs_0_255[:, :-1].reshape(-1, *attack_config.input_shape))

        predictions = np.argmax(predictions, axis=1) if num_class > 1 else np.round(predictions)
        if is_untargeted is False:
            unrecovered_adv_indexes = np.where(predictions != target_label)[0]
        else:
            unrecovered_adv_indexes = np.where(predictions == np.argmax(origin_labels, axis=1))[0]
        # check recover effect to each prediction
        if len(unrecovered_adv_indexes) == 0:
            continue
        for prediction_index in unrecovered_adv_indexes:
            indexes = padded_diff_pixels[prediction_index][step_i:step_i + step].astype(int)
            padded_generated_advs_0_255[prediction_index, indexes] = \
                old_padded_generated_advs_0_255[prediction_index, indexes]
            recover_speed[prediction_index][-1] = 0
    if return_recover_speed is False:
        return padded_generated_advs_0_255[:, :-1]
    for index in range(len(diff_pixel_arrs)):
        if len(diff_pixel_arrs[index]) // step == len(recover_speed[index]):
            l = recover_speed[-1]
            recover_speed[index].append(l)
        latest_value = recover_speed[index][len(diff_pixel_arrs[index]) // step]
        latest_value = latest_value if latest_value == 0 else len(diff_pixel_arrs[index]) - step * (
                len(diff_pixel_arrs[index]) // step)
        recover_speed[index][len(diff_pixel_arrs[index]) // step] = latest_value
        recover_speed[index] = recover_speed[index][0:len(diff_pixel_arrs[index]) // step + 1]

    # ignore the new pixel padded to image
    return padded_generated_advs_0_255[:, :-1], recover_speed


def padd_to_arrs(arrs, padded_value=784, max_length=None, d_type=float):
    max_length = max(map(len, arrs)) if max_length is None else max_length
    result = []
    for arr in arrs:
        padded_arr = np.concatenate((arr, [padded_value] * (max_length - len(arr))))
        result.append(padded_arr)
    return np.asarray(result).astype(d_type)
