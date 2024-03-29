"""
Created by khadm on 1/4/2022
Feature: 
"""
import os.path
import time

import numpy as np
import tensorflow as tf

from src.hpba_v2_2_3.src.attacker.constants import *
from src.hpba_v2_2_3.src.classifier.substitute_classifier import SubstituteClassifier
from src.hpba_v2_2_3.src.utility.statistics import filter_by_label
from src.hpba_v2_2_3.src.utility.utils import mkdirs, get_timestamp
from src.utils.attack_logger import AttackLogger

logger = AttackLogger.get_logger()


class Attacker:

    def __init__(self, trainX, trainY,
                 target_classifier,
                 substitute_classifier: tf.keras.models.Model,
                 substitute_classifier_name: str,
                 origin_label: int,
                 target_label: int = None,
                 # target_position: int = 2,
                 num_class=MNIST_NUM_CLASSES,
                 method_name=ABSTRACT_METHOD_NAME,
                 attack_type=ATTACK_WHITEBOX_TYPE,
                 quality_loss_str=LOSS_MSE,
                 attack_stop_condition=None,
                 outputFolder=None):
        self.start_time = time.time()
        self.end_time = None
        self.origin_label = origin_label
        self.trainX = trainX
        self.trainY = trainY.astype('float32')
        self.classifier = target_classifier
        self.classifier_name = self.classifier.name
        self.isuntargeted = True if target_label == -1 or target_label is None else False
        self.target_label = self.origin_label if self.is_untargeted() else target_label
        # self.target_position = target_position
        self.method_name = method_name
        self.num_class = num_class
        self.attack_type = attack_type
        self.quality_loss_str = quality_loss_str

        self.attack_stop_condition = attack_stop_condition

        self.substitute_classifier = substitute_classifier
        # self.outputFolder = outputFolder

        # if self.attack_type == ATTACK_BLACKBOX_TYPE:
        #     # blackbox
        #     self.origin_images = self.trainX
        #     self.origin_labels = self.trainY
        # else:
        #     # whitebox
        #     self.origin_images, self.origin_labels = filter_by_label(label=self.origin_label, data_set=self.trainX,
        #                                                              label_set=self.trainY)

        if self.origin_label != -1:
            self.origin_images, self.origin_labels = filter_by_label(label=self.origin_label, data_set=self.trainX,
                                                                     label_set=self.trainY)
        else:
            self.origin_images = self.trainX
            self.origin_label = -1
            self.origin_labels = self.trainY

            # if self.target_label is None:
        #     if self.target_position is not None:
        #         self.target_label = label_ranking(self.origin_images, self.classifier)[-1 * self.target_position]

        self.shared_time_stamp = get_timestamp()
        self.target_vector = None

        self.target_vector = tf.keras.utils.to_categorical(self.target_label, self.num_class,
                                                           dtype='float32') if num_class > 1 else np.array(self.target_label, dtype='float32')

        # if self.target_label != -1:
        #     self.target_vector = tf.keras.utils.to_categorical(self.target_label, self.num_class,
        #                                                    dtype='float32') if num_class > 1 else np.array( self.target_label, dtype='float32')
        # else:
        #     self.target_vector = tf.keras.utils.to_categorical(self.origin_labels, self.num_class)

        self.origin_label_vector = tf.keras.utils.to_categorical(self.origin_label, self.num_class)
        self.general_result_folder = outputFolder  # os.path.abspath(os.path.join(RESULT_FOLDER_PATH, self.method_name))
        self.result_summary_folder = os.path.join(self.general_result_folder, TEXT_RESULT_SUMMARY)
        self.data_folder = os.path.join(self.general_result_folder, TEXT_DATA)
        self.image_folder = os.path.join(self.general_result_folder, TEXT_IMAGE)
        mkdirs([self.general_result_folder, self.result_summary_folder,
                self.data_folder, self.image_folder])

        self.short_file_shared_name = f'{self.method_name}_{self.classifier_name}'
        self.shared_log = f'[{self.method_name}] '
        self.adv_result = None
        self.adv_result_path = os.path.join(self.data_folder,
                                            self.short_file_shared_name + '_adv_' + self.shared_time_stamp + '.npy')
        self.origin_adv_result = None
        self.origin_adv_result_path = os.path.join(self.data_folder,
                                                   self.short_file_shared_name + '_origin_' + self.shared_time_stamp + '.npy')
        self.summary_path = os.path.join(self.result_summary_folder,
                                         self.short_file_shared_name + '_summary_' + self.shared_time_stamp + '.txt')

        self.image_path = os.path.join(self.image_folder,
                                       self.short_file_shared_name + '_random_example_' + self.shared_time_stamp + '.png')

        self.substitute_cls_name = substitute_classifier_name

        if self.attack_type == ATTACK_BLACKBOX_TYPE and self.substitute_classifier is None:
            logger.debug(f'[{ATTACK_BLACKBOX_TYPE}] Not found Substitute Classifier!')
            logger.debug(f'[{ATTACK_BLACKBOX_TYPE}] Creating and training Substitute Classifier')
            self.substitute_cls_folder = os.path.join(self.general_result_folder, 'substitute_classifier')
            mkdirs([self.substitute_cls_folder])
            name = 'substitute_' + self.short_file_shared_name

            self.substitute_cls_path = os.path.join(self.substitute_cls_folder, name + 'h5')
            self.substitute_cls_name = 'black_box'

            sub_classifier = SubstituteClassifier(trainX=self.trainX, trainY=self.trainY,
                                                  target_classifier=self.classifier, num_classes=self.num_class,
                                                  substitute_cls_path=self.substitute_cls_path,
                                                  input_shape=self.trainX[0].shape)
            sub_classifier.train()
            self.substitute_classifier = sub_classifier.classifier
            logger.debug(f'[{ATTACK_BLACKBOX_TYPE}] Training Substitute Classifier. DONE!')

    def is_untargeted(self):
        return self.isuntargeted

    def attack(self):
        pass

    def export_results(self):
        pass
