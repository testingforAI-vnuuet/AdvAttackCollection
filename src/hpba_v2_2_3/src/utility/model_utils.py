import os

from src.hpba_v2_2_3.src._test.run_testcase import test_loop
from src.hpba_v2_2_3.src.utility.constants import *
from src.hpba_v2_2_3.src.attacker.constants import *
from src.hpba_v2_2_3.src.utility.utils import check_path_exists, get_file_name, exit_execution, check_file_extension
import tensorflow as tf
from src.hpba_v2_2_3.custom_objects import user_custom_objects
from src.hpba_v2_2_3.src.attacker.losses import custom_losses
import numpy as np
from src.hpba_v2_2_3.src.classifier.black_box_classifier import BlackBoxClassifier

custom_objects = {**user_custom_objects, **custom_losses}


def analyze_classifier(config_parser, attack_config, logger, shared_exit_msg):
    classifier_path = os.path.abspath(config_parser[CONFIG_TXT.CLASSIFIER][CONFIG_TXT.targetClassifierPath])
    if attack_config.attack_type == ATTACK_BLACKBOX_TYPE:
        path = config_parser[CONFIG_TXT.CLASSIFIER][
            CONFIG_TXT.substituteClassifierPath]
        if path != '':
            substitute_classifier_path = os.path.abspath(path)
        else:
            substitute_classifier_path = None
            attack_config.substitute_classifier_name = 'default_by_hpba'
    else:
        substitute_classifier_path = None

    if not check_path_exists(classifier_path):
        logger.error(f'not found classifier_path: {classifier_path}')
        exit_execution(shared_exit_msg)

    attack_config.classifier_path = classifier_path
    attack_config.substitute_classifier_path = substitute_classifier_path

    # read target classifier
    ok, msg, classifier, classifier_name = read_model(attack_config.classifier_path)
    if ok is False:
        logger.error(msg)
        exit_execution(shared_exit_msg)
    logger.debug(msg)

    if substitute_classifier_path is not None:
        ok, msg, substitute_classifier, substitute_classifier_name = read_model(substitute_classifier_path)
        if ok is False:
            logger.debug(msg)
            exit_execution(shared_exit_msg)
        attack_config.substitute_classifier = tf.keras.models.Model(inputs=substitute_classifier.inputs,
                                                                    outputs=substitute_classifier.outputs,
                                                                    name=substitute_classifier_name)
        attack_config.substitute_classifier = substitute_classifier
        attack_config.substitute_classifier_name = substitute_classifier_name
    else:
        attack_config.substitute_classifier = None

    # list of functions to read classifier model
    read_model_function_list = [read_model_from_h5, read_model_from_latest_format_tf2, read_model_from_checkpoint]
    # check if read model successfully
    ok = False
    classifier = None
    classifier_name = None
    for read_function in read_model_function_list:
        classifier, classifier_name, ok, msg = read_function(classifier_path)
        logger.debug(msg)
        if ok is True:
            break
    if ok is False:
        logger.error(f'Cannot read model from {classifier_path}')
        exit_execution(shared_exit_msg)

    # read model information
    attack_config.classifier = tf.keras.models.Model(inputs=classifier.inputs, outputs=classifier.outputs,
                                                     name=classifier_name)
    attack_config.classifier_name = classifier_name
    model_config = attack_config.classifier.get_config()
    attack_config.input_shape = model_config['layers'][0]['config']['batch_input_shape'][1:]
    attack_config.total_element_a_data = np.prod(attack_config.input_shape)
    attack_config.classifier = BlackBoxClassifier(
        core_classifier=attack_config.classifier) if attack_config.attack_type == ATTACK_BLACKBOX_TYPE else attack_config.classifier

    # get num class (or length of output vector)
    fake_data_sample = np.random.rand(1, *attack_config.input_shape)
    fake_prediction = attack_config.classifier.predict(fake_data_sample)
    attack_config.num_class = len(fake_prediction[0])


def read_model(model_path, logger=None):
    if not check_path_exists(model_path):
        return False, 'Path not found: ' + str(model_path), None, None

    # list of functions to read classifier model by supported formats
    read_model_function_list = [read_model_from_h5, read_model_from_latest_format_tf2, read_model_from_checkpoint]
    ok = False
    classifier = None
    classifier_name = None

    for read_function in read_model_function_list:
        classifier, classifier_name, ok, msg = read_function(model_path)
        logger.debug(msg) if logger is not None else None
        if ok is True:
            break
    msg = 'Found classifier: ' if ok is True else 'Cannot read classifier from: '
    msg += str(model_path)
    return ok, msg, classifier, classifier_name


def read_model_from_h5(classifier_path: str):
    """
    read model with h5 format. Learn more: https://www.tensorflow.org/guide/keras/save_and_serialize#keras_h5_format
    :param classifier_path:  .h5 path to the classifier
    :return:
        model: tensorflow model
        model_name: name of model
        ok: if classifier_path is valid
        msg: returned message
    """
    if not check_file_extension(classifier_path, MODEL_H5_EXTENSION):
        return None, None, False, f'Model format does not match to {MODEL_H5_EXTENSION}\n'

    classifier = tf.keras.models.load_model(classifier_path, custom_objects=custom_objects, compile=False)
    classifier_name = get_file_name(classifier_path)
    # model_config = attack_config.classifier.get_config()
    # attack_config.input_shape = model_config['layers'][0]['config']['batch_input_shape'][1:]
    # attack_config.total_element_a_data = np.prod(attack_config.input_shape)
    return classifier, classifier_name, True, f'Found model with format {MODEL_H5_EXTENSION}'


def read_model_from_latest_format_tf2(classifier_folder: str):
    """
    read model from folder. Learn more: https://www.tensorflow.org/guide/keras/save_and_serialize#whole-model_saving_loading
    :param classifier_folder:path to the classifier folder
    :return:
        model: tensorflow model
        model_name: name of model
        ok: if classifier_path is valid
        msg: returned message
    """
    try:

        classifer = tf.keras.models.load_model(classifier_folder, custom_objects=custom_objects, compile=False)
        classifer_name = get_file_name(classifier_folder)
    except:
        return None, None, False, "Not found model saved in folder as tf.20 version"

    return classifer, classifer_name, True, 'Found model saved in folder as tf2.0 version'


# def read_model_from_pb(classifer_path: str):
#     """
#     read model with pb format. Learn more:
#     :param classifer_path: pd path to the classifier
#     :return:
#         model: tensorflow model
#         model_name: name of model
#         ok: if classifier_path is valid
#         msg: returned message
#     """


def read_model_from_checkpoint(classifier_ckpt_path: str):
    """
    read model with TF Checkpoint format. Learn more: https://www.tensorflow.org/guide/keras/save_and_serialize#tf_checkpoint_format
    :param classifier_path:
    :return:
        model: tensorflow model
        model_name: name of model
        ok: if classifier_path is valid
        msg: returned message
    """
    try:
        classifier = create_model()
        classifier.load_weights(classifier_ckpt_path)
        classifier_name = get_file_name(classifier_ckpt_path)
    except:
        return None, None, False, "Not found model with ckpt format"

    return classifier, classifier_name, True, f'Found model with ckpt format'


def create_model():
    return tf.keras.models.Sequential([
        # TO_DO Please insert stack of layers here
    ])


def validate_two_models(classifier_1: tf.keras.models.Model, classifier_2: tf.keras.models.Model, logger=None):
    config_cls_1 = classifier_1.get_config()
    config_cls_2 = classifier_2.get_config()

    tests = [
        {
            'name': 'test_input_shape',
            'result': config_cls_1['layers'][0]['config']['batch_input_shape'][1:] ==
                      config_cls_2['layers'][0]['config']['batch_input_shape'][1:],
            'expected': True,
            'error message': 'target classifier and substitute classifier should have the same input_shape'
        },
        {
            'name': 'test_output_shape',
            'result': 1,
            'expected': True,
            'error message': 'target classifier and substitute classifier should have the same output_shape'
        }
    ]
    ok = test_loop(test_cases=tests, test_name='Validate target classifier and substitute classifier', logger=logger)
    return ok


def analyze_autoencoder(autoencoder_path, config, logger, shared_exit_msg, custom_objects=None):
    autoencoder_model = None
    autoencoder_model_name = None
    if custom_objects is None:
        custom_objects = {}
    if autoencoder_path is not None and autoencoder_path != '':
        autoencoder_path = os.path.abspath(autoencoder_path)
        ok, msg, autoencoder_model, autoencoder_model_name = read_model(autoencoder_path, logger=logger)
        if ok is False:
            logger.debug(msg)
            exit_execution(shared_exit_msg)
            return
    config.autoencoder_model = autoencoder_model
    config.autoencoder_model_name = autoencoder_model_name