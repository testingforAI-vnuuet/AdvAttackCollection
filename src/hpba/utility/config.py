import configparser

from _test.attacker.test_input_config import test_input_config
from attacker.constants import *
from utility.autoencoder_config import AutoencoderConfig
from utility.constants import *
from utility.model_utils import analyze_classifier, validate_two_models, analyze_autoencoder
from utility.mylogger import *
from utility.utils import *
from utility.utils import exit_execution

logger = MyLogger().getLog()


class attack_config:
    # classifier
    classifier_path = None
    classifier_name = None
    classifier = None

    substitute_classifier_path = None
    substitute_classifier_name = None
    substitute_classifier = None

    # attack
    attack_type = ATTACK_WHITEBOX_TYPE
    original_class = None
    target_class = None
    target_position = None
    recover_speed = None
    weight = None
    number_data_to_attack = None
    number_data_to_train_autoencoder = None
    max_number_advs_to_optimize = None
    use_optimize_phase = True
    quality_loss = None
    L2_threshold_to_stop_attack = None
    L0_threshold_to_stop_attack = None
    SSIM_threshold_to_stop_attack = None

    # data
    training_path = None
    training_data = None
    label_path = None
    label_data = None
    num_class = None
    input_shape = None
    input_range = None
    input_size = None
    total_element_a_data = None
    epoch_to_optimize = None
    batch_to_optimize = None
    data_name = None

    # autoencoder training
    autoencoder_model = None
    autoencoder_model_name = None
    epochs = 500
    batch_size = 256
    print_result_every_epochs = 3
    learning_rate = 0.001
    autoencoder_config = None


def analyze_config(config_path):
    config_parser = configparser.ConfigParser()
    config_parser.read(config_path)

    # untargeted attack
    attack_config.L2_threshold_to_stop_attack = config_parser[CONFIG_TXT.ATTACK][
        CONFIG_TXT.L2thresholdtoStopAttack]
    if attack_config.L2_threshold_to_stop_attack == '':
        attack_config.L2_threshold_to_stop_attack = None
    else:
        attack_config.L2_threshold_to_stop_attack = float(attack_config.L2_threshold_to_stop_attack)

    attack_config.L0_threshold_to_stop_attack = config_parser[CONFIG_TXT.ATTACK][
        CONFIG_TXT.L0thresholdtoStopAttack]
    if attack_config.L0_threshold_to_stop_attack == '':
        attack_config.L0_threshold_to_stop_attack = None
    else:
        attack_config.L0_threshold_to_stop_attack = float(attack_config.L0_threshold_to_stop_attack)

    attack_config.SSIM_threshold_to_stop_attack = config_parser[CONFIG_TXT.ATTACK][
        CONFIG_TXT.SSIMthresholdtoStopAttack]
    if attack_config.SSIM_threshold_to_stop_attack == '':
        attack_config.SSIM_threshold_to_stop_attack = None
    else:
        attack_config.SSIM_threshold_to_stop_attack = float(
            attack_config.SSIM_threshold_to_stop_attack)

    # attack type
    attack_type_int = int(config_parser[CONFIG_TXT.CLASSIFIER][CONFIG_TXT.attack_type])
    if attack_type_int not in [0, 1]:
        logger.error(shared_incorrect_para_msg.format(param=CONFIG_TXT.attack_type))
        exit_execution(shared_exit_msg)
    attack_config.attack_type = ATTACK_WHITEBOX_TYPE if attack_type_int == 1 else ATTACK_BLACKBOX_TYPE

    #  classifier path
    analyze_classifier(config_parser=config_parser, attack_config=attack_config, logger=logger,
                       shared_exit_msg=shared_exit_msg)

    if attack_config.attack_type == ATTACK_BLACKBOX_TYPE and attack_config.substitute_classifier is not None:
        ok = validate_two_models(classifier_1=attack_config.classifier.core_classifier,
                                 classifier_2=attack_config.substitute_classifier,
                                 logger=logger)
        if not ok:
            exit_execution()

    attack_config.original_class = int(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.originalLabel])
    attack_config.weight = float(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.weight])
    attack_config.number_data_to_attack = int(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.numberDataToAttack])
    attack_config.number_data_to_train_autoencoder = int(
        config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.numberDataToTrainAutoencoder])
    attack_config.use_optimize_phase = int(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.useOptimizePhase])
    attack_config.recover_speed = float(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.recoverSpeed])
    attack_config.epoch_to_optimize = int(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.epochToOptimize])
    attack_config.batch_to_optimize = int(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.batchToOptimize])
    attack_config.epochs = int(config_parser[CONFIG_TXT.AUTOENCODER_TRAINING][CONFIG_TXT.epochs])
    attack_config.batch_size = int(config_parser[CONFIG_TXT.AUTOENCODER_TRAINING][CONFIG_TXT.batch_size])
    attack_config.print_result_every_epochs = int(
        config_parser[CONFIG_TXT.AUTOENCODER_TRAINING][CONFIG_TXT.print_result_every_epochs])
    attack_config.learning_rate = float(config_parser[CONFIG_TXT.AUTOENCODER_TRAINING][CONFIG_TXT.learning_rate])
    attack_config.quality_loss = str(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.quality_loss]).lower()

    ok = test_input_config(attack_config, logger)
    if not ok:
        exit_execution(shared_exit_msg)

    attack_config.max_number_advs_to_optimize = attack_config.number_data_to_attack
    attack_config.use_optimize_phase = True if attack_config.use_optimize_phase == 1 else False

    autoencoder_path = str(config_parser[CONFIG_TXT.AUTOENCODER_TRAINING][CONFIG_TXT.autoencoder_model_path])
    analyze_autoencoder(autoencoder_path=autoencoder_path, config=attack_config, logger=logger,
                        shared_exit_msg=shared_exit_msg, custom_objects=None)
    attack_config.autoencoder_config = AutoencoderConfig(epochs=attack_config.epochs,
                                                         batch_size=attack_config.batch_size,
                                                         print_result_every_epochs=attack_config.print_result_every_epochs,
                                                         learning_rate=attack_config.learning_rate,
                                                         autoencoder_model=attack_config.autoencoder_model,
                                                         autoencoder_model_name=attack_config.autoencoder_model_name)

    # data path
    if int(config_parser[CONFIG_TXT.DATA][CONFIG_TXT.useDataFolder]) == 0:
        read_data_from_npy(config_parser=config_parser, attack_config=attack_config, logger=logger,
                           shared_exit_msg=shared_exit_msg)
    else:
        read_image_from_folder(config_parser=config_parser, attack_config=attack_config, logger=logger,
                               shared_exit_msg=shared_exit_msg)

    attack_config.input_range = get_range_of_input(attack_config.training_data)

    # get target class
    target_class = int(config_parser[CONFIG_TXT.ATTACK][CONFIG_TXT.targetLabel])
    if target_class == attack_config.original_class:
        logger.error(shared_incorrect_para_msg.format(
            param=CONFIG_TXT.targetLabel) + ' It should not be original label.')

    # if target_class not in range(-1, attack_config.num_class):
    #     if target_position not in range(1, attack_config.num_class + 1):
    #         logger.error(f'target label or target position are not correct')
    #         exit_execution(shared_exit_msg)
    #     else:
    #         attack_config.target_position = target_position
    # else:
    #     attack_config.target_class = target_class
    #     attack_config.target_position = None

    if target_class not in range(-1, attack_config.num_class):
        logger.error(f'target label is not correct')
        exit_execution(shared_exit_msg)
    else:
        attack_config.target_class = target_class
