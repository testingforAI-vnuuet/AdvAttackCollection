"""
Created by khadm on 1/5/2022
Feature: 
"""
from src.hpba.attacker.constants import LOSS_MSE, LOSS_SSIM, LOSS_SSIM_MULTISCALE
from src.hpba.utility.constants import CONFIG_TXT, shared_incorrect_para_msg
import numpy as np
from src.hpba._test.run_testcase import test_loop


def test_input_config(attack_config, logger):
    tests = [
        {
            'name': f'test {CONFIG_TXT.originalLabel}',
            'result': attack_config.original_class in list(range(0,
                                                                 attack_config.num_class)) or attack_config.original_class == 0 or attack_config.original_class == 1,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.originalLabel) + f' It should in ({0}, {attack_config.num_class})'
        },
        {
            'name': f'test {CONFIG_TXT.weight}',
            'result': 0.00001 <= attack_config.weight <= 1,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.weight) + ' It should stay between (0, 1)'
        },
        {
            'name': f'test {CONFIG_TXT.numberDataToAttack}',
            'result': attack_config.number_data_to_attack > 0 or attack_config.number_data_to_attack == -1,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.numberDataToAttack) + ' It should be greater than 0.'
        },
        {
            'name': f'test {CONFIG_TXT.numberDataToTrainAutoencoder}',
            'result': attack_config.number_data_to_train_autoencoder > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.numberDataToTrainAutoencoder) + ' It should be greater than 0.'
        },
        {
            'name': f'test {CONFIG_TXT.useOptimizePhase}',
            'result': attack_config.use_optimize_phase in [0, 1],
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.useOptimizePhase) + ' It should be 0 or 1.'
        },
        {
            'name': f'test {CONFIG_TXT.recoverSpeed}',
            'result': attack_config.recover_speed in np.arange(0.1, 1, 0.1),
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.recoverSpeed) + f' It should in {list(np.arange(0.1, 1, 0.1))}'
        },
        {
            'name': f'test {CONFIG_TXT.epochToOptimize} and {CONFIG_TXT.batchToOptimize}',
            'result': attack_config.epoch_to_optimize > 0 and attack_config.batch_to_optimize > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.epochToOptimize) + ' or ' + shared_incorrect_para_msg.format(
                param=CONFIG_TXT.batchToOptimize)
        },
        {
            'name': f'test {CONFIG_TXT.epochs}',
            'result': attack_config.epochs > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.epochs) + 'It should be great than 0.'
        },
        {
            'name': f'test {CONFIG_TXT.batch_size}',
            'result': attack_config.batch_size > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.batch_size) + 'It should be great than 0.'
        },
        {
            'name': f'test {CONFIG_TXT.print_result_every_epochs}',
            'result': attack_config.print_result_every_epochs > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.print_result_every_epochs) + 'It should be great than 0.'
        },
        {
            'name': f'test {CONFIG_TXT.learning_rate}',
            'result': attack_config.learning_rate > 0,
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.learning_rate) + 'It should be great than 0.'
        },
        {
            'name': f'test {CONFIG_TXT.quality_loss}',
            'result': attack_config.quality_loss in [LOSS_MSE, LOSS_SSIM, LOSS_SSIM_MULTISCALE],
            'expected': True,
            'error_message': shared_incorrect_para_msg.format(
                param=CONFIG_TXT.quality_loss) + f'It should be in [{LOSS_MSE}, {LOSS_SSIM}, {LOSS_SSIM_MULTISCALE}].'
        },
    ]
    ok = test_loop(test_cases=tests, test_name='validate input config', logger=logger)
    return ok
