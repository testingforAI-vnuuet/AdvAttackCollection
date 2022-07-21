from src.hpba.attacker.hpba import *
from src.hpba.utility.config import attack_config, analyze_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.config.run_functions_eagerly(True)

logger = AttackLogger.get_logger()

config_file = os.path.abspath('D:\Things\PyProject\AdvAttackCollection\src\hpba\config.ini')

if __name__ == '__main__':
    logger.debug('robustness START')
    logger.debug('reading configuration')
    analyze_config(config_file)
    logger.debug('reading configuration DONE')
    attacker = HPBA(origin_label=attack_config.original_class, trainX=attack_config.training_data,
                    trainY=attack_config.label_data,
                    weight=attack_config.weight, target_classifier=attack_config.classifier,
                    step_to_recover=attack_config.recover_speed,
                    num_images_to_attack=attack_config.number_data_to_attack,
                    num_images_to_train=attack_config.number_data_to_train_autoencoder,
                    num_class=attack_config.num_class,
                    use_optimize_phase=attack_config.use_optimize_phase,
                    substitute_classifier=attack_config.substitute_classifier,
                    attack_type=attack_config.attack_type,
                    substitute_classifier_name=attack_config.substitute_classifier_name,
                    attack_stop_condition=(attack_config.L0_threshold_to_stop_attack,
                                           attack_config.L2_threshold_to_stop_attack,
                                           attack_config.SSIM_threshold_to_stop_attack),
                    autoencoder_config=attack_config.autoencoder_config,
                    quality_loss_str=attack_config.quality_loss)
    attacker.attack()
    attacker.plot_some_random_images()
    attacker.export_result()
