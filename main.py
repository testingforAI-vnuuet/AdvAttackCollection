import numpy as np

from src.attack_config.config import AttackConfig
from src.attack_config.config_keys import *
from src.bim_pgd.UntargetedBIM_PGD import UntargetedBIM_PGD
from src.bis.Untargeted_BIS import UntargetedBIS
from src.cw.CarniliWangerL2 import CarliniWagnerL2
from src.gaussian.UntargetedGaussian import UntargetedGaussian
from src.mi_fgsm.UntargetedMI_FGSM import UntargetedMI_FGSM
from src.utils import utils
from src.utils.attack_logger import AttackLogger
from src.utils.utils import exportAttackResult
from src.hpba_v2_2_3.src.attacker.hpba import HPBA
from src.hpba_v2_2_3.src.utility.config import attack_config as hpba_config
from src.hpba_v2_2_3.src.utility.config import analyze_config as hpba_analyze_config

logger = AttackLogger.get_logger()


class AdvGenerator:
    def __init__(self, config_filepath,
                 limit_origins=None  # just use for testing
                 ):
        self.config_parser = AttackConfig(config_filepath)

        self.output_folder = self.config_parser.output_folder
        self.target_classifier = self.config_parser.target_classifier

        self.images = self.config_parser.images[:limit_origins]
        self.labels = self.config_parser.labels[:limit_origins]

        # self.images = self.config_parser.images
        # self.labels = self.config_parser.labels

        self.attack_config = self.config_parser.attack_config

        pred = self.target_classifier.predict(self.images)
        pred = np.argmax(pred, axis=1)
        true_indexes = np.where(pred == self.labels)[0]
        logger.debug(f'Number of correctly predicted images = {len(true_indexes)}')
        self.images = self.images[true_indexes]
        self.labels = self.labels[true_indexes]

    def attack(self):
        for key in self.attack_config.keys():
            if key == UNTARGETED_FGSM:
                fgsm_config = self.attack_config[key]
                if fgsm_config[ENABLE] == TRUE:
                    epsilons = fgsm_config[EPSILON]
                    for ep in epsilons:
                        show_method(f'FGSM with ep = {ep}')
                        attacker = UntargetedBIM_PGD(
                            X=self.images,
                            Y=self.labels,
                            target_classifier=self.target_classifier,
                            epsilon=float(ep),
                            batch_size=int(fgsm_config[BATCH_SIZE]),
                            max_iteration=1
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_fgsm_ep={ep}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels,
                            logger=logger,
                        )

            elif key == UNTARGETED_MI_FGSM:
                mi_fgsm_config = self.attack_config[key]
                if mi_fgsm_config[ENABLE] == TRUE:
                    epsilons = mi_fgsm_config[EPSILON]
                    for ep in epsilons:
                        show_method(f'{UntargetedMI_FGSM.__name__} with ep = {ep}')
                        attacker = UntargetedMI_FGSM(
                            X=self.images,
                            Y=self.labels,
                            alpha=float(ep),
                            target_classifier=self.target_classifier,
                            batch_size=int(mi_fgsm_config[BATCH_SIZE]),
                            max_iteration=int(mi_fgsm_config[MAX_ITERATION]),
                            decay_factor=float(mi_fgsm_config[DECAY_FACTOR])
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_mi_fgsm_ep={ep}_iter={mi_fgsm_config[MAX_ITERATION]}'
                                 f'_decay={mi_fgsm_config[DECAY_FACTOR]}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels,
                            logger=logger,
                        )

            elif key == UNTARGETED_BIM_PGD:
                bim_pgd_config = self.attack_config[key]
                if bim_pgd_config[ENABLE] == TRUE:
                    epsilons = bim_pgd_config[EPSILON]
                    for ep in epsilons:
                        show_method(f'{UntargetedBIM_PGD.__name__} with epsilon = {ep}')
                        attacker = UntargetedBIM_PGD(
                            X=self.images,
                            Y=self.labels,
                            target_classifier=self.target_classifier,
                            epsilon=float(ep),
                            batch_size=int(bim_pgd_config[BATCH_SIZE]),
                            max_iteration=int(bim_pgd_config[MAX_ITERATION]),
                            max_ball=float(bim_pgd_config[MAX_BALL])
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_bim_pgd_ep={ep}_iter={bim_pgd_config[MAX_ITERATION]}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels,
                            logger=logger,
                        )

            elif key == UNTARGETED_BIS:
                bis_config = self.attack_config[key]
                if bis_config[ENABLE] == TRUE:
                    epsilons = bis_config[EPSILON]
                    for ep in epsilons:
                        show_method(f'{UntargetedBIS.__name__} with epsilon = {ep}')
                        attacker = UntargetedBIS(
                            X=self.images,
                            Y=self.labels,
                            target_classifier=self.target_classifier,
                            epsilon=float(ep),
                            batch_size=int(bis_config[BATCH_SIZE]),
                            max_iteration=int(bis_config[MAX_ITERATION])
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_bis_ep={ep}_iter={bis_config[MAX_ITERATION]}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels,
                            logger=logger,
                        )

            elif key == UNTARGETED_GAUSS:
                gauss_config = self.attack_config[key]
                if gauss_config[ENABLE] == TRUE:
                    epsilons = gauss_config[EPSILON]
                    for ep in epsilons:
                        show_method(f'{UntargetedGaussian.__name__} with epsilon = {ep}')
                        attacker = UntargetedGaussian(
                            X=self.images,
                            Y=self.labels,
                            target_classifier=self.target_classifier,
                            epsilon=float(ep),
                            batch_size=int(gauss_config[BATCH_SIZE]),
                            max_iteration=int(gauss_config[MAX_ITERATION])
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_gauss_ep={ep}_iter={gauss_config[MAX_ITERATION]}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels,
                            logger=logger,
                        )

            elif key == HPBA_ATTACK:
                hpba_attack_config = self.attack_config[key]
                if hpba_attack_config[ENABLE] == TRUE:
                    show_method(f'{HPBA.__name__}')
                    hpba_config_path = hpba_attack_config[CONFIG_FILEPATH]
                    hpba_output_path = hpba_attack_config[OUTPUT_FOLDER]
                    hpba_config.num_class = self.config_parser.num_class
                    hpba_config.input_shape = self.config_parser.classifier_input_shape
                    hpba_analyze_config(hpba_config_path)
                    attacker = HPBA(
                        origin_label=hpba_config.original_class, trainX=self.images,
                        trainY=self.labels, target_label=hpba_config.target_class,
                        weight=hpba_config.weight, target_classifier=self.target_classifier,
                        step_to_recover=hpba_config.recover_speed,
                        num_images_to_attack=hpba_config.number_data_to_attack,
                        num_images_to_train=hpba_config.number_data_to_train_autoencoder,
                        num_class=hpba_config.num_class,
                        use_optimize_phase=hpba_config.use_optimize_phase,
                        substitute_classifier=self.target_classifier,
                        attack_type=hpba_config.attack_type,
                        substitute_classifier_name=hpba_config.substitute_classifier_name,
                        attack_stop_condition=(hpba_config.L0_threshold_to_stop_attack,
                                               hpba_config.L2_threshold_to_stop_attack,
                                               hpba_config.SSIM_threshold_to_stop_attack),
                        autoencoder_config=hpba_config.autoencoder_config,
                        quality_loss_str=hpba_config.quality_loss,
                        outputFolder=hpba_output_path
                    )
                    attacker.attack()

                    if len(attacker.origin_adv_result) > 0:
                        advs = attacker.optimized_adv if attacker.use_optimize_phase else attacker.adv_result
                        true_labels = np.argmax(
                            self.target_classifier.predict(attacker.origin_adv_result), axis=1
                        ).reshape(-1)
                        utils.confirm_adv_attack(self.target_classifier, advs, attacker.origin_adv_result,
                                                 true_labels,
                                                 self.images)
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'hpba_attack',
                            final_advs=advs,
                            final_origin=attacker.origin_adv_result,
                            final_true_labels=true_labels,
                            logger=logger,
                        )
                    else:
                        logger.debug("There is no generated adversarial examples")

            elif key == UNTARGETED_CW_L2:
                cw_l2_config = self.attack_config[key]
                if cw_l2_config[ENABLE] == TRUE:
                    confidences = cw_l2_config[CONFIDENCE]
                    for conf in confidences:
                        show_method(f'{CarliniWagnerL2.__name__} with confidence = {conf}')
                        attacker = CarliniWagnerL2(
                            model_fn=self.target_classifier,
                            batch_size=int(cw_l2_config[BATCH_SIZE]),
                            confidence=float(conf),
                            max_iterations=int(cw_l2_config[MAX_ITERATION])
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack(self.images, self.labels)
                        utils.confirm_adv_attack(self.target_classifier, final_advs, final_origin,
                                                 final_true_labels,
                                                 self.images)
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_cw_l2_conf={conf}_iter={cw_l2_config[MAX_ITERATION]}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels,
                            logger=logger,
                        )


def show_method(name):
    logger.debug('')
    logger.debug('')
    logger.debug('')
    logger.info(
        '-----------------------------------------------------------------------------------------------------------------------------------------------')
    logger.info('|')
    logger.info(
        f'|                                               {str(name).lower().replace("untargeted", "untargeted_").upper()}')
    logger.info('|')
    logger.info(
        '-----------------------------------------------------------------------------------------------------------------------------------------------')


if __name__ == "__main__":
    adv_generator = AdvGenerator("/Users/ducanhnguyen/Documents/testingforAI-vnuuet/AdvAttackCollection/config.ini",
                                 limit_origins=100)
    adv_generator.attack()
