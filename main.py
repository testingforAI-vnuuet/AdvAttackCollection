from src.attack_config.config import AttackConfig
from src.attack_config.config_keys import *
from src.bim_pgd.UntargetedBIM_PGD import UntargetedBIM_PGD
from src.bis.Untargeted_BIS import UntargetedBIS
from src.gaussian.UntargetedGaussian import UntargetedGaussian
from src.utils.utils import exportAttackResult


class AdvGenerator:
    def __init__(self, config_filepath):
        self.config_parser = AttackConfig(config_filepath)

        self.output_folder = self.config_parser.output_folder
        self.target_classifier = self.config_parser.target_classifier
        self.images = self.config_parser.images
        self.labels = self.config_parser.labels

        self.attack_config = self.config_parser.attack_config

    def attack(self):
        for key in self.attack_config.keys():
            if key == UNTARGETED_FGSM:
                fgsm_config = self.attack_config[key]
                if fgsm_config[ENABLE] == TRUE:
                    epsilons = fgsm_config[EPSILON]
                    for ep in epsilons:
                        attacker = UntargetedBIM_PGD(
                            X=self.images,
                            Y=self.labels,
                            target_classifier=self.target_classifier,
                            epsilon=float(ep),
                            batch_size=int(fgsm_config[BATCH_SIZE]),
                            max_iteration=1,
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_fgsm_ep={ep}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels
                        )

            if key == UNTARGETED_BIM_PGD:
                bim_pgd_config = self.attack_config[key]
                if bim_pgd_config[ENABLE] == TRUE:
                    epsilons = bim_pgd_config[EPSILON]
                    for ep in epsilons:
                        attacker = UntargetedBIM_PGD(
                            X=self.images,
                            Y=self.labels,
                            target_classifier=self.target_classifier,
                            epsilon=float(ep),
                            batch_size=int(bim_pgd_config[BATCH_SIZE]),
                            max_iteration=int(bim_pgd_config[MAX_ITERATION])
                        )
                        final_origin, final_advs, final_true_labels = attacker.attack()
                        exportAttackResult(
                            output_folder=self.output_folder,
                            target_classifier=self.target_classifier,
                            name=f'untargeted_bim_pgd_ep={ep}_iter={int(bim_pgd_config[MAX_ITERATION])}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels
                        )

            if key == UNTARGETED_BIS:
                bis_config = self.attack_config[key]
                if bis_config[ENABLE] == TRUE:
                    epsilons = bis_config[EPSILON]
                    print(epsilons)
                    for ep in epsilons:
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
                            name=f'untargeted_bis_ep={ep}_iter={int(bis_config[MAX_ITERATION])}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels
                        )

            if key == UNTARGETED_GAUSS:
                gauss_config = self.attack_config[key]
                if gauss_config[ENABLE] == TRUE:
                    epsilons = gauss_config[EPSILON]
                    for ep in epsilons:
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
                            name=f'untargeted_gauss_ep={ep}_iter={int(gauss_config[MAX_ITERATION])}',
                            final_advs=final_advs,
                            final_origin=final_origin,
                            final_true_labels=final_true_labels
                        )


if __name__ == "__main__":
    adv_generator = AdvGenerator("D:\Things\PyProject\AdvAttackCollection\config.ini")
    adv_generator.attack()
