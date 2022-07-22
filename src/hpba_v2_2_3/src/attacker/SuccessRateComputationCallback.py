from tensorflow import keras

from src.hpba_v2_2_3.src.utility.statistics import filter_candidate_adv
from src.hpba_v2_2_3.src.utility.utils import compute_distance
import numpy as np

'''
This callback is used to control the training process of autoencoder
'''


class SuccessRateComputationCallback(keras.callbacks.Callback):

    def __init__(self, origin_images, target_label, classifier, attack_stop_condition, is_untargeted,
                 origin_label, print_every_epoch, num_class):
        super(SuccessRateComputationCallback, self).__init__()
        self.origin_images = origin_images
        self.target_label = target_label
        self.classifier = classifier
        self.is_untargeted = is_untargeted
        self.origin_label = origin_label
        self.L0_threshold_to_stop_attack, self.L2_threshold_to_stop_attack, self.SSIM_threshold_to_stop_attack \
            = attack_stop_condition
        self.print_every_epoch = print_every_epoch
        self.num_class = num_class

    def on_epoch_end(self, epoch, logs=None):
        max_epoch = self.params.get('epochs', -1)
        if epoch % self.print_every_epoch == 0 or epoch == max_epoch:

            adv_result, origin_adv_result = self.compute_sr()
            if len(adv_result) == 0:
                return

            L0, L2, SSIM = self.compute_distance(adv_result, origin_adv_result)

            n_satis = 0
            if self.L2_threshold_to_stop_attack is not None \
                    and self.L2_threshold_to_stop_attack > 0:
                if np.average(L2) <= self.L2_threshold_to_stop_attack:
                    print(f'\t| Reach the average L2 threshold')
                    n_satis += 1
                else:
                    print(f'\t| Average L2 is unsatisfiable!')

            if self.L0_threshold_to_stop_attack is not None \
                    and self.L0_threshold_to_stop_attack > 0:
                if np.average(L0) <= self.L0_threshold_to_stop_attack:
                    print(f'\t| Reach the average L0 threshold')
                    n_satis += 1
                else:
                    print(f'\t| Average L0 is unsatisfiable!')

            if self.SSIM_threshold_to_stop_attack is not None \
                    and self.SSIM_threshold_to_stop_attack > 0:
                if np.average(SSIM) >= self.SSIM_threshold_to_stop_attack:
                    print(f'\t| Reach the average SSIM threshold')
                    n_satis += 1
                else:
                    print(f'\t| Average SSIM is unsatisfiable!')

            if n_satis > 0:
                if self.L2_threshold_to_stop_attack is not None:
                    n_satis -= 1

                if self.L0_threshold_to_stop_attack is not None:
                    n_satis -= 1

                if self.SSIM_threshold_to_stop_attack is not None:
                    n_satis -= 1

                if n_satis == 0:
                    self.model.stop_training = True

    def compute_sr(self):
        '''
        compute success rate
        :return:
        '''
        generated_candidates = self.model.predict(self.origin_images)
        adv_result, _, origin_adv_result, _ = filter_candidate_adv(
            origin_data=self.origin_images,
            candidate_adv=generated_candidates,
            target_label=self.target_label,
            cnn_model=self.classifier, is_untargeted=self.is_untargeted, origin_label=self.origin_label,
            num_class=self.num_class)
        print(f"\t| Current success rate = {len(adv_result) * 100. / len(self.origin_images): .2f}%")

        return adv_result, origin_adv_result

    def compute_distance(self, adv_result, origin_adv_result):
        L0, L2, SSIM = compute_distance(adv_result, origin_adv_result)
        print(f'\t| L0 min/max/avg: {np.min(L0)}/ {np.max(L0)}/ {np.average(L0):.2f}')
        print(
            f'\t| L2 min/max/avg: {np.round(np.min(L2), 2):.2f}/ {np.round(np.max(L2), 2):.2f}/ {np.round(np.average(L2), 2):.2f}')
        print(
            f'\t| SSIM min/max/avg: {np.round(np.min(SSIM), 2):.2f}/ {np.round(np.max(SSIM), 2):.2f}/ {np.round(np.average(SSIM), 2):.2f}')
        return L0, L2, SSIM
