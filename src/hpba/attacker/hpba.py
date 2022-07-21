"""
Created At: 14/07/2021 15:39
"""
import random
import time
import warnings

from tensorflow.python.keras.callbacks import ModelCheckpoint

from src.hpba.attacker.SuccessRateComputationCallback import SuccessRateComputationCallback
from src.hpba.attacker.attacker import Attacker
from src.hpba.attacker.autoencoder import AutoEncoder
from src.hpba.attacker.constants import *
from src.hpba.attacker.losses import AE_LOSSES
from src.hpba.data_preprocessing.data_preprocessing import DataPreprocessing
from src.hpba.utility.autoencoder_config import AutoencoderConfig
from src.hpba.utility.config import attack_config
from src.hpba.utility.constants import *
from src.hpba.utility.optimize_advs import optimize_advs
from src.hpba.utility.statistics import *
from src.hpba.utility.utils import *

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
tf.config.experimental_run_functions_eagerly(True)

logger = AttackLogger.get_logger()


class HPBA(Attacker):
    def __init__(self, origin_label,
                 trainX,
                 trainY,
                 target_classifier,
                 substitute_classifier,
                 weight,
                 step_to_recover=12.,
                 num_images_to_attack=1000,
                 pattern=ALL_PATTERN,
                 num_class=MNIST_NUM_CLASSES,
                 num_images_to_train=1000,
                 use_optimize_phase=True,
                 attack_type=ATTACK_WHITEBOX_TYPE,
                 substitute_classifier_name=None,
                 attack_stop_condition=None,
                 autoencoder_config: AutoencoderConfig = None,
                 quality_loss_str=LOSS_MSE):

        super().__init__(trainX=trainX,
                         trainY=trainY,
                         target_classifier=target_classifier,
                         substitute_classifier=substitute_classifier,
                         substitute_classifier_name=substitute_classifier_name,
                         num_class=num_class,
                         method_name=HPBA_METHOD_NAME,
                         attack_type=attack_type,
                         origin_label=origin_label,
                         attack_stop_condition=attack_stop_condition,
                         quality_loss_str=quality_loss_str)

        self.weight = weight
        self.step_to_recover = step_to_recover
        self.num_images_to_attack = num_images_to_attack if len(self.origin_images) > num_images_to_train else len(
            self.origin_images)

        self.pattern = pattern
        self.num_images_to_train = num_images_to_train
        self.autoencoder_config = autoencoder_config
        self.is_data_inside_0_1_or_0_255 = check_inside_range(self.origin_images,
                                                              checked_range=(0, 1)) or check_inside_range(
            self.origin_images, checked_range=(0, 255))

        if self.attack_type == ATTACK_WHITEBOX_TYPE:
            self.file_shared_name = '{method_name}_{classifier_name}_ori={origin_label}_tar={target_label}_weight={weight}_num={num_images}_loss={quality_loss}'.format(
                method_name=self.method_name, classifier_name=self.classifier_name, origin_label=self.origin_label,
                target_label=-1 if self.is_untargeted() else self.target_label,
                weight=str(self.weight).replace('.', ','),
                num_images=self.num_images_to_train, quality_loss=self.quality_loss_str)
        else:
            self.file_shared_name = '{method_name}_{classifier_name}_{sub_model_name}_ori={origin_label}_tar={target_label}_weight={weight}_num={num_images}loss={quality_loss}'.format(
                method_name=self.method_name, classifier_name=self.classifier_name, origin_label=self.origin_label,
                target_label=-1 if self.is_untargeted() else self.target_label,
                weight=str(self.weight).replace('.', ','),
                num_images=self.num_images_to_train, sub_model_name=self.substitute_cls_name,
                quality_loss=self.quality_loss_str)

        self.autoencoder_folder = os.path.join(self.general_result_folder, TEXT_AUTOENCODER)
        mkdirs([self.autoencoder_folder])
        mkdirs([self.general_result_folder, self.result_summary_folder,
                self.data_folder, self.image_folder])

        self.autoencoder = None
        self.autoencoder_file_path = os.path.join(self.autoencoder_folder,
                                                  self.file_shared_name + '_' + TEXT_AUTOENCODER + '.h5')

        self.optimal_epoch = None
        self.smooth_adv_speed = None
        self.optimized_adv = None
        self.optimized_adv_0_255 = None
        self.optimized_adv_path = os.path.join(self.data_folder,
                                               self.short_file_shared_name + '_optimized_adv_' + self.shared_time_stamp + '.npy')

        self.L0_befores = None
        self.L0_afters = None
        self.L2_befores = None
        self.L2_afters = None
        self.SSIM_befores = None
        self.SSIM_afters = None
        self.use_optimize_phase = use_optimize_phase

    def attack(self):
        input_range = get_range_of_input(self.origin_images)
        ae_trainee = AutoEncoder(input_range=input_range)

        white_box_classifier = self.classifier if self.substitute_classifier is None else self.substitute_classifier

        if check_path_exists(self.autoencoder_file_path) and self.autoencoder_config.autoencoder_model is None:
            logger.debug(self.shared_log +
                         f'found pre-trained autoencoder for: origin_label = {self.origin_label}, target_label = {self.target_label}')
            self.autoencoder = tf.keras.models.load_model(self.autoencoder_file_path, compile=False,
                                                          custom_objects={
                                                              'range_custom_activation': wrap_range_custom_activation(
                                                                  min_value=input_range[0], max_value=input_range[1])
                                                             })

        else:
            logger.debug(self.shared_log +
                         f'not found pre-trained autoencoder for: origin_label = {self.origin_label}, target_label = {self.target_label}')
            logger.debug(
                self.shared_log + f'training autoencoder for: origin_label={self.origin_label}, target_label={self.target_label}')
            if self.autoencoder_config.autoencoder_model is not None:
                self.autoencoder = self.autoencoder_config.autoencoder_model
            else:
                if self.trainX[0].shape[-1] == 3:
                    self.autoencoder = ae_trainee.get_3d_architecture(input_shape=self.trainX[0].shape)
                else:
                    self.autoencoder = ae_trainee.apdative_architecture(input_shape=self.trainX[0].shape)

            adam = tf.keras.optimizers.Adam(learning_rate=self.autoencoder_config.learning_rate, beta_1=0.9,
                                            beta_2=0.999, amsgrad=False)

            au_loss = AE_LOSSES.general_loss(classifier=white_box_classifier, target_vector=self.target_vector,
                                             beta=self.weight, input_shape=self.trainX[0].shape,
                                             is_untargeted=self.is_untargeted(), quality_loss_str=self.quality_loss_str,
                                             num_class=self.num_class)

            self.autoencoder.compile(optimizer=adam, loss=au_loss)

            # add callbacks and train autoencoder
            model_checkpoint = ModelCheckpoint(self.autoencoder_file_path,
                                               save_best_only=True, monitor='loss',
                                               mode='min')
            sr_computation = SuccessRateComputationCallback(
                self.origin_images[:self.num_images_to_train],
                self.target_label,
                self.classifier,
                self.attack_stop_condition,
                self.is_untargeted(), self.origin_label,
                print_every_epoch=self.autoencoder_config.print_result_every_epochs,
                num_class=self.num_class
            )

            history = self.autoencoder.fit(
                self.origin_images[:self.num_images_to_train],
                self.origin_images[:self.num_images_to_train],
                epochs=self.autoencoder_config.epochs,
                batch_size=self.autoencoder_config.batch_size,
                callbacks=[model_checkpoint, sr_computation],
                verbose=1
            )

            self.autoencoder.save(self.autoencoder_file_path)
            logger.debug(self.shared_log + 'training autoencoder DONE!')
            self.optimal_epoch = len(history.history['loss'])

        logger.debug(self.shared_log + 'filtering generated candidates!')
        generated_candidates = self.autoencoder.predict(self.origin_images[:self.num_images_to_attack])

        ranking_strategy = COI_RANKING
        optimized_classifier = self.classifier
        if self.substitute_classifier is not None:
            ranking_strategy = SEQUENTIAL_RANKING

        self.adv_result, _, self.origin_adv_result, _ = filter_candidate_adv(
            origin_data=self.origin_images[:self.num_images_to_attack], candidate_adv=generated_candidates,
            target_label=self.target_label,
            cnn_model=optimized_classifier, is_untargeted=self.is_untargeted(), origin_label=self.origin_label,
            num_class=self.num_class)

        logger.debug(self.shared_log + 'filtering generated candidates DONE!')
        if self.adv_result is None or len(self.adv_result) == 0:
            self.end_time = time.time()
            logger.error("Cannot generate adv")
            return
        self.L0_befores, self.L2_befores, self.SSIM_befores = compute_distance(self.adv_result, self.origin_adv_result)

        # if found adv, continue optimization phase
        if self.use_optimize_phase is True:
            logger.debug(self.shared_log + 'optimizing generated advs')
            actual_recover_step = round(self.step_to_recover * np.average(self.L0_befores))

            self.optimized_adv_0_255 = optimize_advs(classifier=optimized_classifier,
                                                     generated_advs=self.adv_result,
                                                     origin_images=self.origin_adv_result,
                                                     target_label=self.target_label,
                                                     step=actual_recover_step, num_class=self.num_class,
                                                     ranking_type=ranking_strategy,
                                                     batch_size=attack_config.batch_to_optimize,
                                                     epoch_to_optimize=attack_config.epoch_to_optimize,
                                                     is_untargeted=self.is_untargeted())

            self.optimized_adv = np.array(self.optimized_adv_0_255)
            np.save(self.optimized_adv_path, self.optimized_adv)
            self.optimized_adv = np.asarray(self.optimized_adv).reshape(self.adv_result.shape)
            self.L0_afters, self.L2_afters, self.SSIM_afters = compute_distance(self.optimized_adv,
                                                                                self.origin_adv_result)
            logger.debug(self.shared_log + 'optimizing generated advs DONE')
        else:
            self.optimized_adv = None
            self.optimized_adv_0_255 = None

        self.end_time = time.time()
        np.save(self.adv_result_path, self.adv_result)
        np.save(self.origin_adv_result_path, self.origin_adv_result)

    def export_result(self, end_text=''):
        origin_folder = '/origin_' + 'originClass=' + str(self.origin_label) + '_predictedClass=' + str(
            self.origin_label) + '_timestamp=' + self.shared_time_stamp
        optimized_adv_folder = '/optimized_adv_' + 'originClass=' + str(self.origin_label) + '_predictedClass=' + str(
            self.target_label) + '_timestamp=' + self.shared_time_stamp
        non_optimized_adv_folder = '/non_optimized_adv_' + 'originClass=' + str(
            self.origin_label) + '_predictedClass=' + str(
            self.target_label) + '_timestamp=' + self.shared_time_stamp

        if self.adv_result is None:
            self.attack()

        logger.debug(self.shared_log + 'exporting results')

        result = '<=======' + '\n'
        result += 'Configuration:\n'
        result += 'Attack type: '
        result += ATTACK_BLACKBOX_TYPE if self.substitute_classifier is not None else ATTACK_WHITEBOX_TYPE
        result += '\n'
        result += '\tClassifier name: ' + str(self.classifier_name) + '\n'
        result += '\tOriginal label: ' + str(self.origin_label) + '\n'
        if self.is_untargeted():
            result += '\tTarget label: None (untargeted attack)' + '\n'
        else:
            result += '\tTarget label: ' + str(self.target_label) + '\n'
        result += '\tRecover speed: ' + str(self.step_to_recover) + '\n'
        result += '\tWeight: ' + str(self.weight) + '\n'
        result += '\tNumber data to train autoencoder: ' + str(self.num_images_to_train) + '\n'
        if self.num_images_to_attack == -1:
            result += '\tNumber data to attack: ALL \n'
        else:
            result += '\tNumber data to attack: ' + str(self.num_images_to_attack) + '\n'
        result += '\tUse optimize phase: ' + str(self.use_optimize_phase)
        result += '\tQuality loss: ' + str(self.quality_loss_str)
        result += '\n'
        result += 'Attack result:\n'

        if self.adv_result is not None and self.adv_result.shape[0] != 0:
            if self.is_data_inside_0_1_or_0_255:
                save_images_to_folder(folder_path=self.image_folder + non_optimized_adv_folder,
                                      images=self.adv_result,
                                      prefix_file_name='non_optimized_adv', logger=logger)

            result += f'\tSuccess rate: {(self.adv_result.shape[0] / self.num_images_to_attack) * 100.: .2f}%\n'

            result += '\tL0 distance non-optimized (min/max/avg): ' + '{min_l0}/ {max_l0}/ {avg_l0}'.format(
                min_l0=np.min(self.L0_befores),
                max_l0=np.max(self.L0_befores),
                avg_l0=round(
                    np.average(np.round(self.L0_befores, 2)),
                    2))
            result += '\n'

            if self.use_optimize_phase is True:
                result += '\tL0 distance optimized (min/max/avg): ' + '{min_l0}/ {max_l0}/ {avg_l0}'.format(
                    min_l0=np.min(self.L0_afters),
                    max_l0=np.max(self.L0_afters),
                    avg_l0=round(np.average(self.L0_afters), 2))
                result += '\n'

            result += '\tL2 distance non-optimized (min/max/avg): ' + \
                      f'{np.round(np.min(self.L2_befores), 2):.2f}/ {round(np.max(self.L2_befores), 2):.2f}/ {round(np.average(self.L2_befores), 2):.2f}'
            result += '\n'

            if self.use_optimize_phase is True:
                result += '\tL2 distance optimized (min/max/avg): ' + f'{np.min(self.L2_afters):.2f}/ {round(np.max(self.L2_afters), 2):.2f}/ {round(np.average(self.L2_afters), 2):.2f}'
                result += '\n'

            result += '\tSSIM distance non-optimized (min/max/avg): ' + \
                      f'{np.round(np.min(self.SSIM_befores), 2):.2f}/ {round(np.max(self.SSIM_befores), 2):.2f}/ {round(np.average(self.SSIM_befores), 2):.2f}'
            result += '\n'

            if self.use_optimize_phase is True:
                result += '\tSSIM distance optimized (min/max/avg): ' + f'{np.min(self.SSIM_afters):.2f}/ {round(np.max(self.SSIM_afters), 2):.2f}/ {round(np.average(self.SSIM_afters), 2):.2f}'
                result += '\n'

            if self.use_optimize_phase is True:
                if self.is_data_inside_0_1_or_0_255:
                    save_images_to_folder(folder_path=self.image_folder + optimized_adv_folder,
                                          images=self.optimized_adv_0_255,
                                          prefix_file_name='optimized_adv', logger=logger)
                result += f'\tExecution time: {np.round(self.end_time - self.start_time, 2):.2f} seconds'
                result += '\n'

                result += '\tAutoencoder path: ' + str(self.autoencoder_file_path) + '\n'
                result += '\tAdv path: ' + str(self.adv_result_path) + '\n'
                result += '\tOptimized adv path: ' + str(self.optimized_adv_path) + '\n'
                result += '\tOriginal data path: ' + str(self.origin_adv_result_path) + '\n'
                save_images_to_folder(folder_path=self.image_folder + origin_folder,
                                      images=self.origin_adv_result,
                                      prefix_file_name='origin', logger=logger)

        else:
            result += '\tSuccess rate: 0%' + '\n'
            result += f'\tExecution time: {np.round(self.end_time - self.start_time, 2):.2f} seconds'
            result += '\n'

        result += '=======>\n'
        result += end_text
        write_to_file(content=result, path=self.summary_path)

        logger.debug(self.shared_log + 'exporting results DONE!')
        logger.info(f'Please open file: {self.summary_path} to view results!')

    def plot_some_random_images(self):
        if self.adv_result is None or len(self.adv_result) == 0:
            return
        if not self.is_data_inside_0_1_or_0_255:
            return
        fig = plt.figure(figsize=(8, 8))
        columns = 3
        rows = 3
        images = []
        tmp_optimized = self.optimized_adv if self.use_optimize_phase is True else self.adv_result
        ramdom_indexs = [random.randint(0, tmp_optimized.shape[0] - 1) for i in range(rows)]
        for i in ramdom_indexs:
            images.append(self.origin_adv_result[i])
            images.append(self.adv_result[i])
            images.append(tmp_optimized[i])
        for i in range(1, columns * rows + 1):
            img = images[i - 1]
            ax = plt.subplot(rows, columns, i)
            if i % rows == 1:
                title = 'origin'
            elif i % rows == 2:
                title = 'adv'
            else:
                title = 'optimized adv'
                if self.use_optimize_phase is False:
                    title += '(not use)'
                    img = np.ones_like(img)
            if img.shape[-1] == 1:
                plt.imshow(img.reshape(img.shape[:-1]), cmap='gray')
            else:
                plt.imshow(img, cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i < 4:
                ax.set_title(title)
        fig.savefig(self.image_path)
        plt.close(fig)


if __name__ == '__main__':
    logger.debug('pre-processing data')
    (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

    trainX, trainY = DataPreprocessing.quick_preprocess_data(trainX, trainY, num_classes=MNIST_NUM_CLASSES,
                                                             rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                             chl=MNIST_IMG_CHL)
    testX, testY = DataPreprocessing.quick_preprocess_data(testX, testY, num_classes=MNIST_NUM_CLASSES,
                                                           rows=MNIST_IMG_ROWS, cols=MNIST_IMG_COLS,
                                                           chl=MNIST_IMG_CHL)

    np.save('../../data/mnist/mnist_training.npy', trainX)
    np.save('../../data/mnist/mnist_label.npy', trainY)
