import sys

from configobj import ConfigObj
from config_keys import *
from src.utils.attack_logger import AttackLogger
from src.utils.utils import *

logger = AttackLogger().get_logger()


class AttackConfig:
    def __init__(self, config_filepath):
        self.config_filepath = config_filepath
        self.config_parser = ConfigObj(self.config_filepath)

        # attributes for general configurations: output folder, target classifier, data
        self.general_config = None
        self.output_folder = None  # folder path to save attack results
        self.pixel_range = None  # range of valid pixel value of input images

        self.target_classifier = None  # target classifier to be attacked
        self.classifier_input_shape = None
        self.classifier_output_shape = None
        self.num_class = None

        self.images = None
        self.labels = None

        self.analyze_config()

    def analyze_config(self):
        # analyze configurations, analyzing orders must be keep unchanged

        # analyze general configurations
        self.general_config = self.config_parser[GENERAL_CONFIG]

        # validate output folder path
        self.output_folder = self.general_config[OUTPUT_FOLDER]
        if not check_path_exists(self.output_folder):
            mkdir(self.output_folder)
        elif any(os.scandir(self.output_folder)):
            logger.warning(f'Folder {self.output_folder} is not empty!')

        self.pixel_range = np.array(self.general_config[PIXEL_RANGE], dtype='float32')

        self.load_target_classifier()
        self.load_data()

        # analyze attack configuration
        # Todo: Define attack class
        # Todo: analyze attack configuration

        # temporarily disable to debug
        # try:
        #     self.general_config = self.config_parser[GENERAL_CONFIG]
        #     self.load_target_classifier()
        #     self.load_data()
        # except Exception as e:
        #     logger.debug(f'Failed to read attack\'s configurations: {e}')
        #     sys.exit()

    def load_target_classifier(self):
        # load target classifier
        path_to_classifier = str(self.general_config[TARGET_CLASSIFIER_PATH])
        logger.info(f'Reading target classifier at: {path_to_classifier}')

        model, _, success, message = load_model(path_to_classifier, logger)  # load from file model.h5 or model folder
        if not success:
            logger.error(message)
            sys.exit()
        logger.info(message)

        self.target_classifier = model
        self.classifier_input_shape = model.inputs[0].shape[1:]  # get input shape of target classifier
        self.classifier_output_shape = model.outputs[0].shape[1:]  # get output shape of target classifier
        self.num_class = self.classifier_output_shape[0]

    def load_data(self):
        # load test data
        if self.general_config[USE_DATA_FOLDER] == '1':
            self.images, self.labels = load_image_from_folder(
                self.general_config[DATA_FOLDER_PATH], self.classifier_input_shape[:-1], logger
            )
        else:
            self.images = load_data_from_npy(self.general_config[IMAGES_DATA_PATH], logger)
            self.labels = load_data_from_npy(self.general_config[LABELS_DATA_PATH], logger)

        if len(self.images) != len(self.labels):
            logger.error(f'Size of images ({len(self.images)}) set must be '
                         f'identical to  size of labels set ({len(self.labels)})')
            sys.exit()

        # evaluate images shape and classifier input shape
        if self.images.shape[1:] != self.classifier_input_shape:
            logger.error(f'Images shape {self.images.shape[1:]} must be '
                         f'identical to classifier\'s input shape {self.classifier_input_shape}')
            sys.exit()

        # Todo: Define general input label shape of attack methods
        # # whether label is in one-hot vector shape or not
        # # temporarily disable because of undefined input label shape of each attack method
        # if self.labels.shape[1:] == (1,) or len(self.labels.shape) == 1:
        #     logger.warning(f'Labels set may not be in one-hot vector shape: {self.labels.shape[1:]}')
        #     logger.debug(f'Converting labels set to one-hot vectors')
        #
        #     # check whether sparse labels data value is in valid range or not
        #     if np.min(self.labels) < 0 or np.max(self.labels) > self.classifier_output_shape[0]:
        #         logger.error(f'Error in sparse labels data: min={np.min(self.labels)}, max={np.max(self.labels)}')
        #         sys.exit()
        #
        #     # convert sparse labels data to one hot vectors
        #     # some labels may not appear but still convert to match output of classifier
        #     self.labels = tf.keras.utils.to_categorical(self.labels, self.num_class)
        #     logger.debug(f'Labels\' shape after converting: {self.labels.shape[1:]}')

        # evaluate pixel value range of input images data
        if not validate_input_value(self.images, self.pixel_range):
            logger.error(f'Error in value range of input images data')
            sys.exit()

        logger.info(f'Load data complete!')



if __name__ == '__main__':
    config = AttackConfig('D:\Things\PyProject\AdvAttackCollection\config.ini')

    # example for change config.ini in runtime
    # config = ConfigObj('D:\\Things\\PyProject\\AdvAttackCollection\\config.ini')
    # temp = config[GENERAL_CONFIG][OUTPUT_FOLDER]
    # while(1):
    #     config.reload()
    #     if config[GENERAL_CONFIG][OUTPUT_FOLDER] != temp:
    #         temp = config[GENERAL_CONFIG][OUTPUT_FOLDER]
    #         print(temp)

