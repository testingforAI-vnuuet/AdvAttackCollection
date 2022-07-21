#  dataset
MNIST_IMG_ROWS = 28
MNIST_IMG_COLS = 28
MNIST_IMG_CHL = 1
MNIST_NUM_CLASSES = 10

NONAME = 'noname'
MNIST_DATA_NAME = 'mnist'
float32_type = 'float32'
optimizer_adam = 'adam'
optimizer_SGD = 'SGD'

MODEL_H5_EXTENSION = '.h5'
DATA_NP_EXTENSION = '.npy'

ALLOWED_MODEL_EXTENSIONS = ['h5', 'pb', 'ckpt']
ALLOWED_IMAGE_EXTENSIONS = ['png', 'jpg']

shared_exit_msg = 'Please check out configuration!'
shared_incorrect_para_msg = '{param} is not correct.'


class CONFIG_TXT:
    CLASSIFIER = 'CLASSIFIER'
    attack_type = 'attack_type'
    targetClassifierPath = 'targetClassifierPath'
    substituteClassifierPath = 'substituteClassifierPath'

    DATA = 'DATA'
    useDataFolder = 'useDataFolder'
    dataFolder = 'dataFolder'
    trainingDataPath = 'trainingDataPath'
    labelDataPath = 'labelDataPath'

    ATTACK = 'ATTACK'
    originalLabel = 'originalLabel'
    targetLabel = 'targetLabel'
    useOptimizePhase = 'use_optimize_phase'
    recoverSpeed = 'recoverSpeed'
    weight = 'beta'
    numberDataToAttack = 'numberDataToAttack'
    numberDataToTrainAutoencoder = 'numberDataToTrainAutoencoder'
    maxNumberAdvsToOptimize = 'maxNumberAdvsToOptimize'
    epochToOptimize = 'epoch_to_optimize'
    batchToOptimize = 'batch_to_optimize'
    quality_loss = 'quality_loss'
    L2thresholdtoStopAttack = 'L2_threshold_to_stop_attack'
    L0thresholdtoStopAttack = 'L0_threshold_to_stop_attack'
    SSIMthresholdtoStopAttack = 'SSIM_threshold_to_stop_attack'

    AUTOENCODER_TRAINING = 'AUTOENCODER_TRAINING'
    autoencoder_model_path = 'autoencoder_model_path'
    epochs = 'epochs'
    batch_size = 'batch_size'
    print_result_every_epochs = 'print_result_every_epochs'
    learning_rate = 'learning_rate'


class DEFEND_CONFIG_TXT:
    DEFENDER = 'DEFENDER'
    use_defender = 'use_defender'
    DEFENDER_AE_CLEANER = 'DEFENDER_AE_CLEANER'
    target_classifier_path = 'target_classifier_path'
    target_classifier = 'target_classifier'
    autoencoder_cleaner_path = 'autoencoder_cleaner_path'
    autoencoder_cleaner = 'autoencoder_cleaner'
    noise_volume = 'noise_volume'
    epochs = 'epochs'
    batch_size = 'batch_size'
    learning_rate = 'learning_rate'
    val_ratio = 'val_ratio'
