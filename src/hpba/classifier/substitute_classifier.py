"""
Created by khadm on 1/15/2022
Feature: 
"""
from src.hpba.classifier.black_box_classifier import BlackBoxClassifier
import tensorflow as tf
import numpy as np
from src.utils.attack_logger import AttackLogger

logger = AttackLogger.get_logger()

class SubstituteClassifier:
    MAX_TRAINING = 4  # >= 1 (could be modified based on experience)
    EPSILON = 0.01  # > 0 (could be modified based on experience)
    NUM_DATA_TO_AUGMENT = 1000  # > 0 (could be modified based on experience)

    def __init__(self, trainX: np.ndarray, trainY: np.array, target_classifier: BlackBoxClassifier, input_shape: list,
                 num_classes: int, substitute_cls_path: str):
        self.trainX = trainX
        self.trainY = trainY
        self.target_classifier = target_classifier
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.substitute_cls_path = substitute_cls_path

        self.classifier = self.define_architecture()

    def define_architecture(self):
        #  model from the paper: https://arxiv.org/pdf/1511.04508.pdf (table I, II)

        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(200, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(200, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(units=self.num_classes, activation='relu')(x)
        last_activation = 'softmax' if self.num_classes > 1 else 'sigmoid'
        outputs = tf.keras.layers.Activation(activation=last_activation)(x)

        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        if self.num_classes == 1:
            loss = tf.keras.losses.binary_crossentropy
        elif len(self.trainY.shape) == 1:
            loss = tf.keras.losses.sparse_categorical_crossentropy
        else:
            loss = tf.keras.losses.categorical_crossentropy
        model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
        return model

    def train(self):
        if self.classifier is None:
            logger.debug(f'Substitute model is not defined. We will use a predefined model.')
            self.classifier = self.define_architecture()
        # train_x, test_x, train_y, test_y = train_test_split(self.trainX, self.trainY, test_size=0.33, random_state=42)
        train_x = np.array(self.trainX)
        train_y = np.array(self.trainY)
        # get the accuracy from target classifier
        test_x_target_predict = self.target_classifier.predict(train_x)
        if self.num_classes > 1:
            test_x_target_predict = np.argmax(test_x_target_predict, axis=-1)
            test_y_label = np.argmax(train_y, axis=-1)
        else:
            test_x_target_predict = np.round(test_x_target_predict).reshape(-1)
            test_y_label = np.array(train_y)
        matches = len(np.where(test_y_label == test_x_target_predict)[0])
        target_cls_acc = float(matches) / len(test_x_target_predict)

        logger.debug('The substitute model is training')
        for iter in range(self.MAX_TRAINING):
            logger.debug(f'[{iter + 1}/{self.MAX_TRAINING}] Training iteration')
            logger.debug(f'Size of the training set = {len(train_x)}')
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, mode='max',
                                                              min_delta=1e-3, restore_best_weights=True)
            callbacks = [early_stopping]
            self.classifier.fit(x=train_x, y=train_y, callbacks=callbacks,
                                batch_size=512,
                                epochs=50)
            sub_acc = self.classifier.evaluate(train_x, train_y)[1]
            logger.debug(f'Accuracy of substitute model = {sub_acc}')
            if abs(sub_acc - target_cls_acc) < 0.05:
                break

            # randomly select num_data_to_augment (default: 1000) samples to Memory Limit Exceeded Error
            logger.debug(
                'The accuracy of substitute model is not good. We will expand the training set and then retrain!')
            curr_length = len(train_x)
            p = np.random.permutation(curr_length)
            train_x = train_x[p]
            train_y = train_y[p]

            chosen_train_x = np.array(train_x[p[:self.NUM_DATA_TO_AUGMENT]])
            chosen_train_y = np.array(train_y[p[:self.NUM_DATA_TO_AUGMENT]])

            train_x_target_predict = self.target_classifier.predict(chosen_train_x)
            train_x_tensor = tf.convert_to_tensor(chosen_train_x)

            if self.num_classes > 1:
                train_x_target_predict = np.argmax(train_x_target_predict, axis=-1)
                train_x_target_predict_categorical = tf.keras.utils.to_categorical(train_x_target_predict,
                                                                                   self.num_classes)
                with tf.GradientTape() as tape:
                    tape.watch(train_x_tensor)
                    preds = self.classifier(train_x_tensor)
                    one_hot_mask = tf.one_hot(train_x_target_predict, preds.shape[1], on_value=True, off_value=False,
                                              dtype=tf.bool)
                    loss = preds[one_hot_mask]

                grad = tf.sign(tape.gradient(loss, train_x_tensor)).numpy()
                train_x_tmp = chosen_train_x[:self.NUM_DATA_TO_AUGMENT] + self.EPSILON * grad

            else:
                train_x_target_predict = np.round(train_x_target_predict).astype(np.float32)
                train_x_target_predict_categorical = np.round(train_x_target_predict).reshape(-1)
                with tf.GradientTape() as tape:
                    tape.watch(train_x_tensor)
                    preds = self.classifier(train_x_tensor)

                grad = tf.sign(tape.gradient(preds, train_x_tensor)).numpy()
                preds_npy = preds.numpy()
                preds_npy_sign = np.sign(preds_npy - 0.5)
                preds_npy_sign = preds_npy_sign.reshape(len(preds_npy_sign), 1, 1, 1)
                train_x_tmp = chosen_train_x[:self.NUM_DATA_TO_AUGMENT] + self.EPSILON * grad * preds_npy_sign

            train_x = np.concatenate((train_x, train_x_tmp))
            train_y = np.concatenate((train_y, train_x_target_predict_categorical))

        logger.debug('The substitute model is trained successfully')
        self.classifier.save(self.substitute_cls_path)
