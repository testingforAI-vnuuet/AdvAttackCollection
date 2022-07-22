"""
Created by khadm on 1/20/2022
Feature: 
"""
import tensorflow as tf


class AutoencoderConfig:
    # in progress
    def __init__(self, epochs: int, batch_size: int, print_result_every_epochs: int, learning_rate: float,
                 autoencoder_model: tf.keras.models.Model, autoencoder_model_name: str):
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_result_every_epochs = print_result_every_epochs
        self.learning_rate = learning_rate
        self.autoencoder_model = autoencoder_model
        self.autoencoder_model_name = autoencoder_model_name
        self.loss_function = 'mse'

    def get_shared_name(self):
        str_lr = str(self.learning_rate).replace('.', ',')
        return f'epochs={self.epochs}_batchsize={self.batch_size}_learningrate={str_lr}_loss={self.loss_function}'
