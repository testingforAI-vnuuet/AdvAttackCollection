"""
Created by khadm on 2/16/2022
Feature:
    Support custom layers.
    Reference: https://www.tensorflow.org/guide/keras/save_and_serialize#registering_the_custom_object
"""
import tensorflow as tf


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(CustomLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

    def get_config(self):
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config


# Change here: serialize custom objects as dict.
user_custom_objects = {'CustomLayer': CustomLayer}
