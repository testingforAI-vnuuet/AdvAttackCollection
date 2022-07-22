"""
Created by khadm on 1/6/2022
Feature: 
"""
import tensorflow as tf
import numpy as np


class BlackBoxClassifier:
    # only return the prediction
    def __init__(self, core_classifier: tf.keras.models.Model):
        self.core_classifier = core_classifier
        self.name = core_classifier.name

    def predict(self, inputs: np.ndarray):
        # simulate query to the cloud
        return self.core_classifier.predict(inputs)

    def evaluate(self, train, label):
        return self.core_classifier.evaluate(train, label)
