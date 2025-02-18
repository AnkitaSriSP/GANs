import streamlit as st
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import cv2

def plot_image(image_array):
    image_array = np.array(image_array).reshape((2,2)) * 255  # Scale values to 0-255
    image_array = image_array.astype(np.uint8)
    image = cv2.resize(image_array, (100, 100), interpolation=cv2.INTER_NEAREST)  # Resize for better visibility
    st.image(image)

def generate_random_image():
    return [np.random.random(), np.random.random(), np.random.random(), np.random.random()]

def sigmoid(x):
    return np.exp(x)/(1.0+np.exp(x))

class Discriminator():
    def __init__(self, learning_rate):
        #self.weights = np.array([0.0 for i in range(4)])
        #self.bias = 0.0
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.bias = np.random.normal()
        self.lr=learning_rate
    
    def forward(self, x):
        # Forward pass
        return sigmoid(np.dot(x, self.weights) + self.bias)
    
    def error_from_image(self, image):
        prediction = self.forward(image)
        # We want the prediction to be 1, so the error is -log(prediction)
        return -np.log(prediction)
    
    def derivatives_from_image(self, image):
        prediction = self.forward(image)
        derivatives_weights = -image * (1-prediction)
        derivative_bias = -(1-prediction)
        return derivatives_weights, derivative_bias
    
    def update_from_image(self, x):
        ders = self.derivatives_from_image(x)
        self.weights -= self.lr * ders[0]
        self.bias -= self.lr * ders[1]

    def error_from_noise(self, noise):
        prediction = self.forward(noise)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        return -np.log(1-prediction)
    
    def derivatives_from_noise(self, noise):
        prediction = self.forward(noise)
        derivatives_weights = noise * prediction
        derivative_bias = prediction
        return derivatives_weights, derivative_bias
    
    def update_from_noise(self, noise):
        ders = self.derivatives_from_noise(noise)
        self.weights -= self.lr * ders[0]
        self.bias -= self.lr * ders[1]

class Generator():
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.biases = np.array([np.random.normal() for i in range(4)])
        self.lr=learning_rate

    def forward(self, z):
        # Forward pass
        return sigmoid(z * self.weights + self.biases)

    def error(self, z, discriminator):
        x = self.forward(z)
        y = discriminator.forward(x)
        return -np.log(y)

    def derivatives(self, z, discriminator):
        discriminator_weights = discriminator.weights
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1-y) * discriminator_weights * x *(1-x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias

    def update(self, z, discriminator):
        derivs = self.derivatives(z, discriminator)
        self.weights -= self.lr * derivs[0]
        self.biases -= self.lr * derivs[1]
