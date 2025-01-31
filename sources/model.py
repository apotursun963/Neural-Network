"""
en:
This code contains a MLP neural netwrok class (NeuralNet).
The netwrok includes input, hidden, and output layers.
It makes predictions using forward propagation, computes errors with backpropagation,
and updates parameters (weights and biases) to train the model.

tr:
Bu kod, Çok katmanlı (MLP) yapay sinir ağı sınıfını (NeuralNet) tanımlar.
Sinir ağı giriş katmanı, gizli katmanlar ve çıkış katmanını içerir.
ileri besleme ile tahminler yapar, geri yayılım ile hataları hesaplar ve
parametreleri (ağırlıklar ve biaslar) güncelleyerek modeli eğitir.
"""

from utils import softmax, relu, relu_derivative, accuracy, cross_entropy_loss
import numpy as np

class NeuralNet:
    def __init__(self, input_unit, hidden_units, output_unit):
        self.hidden_lyrs = len(hidden_units)
        self.input_unit = input_unit
        self.hidden_units = hidden_units
        self.output_unit = output_unit
        self.initialize_parameters()

    def initialize_parameters(self):
        self.weights = []; self.biases = []

        # Initializing weights and biases for the input to first hidden layer
        self.weights.append(
            np.random.randn(self.input_unit, self.hidden_units[0]) * np.sqrt(2 / (self.input_unit + self.hidden_units[0]))      # He initialization
        )
        self.biases.append(np.zeros((1, self.hidden_units[0]))) 

        # Initializing weights and biases for hidden layers
        for i in range(self.hidden_lyrs -1):
            self.weights.append(
                np.random.randn(self.hidden_units[i], self.hidden_units[i + 1]) * np.sqrt(2 / (self.hidden_units[i] + self.hidden_units[i + 1]))
            ) 
            self.biases.append(np.zeros((1, self.hidden_units[i + 1]))) 

        # Initializing weights and biases for the last hidden layer to output layer
        self.weights.append(
            np.random.randn(self.hidden_units[len(self.hidden_units) -1], self.output_unit) * np.sqrt(2 / (self.hidden_units[-1] + self.output_unit))
        ) 
        self.biases.append(np.zeros((1, self.output_unit)))

    def feedforward(self, X):
        self.lyr_outputs = []

        # first hidden layer output
        first_output = relu(np.dot(X, self.weights[0]) + self.biases[0])   
        self.lyr_outputs.append(first_output)

        # Subsequent hidden layers output
        for i in range(self.hidden_lyrs -1):
            output = relu(np.dot(self.lyr_outputs[i], self.weights[i + 1]) + self.biases[i + 1])   
            self.lyr_outputs.append(output)

        # final output layer
        self.final_output = softmax(np.dot(self.lyr_outputs[-1], self.weights[-1]) + self.biases[-1])   
        return (self.final_output)

    def backpropagation(self, inputs, Y):
        m = Y.shape[0]
        error_list = []
        derivative_W = []
        derivative_B = []

        # Computing error and gradients at the output layer
        error_list.append(self.final_output - Y)                            
        derivative_W.append((1/m) * np.dot(self.lyr_outputs[-1].T, error_list[0]))  
        derivative_B.append((1/m) * np.sum(error_list[0], axis=0))

        # Computing error and gradients for hidden layers
        for i in range(self.hidden_lyrs):
            error_list.append(
                np.dot(error_list[-1], self.weights[len(self.weights) -i -1].T) * relu_derivative(self.lyr_outputs[len(self.lyr_outputs) -i -1])
            )
            derivative_W.append((1/m) * np.dot(inputs.T if i == (self.hidden_lyrs - 1) else self.lyr_outputs[len(self.lyr_outputs) -i -2].T, error_list[-1]))
            derivative_B.append((1/m) * np.sum(error_list[-1], axis=0))
        return (derivative_W[::-1], derivative_B[::-1])

    def update_parameters(self, dW, dB, alpha):
        for idx, (w, b, dw, db) in enumerate(zip(self.weights,self.biases, dW, dB)):
            self.weights[idx] = w - alpha * dw
            self.biases[idx] = b - alpha * db
        return (self.weights, self.biases)

    def train(self, X, Y, epoch, learning_rate):
        loss_list = []
        accuracy_list = []
        for i in range(1, epoch +1):
            predictions = self.feedforward(X)
            dW, dB = self.backpropagation(X, Y)
            weights, biases = self.update_parameters(dW, dB, learning_rate)

            loss = cross_entropy_loss(Y, predictions)
            loss_list.append(loss)
            acc = accuracy(Y, predictions)
            accuracy_list.append(acc)
            
            # print the progress every 50 epochs
            if i % 50 == 0:
                print(f"Epoch: {i} | Loss: {loss} | Accuracy: %{acc * 100:.2f}")  
        return (weights, biases, loss_list, accuracy_list)

    def evulate(self, X, Y):
        pred = self.feedforward(X)
        loss = cross_entropy_loss(Y, pred)
        acc = accuracy(Y, pred)
        return (f"Test acc: {acc * 100:.2f} | Test loss: {loss}")
