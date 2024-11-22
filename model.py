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
        """
        Initializing the neural network with input, hidden and output layers.
        """
        self.hidden_lyrs = len(hidden_units)
        self.input_unit = input_unit
        self.hidden_units = hidden_units
        self.output_unit = output_unit
        self.__initialize_parameters()

    def __initialize_parameters(self):
        """
        Initializing the weights and biases for all layers using Xavier initialization.
        """
        self.weights = []; self.biases = []

        # Initializing weights and biases for the input to first hidden layer
        self.weights.append(
            np.random.randn(self.input_unit, self.hidden_units[0]) * np.sqrt(2 / (self.input_unit + self.hidden_units[0]))
        )
        self.biases.append(np.zeros((1, self.hidden_units[0]))) 

        # Initializing weights and biases for hidden layers
        for i in range(self.hidden_lyrs -1):
            self.weights.append(
                np.random.randn(self.hidden_units[i], self.hidden_units[i+1]) * np.sqrt(2 / (self.hidden_units[i] + self.hidden_units[i+1]))
            ) 
            self.biases.append(np.zeros((1, self.hidden_units[i+1]))) 

        # Initializing weights and biases for the last hidden layer to output layer
        self.weights.append(
            np.random.randn(self.hidden_units[len(self.hidden_units) -1], self.output_unit) * np.sqrt(2 / (self.hidden_units[-1] + self.output_unit))
        ) 
        self.biases.append(np.zeros((1, self.output_unit)))

    def feedforward(self, X):
        """
        feed-forward stage from input layer to output layer
        """
        self.lyr_outputs = []

        # first hidden layer output
        first_output = relu(np.dot(X, self.weights[0]) + self.biases[0])   
        self.lyr_outputs.append(first_output)

        # Subsequent hidden layers output
        for i in range(self.hidden_lyrs -1):
            output = relu(np.dot(self.lyr_outputs[i], self.weights[i+1]) + self.biases[i+1])   
            self.lyr_outputs.append(output)

        # final output layer
        self.final_output = softmax(np.dot(self.lyr_outputs[-1], self.weights[-1]) + self.biases[-1])   
        return (self.final_output)

    def backpropagation(self, inputs, Y):
        """
        back-propagation stage from the output layer to the input layer to compute the gradients of weights and biases
        """
        m = Y.shape[0]
        error_list = []
        dW = []; dB = []

        # Computing error and gradients at the output layer
        error_list.append(self.final_output - Y)                            
        dW.append((1/m) * np.dot(self.lyr_outputs[-1].T, error_list[0]))  
        dB.append((1/m) * np.sum(error_list[0], axis=0))

        # Computing error and gradients for hidden layers
        for i in range(self.hidden_lyrs):
            error_list.append(
                np.dot(error_list[-1], self.weights[len(self.weights) -i -1].T) * relu_derivative(self.lyr_outputs[len(self.lyr_outputs) -i -1])
            )
            dW.append((1/m) * np.dot(inputs.T if i == (self.hidden_lyrs - 1) else self.lyr_outputs[len(self.lyr_outputs) -i -2].T, error_list[-1]))
            dB.append((1/m) * np.sum(error_list[-1], axis=0))
        return (dW[::-1], dB[::-1])

    def __update_parameters(self, dW, dB, alpha):
        """
        Updating weights and biases using gradient descent.
        """
        for idx, (w, b, dw, db) in enumerate(zip(self.weights,self.biases, dW, dB)):
            self.weights[idx] = w - alpha * dw
            self.biases[idx] = b - alpha * db
        return (self.weights, self.biases)

    def train(self, X, Y, epoch, learning_rate):
        """
        Training the neural netwrok over a given number of epochs
        """
        loss_list = []
        accuracy_list = []
        for i in range(1, epoch +1):
            predictions = self.feedforward(X)
            dW, dB = self.backpropagation(X, Y)
            weights, biases = self.__update_parameters(dW, dB, learning_rate)

            loss = cross_entropy_loss(Y, predictions)
            loss_list.append(loss)
            acc = accuracy(Y, predictions)
            accuracy_list.append(acc)
            
            # print the progress every 50 epochs
            if i % 50 == 0:
                print(f"Epoch: {i} | Loss: {loss} | Accuracy: %{acc * 100:.2f}")  
        return (weights, biases, loss_list, accuracy_list)

    def evulate(self, X, Y):
        """
        Evulating the model on the test data.
        """
        pred = self.feedforward(X)
        loss = cross_entropy_loss(Y, pred)
        acc = accuracy(Y, pred)
        return (f"Test acc: {acc * 100:.2f} | Test loss: {loss}")

