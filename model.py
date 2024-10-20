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


from model_utils import softmax, relu, relu_derivative, accuracy, cross_entropy_loss
from typing import List, Tuple
import numpy as np

class NeuralNet:
    def __init__(
        self,
        input_unit: int, 
        hidden_units: List[int], 
        output_unit: int) -> None:

        self.layer: int = len(hidden_units)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []

        # input to hidden weights & bias
        self.weights.append(np.random.randn(input_unit, hidden_units[0]) * np.sqrt(2 / input_unit))
        self.biases.append(np.zeros((1, hidden_units[0]))) 

        # hidden to hidden weights & biases
        for i in range(self.layer -1):
            self.weights.append(np.random.randn(hidden_units[i], hidden_units[i+1]) * np.sqrt(2 / hidden_units[i])) 
            self.biases.append(np.zeros((1, hidden_units[i+1]))) 

        # last hidden to output weight & bias
        self.weights.append(np.random.randn(hidden_units[len(hidden_units) -1], output_unit) * np.sqrt(2 / hidden_units[-1])) 
        self.biases.append(np.zeros((1, output_unit)))  

    def forward(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        
        self.hidden_output: List[np.ndarray] = []

        # first hidden layer output
        first_output = relu(np.dot(X, self.weights[0]) + self.biases[0])   
        self.hidden_output.append(first_output)

        # second and more hidden layers outputs in a loop
        for i in range(self.layer -1):
            output = relu(np.dot(self.hidden_output[i], self.weights[i+1]) + self.biases[i+1])   
            self.hidden_output.append(output)

        # final output of the model
        self.final_output = softmax(np.dot(self.hidden_output[-1], self.weights[-1]) + self.biases[-1])   
        return (self.final_output)

    def backward(
        self, 
        inputs: np.ndarray, 
        Y: np.ndarray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        
        m = Y.shape[0]
        dW: List[np.ndarray] = []
        dB: List[np.ndarray] = []
        error_list: List[np.ndarray] = []

        # Error and gradients in the output layer
        error_list.append(self.final_output - Y)                            
        dW.append((1/m) * np.dot(self.hidden_output[-1].T, error_list[0]))  
        dB.append((1/m) * np.sum(error_list[0], axis=0))

        # Error and gradients in the hidden layers
        for i in range(self.layer, 0, -1):
            error_list.append(np.dot(error_list[-1], self.weights[i].T) * relu_derivative(self.hidden_output[i-1]))
            dW.append((1/m) * np.dot(inputs.T if i == 1 else self.hidden_output[i-2].T, error_list[-1]))
            dB.append((1/m) * np.sum(error_list[-1], axis=0))
        return (dW, dB)

    def update_parameters(
        self, 
        dW: List[np.ndarray], 
        dB: List[np.ndarray], 
        learning_rate: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:

        for i in range(len(dW)):
            self.weights[i] -= learning_rate * dW[len(dW) -1 -i]        # -> reverse indexing  
            self.biases[i] -= learning_rate * dB[len(dB) -1 -i]
        return (self.weights, self.biases)

    def train(
        self, 
        X: np.ndarray, 
        Y: np.ndarray,
        epoch: int, 
        learning_rate: float
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float], List[float]]:
        
        loss_list: List[float] = []
        accuracy_list: List[float] = []

        for i in range(1, epoch +1):
            predictions = self.forward(X)
            dW, dB = self.backward(X, Y)
            weights, biases = self.update_parameters(dW, dB, learning_rate)

            loss = cross_entropy_loss(Y, predictions)
            loss_list.append(loss)
            acc = accuracy(Y, predictions)
            accuracy_list.append(acc)

            if i % 50 == 0:
                print(f"Epoch: {i} | Loss: {loss} | Accuracy: %{acc * 100:.2f}")  
        return (weights, biases, loss_list, accuracy_list)
    
