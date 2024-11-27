# Neural Netwrok Model Architecture

## Overview
This document provides an overview of the custom neural network model implemented from scracth.
The model consists of an input layer, multiple hidden layers, and an output layer, with each layer having specific parameters. The architecture leverages the ReLu activation function for hidden layers and the softmax activation for the output layer.

## Model Class: `NeuralNet`
The neural network is implemented as a Python class `NeuralNet` with methods to initialize the netwrok, perform feedforward computation, backpropagation, and update parameters.

![Ekran görüntüsü 2024-11-28 021712](https://github.com/user-attachments/assets/08e860b4-02c7-4a9d-998a-3f79876b69ba)

### Initialization
The netwrok is initialized with the following parameters:
- **Input Unit**: The number of input features.
- **Hidden Units**: A list specifying the number of neurons in each hidden layer.
- **Output Unit**: The number of neurons in the output layer.

### Key Components
- **Weights**: Initialized using the `He` initialization method, which scale the weights by the square root of 2 divided by the sum of the input and output units.
- **Biases**: Initialized to zero for all layers.

## Layer-wise Architecture
The model consists of:
- **Input Layer**: Accepts input features.
- **Hidden Layers**: Multiple hidden layers where each layer apllies the ReLu activation function.
- **Output Layer**: Applies the softmax activation function to produce the final output.

### Initialization of Parameters
The weights and biases for each layer are initialized as follows:
1. **Input to First Hidden Layer**:
   - Weights are initialized with random values, sclaed by `sqrt(2 / (input_units + first_hidden_units))`.
   - Biases are initialized to zero.
2. **Hidden Layers**:
   - Weights are initialized similarly to the input-hidden layer connection, using the size of adjacent layers.
   - Biases are initialized to zero.
3. **Last Hidden Layer to Output Layer**:
   - Weights and biases are initialized in the same way as above.

## Methods

### `initialize_parameters()`
This method initializes the weights and biases for all layers in the netwrok, based on the size of the input, hidden, and output layers. The weights are initialized using a random normal distribution, and biases are set to zero.

### `feedforward(X)`
This method computes the forward pass through the network. It calculates the activation for all layers as follows:
- **First Hidden Layer**: The input `X` is multiplied by the weights and biases are added, followed by applying the ReLu activation function.
- **Subsetquent Hidden Layers**: For each hidden layer, the previous layer’s output is used as input for the current layer.
- **Output Layer**: The final output is computed using the softmax activation.

### `backpropagation(inputs, Y)`
This method implements the backpropagation algorithm to compute the gradient of the weights and biases. It works by:
1. Calculating the error at the output layer.
2. Propagating the error backward through each hidden layer.
3. Calculating the gradients of the weights and biases at each layer.

### `update_parameters(dW, dB, alpha)`
This method updates the weights and biases using the gradients calculated by backpropagation. It applies gradient descent with a learning rate `alpha` to adjust the parameters.

## Activation Functions

### ReLu Activation Function
The ReLu (Rectified Linear Unit), is a non-linear activation function used in the hidden layers of neural networks. Its fundamental principle is to zero out negative values while allowing positive values to pass through unchanged. This characteristic helps the model learn quickly.
```python
def relu(x):
    return (np.maximum(0, x))
```
### Softmax Activation Function
The softmax function is a function used in multiple classification problems and converts the output values ​​of the model into probability values ​​between [0, 1]. and the sum of the probability values ​​is always 1.
```python
def softmax(x):
    exps = np.exp(x - np.max(x))
    return (exps / np.sum(exps, axis=1, keepdims=True))
```
