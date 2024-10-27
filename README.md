# Description
This project is designed to create an artificial neural network model from scratch. It attempts to recoding the multilayer perceptron (MLP), one of the deep learning models.
The model is evaluated on test data, calculating the accuracy and loss, and visualizing predictions.

## Usage
```python
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.datasets import mnist
from model import NeuralNet

# Data Pre-Processing
(x_train, y_train), (_ , _) = mnist.load_data()
x_train = x_train.reshape(60000, 28*28) / 255.0
y_train = to_categorical(y_train, num_classes=10)

# Building Model
model = NeuralNet(
    input_unit=784,
    hidden_units=[128, 256, 128],
    output_unit=10
)

# Training Model
model.train(x_train, y_train, epoch=2000, learning_rate=0.01)
```

## accuracy and loss
![acc_loss](https://github.com/user-attachments/assets/21e3ae52-a8e8-4d16-b583-86a4d8965daf)
