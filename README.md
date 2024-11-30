Multi-Layer Perceptron
----------------------

![ann](https://github.com/user-attachments/assets/eedfc466-bb47-40ee-bde4-595746552ffa)

This project is designed to create an `artificial neural network` model from scratch. It attempts to recoding the multilayer perceptron (MLP), one of the deep learning models.
The model is evaluated on test data, calculating the accuracy and loss, and visualizing predictions. for more information >>> [Model-Architecture](docs/model_architecture.md)

## Usage
```python
from model import NeuralNet

# Building Model
model = NeuralNet(
    input_unit=784,
    hidden_units=[128, 256, 128],
    output_unit=10
)

# Training Model
model.train(x_train, y_train, epoch=2000, learning_rate=0.01)

# Testing Model
model.evulate(x_test, y_test)
```

## Accuracy && Loss
![acc_loss](https://github.com/user-attachments/assets/21e3ae52-a8e8-4d16-b583-86a4d8965daf)
