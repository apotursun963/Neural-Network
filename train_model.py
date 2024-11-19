"""
en:
This code is written to train a multilayer perceptron (MLP) model using the MNIST dataset. 
The data consists of handwritten digits of size 28x28 pixels, and the model learns to classify these digits accurately. 
The code first loads and normalizes the data, then creates and trains a model using the NeuralNet class. 
During training, loss and accuracy values are computed and visualized. 
Finally, after training is complete, the model's weights and bias values are saved to separate files.

tr:
Bu kod, MNIST veri kümesi ile çok katmanlı bir yapay sinir ağı (MLP) modelini eğitmek için yazılmıştır.
Veri, 28x28 piksel boyutundaki el yazısı rakamları içerir ve model, 
bu rakamları doğru bir şekilde sınıflandırmayı öğrenir. Kod, öncelikle verileri yükleyip 
normalize eder (belirli aralıklara ölçekler) ardından NeuralNet sınıfını kullanarak bir model oluşturur ve eğitir.
Eğitim sırasında kayıp ve doğruluk değerleri hesaplanır ve görselleştirilir.
Son olarak, eğitim tamamlandığında modelin ağırlıkları ve bias değerleri ayrı dosyalara kaydedilir.
"""

"""
Training results (Eğitim Sonuçları)
-----------------------------------
Epoch: 2000 | Loss: 0.05188147086640287 | Accuracy: %98.64
Training Duration of the Model: 23.92 minute
"""

from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.datasets import mnist
import matplotlib.pyplot as plt
from model import NeuralNet
import numpy as np
import time


# Data Pre-Processing
def load_and_preprocess_data():
    (x_train, y_train), (_ , _) = mnist.load_data()
    x_train = x_train.reshape(60000, 28*28) / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    print(f"x_train shape: {x_train.shape}\ny_train shape: {y_train.shape}")
    return (x_train, y_train)

# Building the MLP model from NeuralNet class
def train_model(x_train, y_train): 
    model = NeuralNet(
        input_unit=784,                         # 784 neurons which means 784 input features 
        hidden_units=[128, 64, 32],             # 3 hidden layers 128, 64, 23 in order
        output_unit=10                          # 10 neurons to predict digit in (0-9)
    )
    
    # HyperParameters
    learning_rate = 1e-1                        # 1e-1 -> 0.1
    epochs = 2000
    
    time_1 = time.time()
    weights, biases, loss_list, accuracy_list = model.train(x_train, y_train, epochs, learning_rate)
    time_2 = time.time()
    print(f"Training Duration of the Model: {(time_2 - time_1) / 60:.2f} minute")
    return (weights, biases, loss_list, accuracy_list)

# Ploting the accuracy and losses of the model
def plot_acc_loss(loss_list, accuracy_list):
    _, axs = plt.subplots(2, 1, figsize=(8,6))
    # accuracy
    axs[0].plot(accuracy_list, label="Accuracy", color="b")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid(True)
    # loss
    axs[0].plot(loss_list, label="Loss", color="r")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)
    
    plt.tight_layout()
    plt.show()

# Saving the parameters
def savae_parameters(weights, biases):
    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        np.save(f"Weight{idx+1}.npy", weight)
        np.save(f"Bias{idx+1}.npy", bias)

# Main function
def main():
    x_train, y_train = load_and_preprocess_data()
    weights, biases, loss_list, accuracy_list = train_model(x_train, y_train)
    plot_acc_loss(loss_list, accuracy_list)
    savae_parameters(weights, biases)

if __name__ == "__main__":
    main()
