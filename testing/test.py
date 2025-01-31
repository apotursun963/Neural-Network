"""
en:
This code provides an application for testing a multilayer neural network model using the MNIST dataset.
During the testing phase, pre-trained weights and biases are loaded.
In the data preprocessing step, the test data is loaded and normalized.
The model makes predictions on the given test images, and the number of correct and incorrect predictions is calculated.
Finally, the accuracy rate and loss rate of the model are computed and visualized alongside examples of the test images.

tr:
Bu kod, MNIST veri kümesi ile çok katmanlı yapay sinir ağı modelini test eder.
Modelin test aşamasında, önceden eğitilmiş "ağırlıklar" ve "biaslar" yüklenir.
Veri ön işleme adımında test verileri yüklenir ve normalleştirilir.
Model, verilen test görüntülerini tahmin eder, doğru ve yanlış tahmin sayıları hesaplanır.
Son olarak, modelin doğruluk oranı ve kayıp oranı hesaplanır ve test görüntülerinin örnekleri ile birlikte görselleştirilir.
"""

"""
Testing results (Test Sonuçları)
--------------------------------
Testing images: 10.000 images 
number of false: 283 | number of correct: 9717
Accuracy rate for 10.000 image is: 97.17 %
Loss rate for 10.000 image is: 2.83 %
"""


from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.datasets import mnist
import matplotlib.pyplot as plt 
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sources.utils import relu, softmax


def load_and_preprocess_data():
    (_ , _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.reshape(10000, 28*28) / 255.0 
    y_test = to_categorical(y_test, num_classes=10)
    print(f"x_test: {x_test.shape}\ny_test: {y_test.shape}")
    return (x_test, y_test)

def load_param(directory="checkpoints"):
    W_lst = [np.load(os.path.join(directory, f"Weight{i + 1}.npy")) for i in range(4)]
    B_lst = [np.load(os.path.join(directory, f"Bias{i + 1}.npy")) for i in range(4)]
    return (W_lst, B_lst)

def predict_single_sample(idx, X, W_lst, B_lst):
    X = X[idx]
    output = X 
    hidden_outputs = []
    for i in range(len(W_lst) -1):
        output = relu(np.dot(output, W_lst[i]) + B_lst[i])
        hidden_outputs.append(output)
    final_output = softmax(np.dot(hidden_outputs[-1], W_lst[-1]) + B_lst[-1])
    return (final_output)

def test_mdl(x_test, y_test, W_lst, B_lst):
    n_correct = 0; n_false = 0
    for idx in range(len(x_test)):
        mdl_pred = np.argmax(predict_single_sample(idx, x_test, W_lst, B_lst))
        n_correct += (np.argmax(y_test[idx]) == mdl_pred)             # condition return True or False 
        n_false += (np.argmax(y_test[idx]) != mdl_pred)               # if True +1 else False +0
    return (n_correct, n_false)

def acc_loss(n_true, n_false, x_test):
    print(f"number of false: {n_false} | number of correct: {n_true}")
    print(f"Accuracy rate for {len(x_test)} image is: {(n_true / (n_true + n_false)) * 100:.2f} %") 
    print(f"Loss rate for {len(x_test)} image is: {(n_false / (n_true + n_false)) * 100:.2f} %") 

def visualize_predictions(x_test, y_test, W_lst, B_lst):
    plt.figure(figsize=(8, 6))
    for idx in range(len(x_test)):
        prediction = predict_single_sample(idx, x_test, W_lst, B_lst)
        plt.subplot(3, 5, idx +1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
        plt.title(f"true: {np.argmax(y_test[idx])}\npred: {np.argmax(prediction)}")
        plt.axis("off")
    plt.show()

def main():
    x_test, y_test = load_and_preprocess_data()
    W_ls, B_ls = load_param()
    n_correct, n_false = test_mdl(x_test, y_test, W_ls, B_ls)
    acc_loss(n_correct, n_false, x_test)
    visualize_predictions(x_test[: 15], y_test[: 15], W_ls, B_ls)

if __name__ == "__main__":
    main()
