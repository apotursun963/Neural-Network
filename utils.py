"""
en
This file contains the function necessary of the model
Activation functions: softmax(), relu(), relu_derivative()
Metrics: accuracy()
Losses: cross_entropy_loss()

tr
Bu dosyaya model için gerekli olan fonksiyonlar yazılmıştır.
Aktivasyon fonksyionları: softmax(), relu(), relu_derivative()
Metrik(performans ölçütleri): accuracy()
Kayıp: cross_entropy_loss()
"""

import numpy as np

def softmax(x):
    """
    en: The softmax function is a function used in multiple classification problems and converts the output values 
    ​​of the model into probability values ​​between [0, 1]. and the sum of the probability values ​​is always 1.

    tr: softmax fonksiyonu çoklu sınıflandırma problemlerinde kullanılan ve Modelin çıktı değerlerini 
    [0, 1] arasında olasılık değerlerine dönüştüren bir foknsiyondur. 
    ve olasılık değerlerin toplamı her zaman 1 olur.
    """
    exps = np.exp(x - np.max(x)) 
    return (exps / np.sum(exps, axis=1, keepdims=True))

def relu(x):
    """
    en: relu, is a non-linear activation function used in the hidden layers of neural networks. 
    Its fundamental principle is to zero out negative values while allowing positive values to pass through unchanged. 
    This characteristic helps the model learn quickly.

    tr: relu, Sinir ağa'daki gizli katmanlarda kullanılan ve doğrusal olmayan (non-linear) bir fonksiyondur.
    temel mantığı, negatif değerleri sıfırlaması ve pozitif değerleri olduğu gibi bırakmasıdır. buda modelin hızlı öğrenmesini sağlar.
    """
    return (np.maximum(0, x))

def relu_derivative(x):
    """
    en: The relu derivative represents the gradient of the ReLU activation function. 
    For negative inputs, the derivative is 0, meaning the neurons cannot be updated and may become "dead." 
    For positive inputs, the derivative is 1, allowing the neurons to be updated normally. 
    In summary, the ReLU derivative determines how neurons are updated during the learning process.

    tr: relu türevi, ReLU aktivasyon fonksiyonunun gradyanını (değişim oranını) ifade eder. Negatif girişler için türev 0'dır, 
    bu durumda nöronlar güncellenemez ve "ölü" hale gelebilir. 
    Pozitif girişler için türev 1'dir, bu da nöronların normal şekilde güncellenmesini sağlar. 
    Özetle, ReLU türevi, modelin öğrenme sürecinde nöronların nasıl güncelleneceğini belirler.
    """
    return (np.where(x > 0, 1, 0))


def accuracy(Y, predictions):
    """
    en: The accuracy is a metric used to evaluate the performance (correctness) of the model. 
    It calculates how accurately the model predicts by comparing the actual labels(values) with the labels predicted by the model.
    Working Method: Actual values, Predicted values, Average calculation

    tr: Modelin performansını (Doğuruluğunu) değerlendirmek için kullanılan bir metrik'tir. Gerçek etiketler (değerler)
    ile modelin tahmin ettiği etiketleri karşılaştırarak modelin ne kadar doğru tahmin ettiğini hesaplar.
    çalışma biçimi; Gerçek değerler, tahmin edilen değerler ve Ortalama hesaplama 
    """
    true = np.argmax(Y, axis=1)
    mdl_prediction = np.argmax(predictions, axis=1)
    return (np.mean(mdl_prediction == true))

def cross_entropy_loss(Y, predictions):
    """
    en: It is a loss function that measures the difference between the values predicted by the model and the actual values. 
    It also evaluates how poorly or well the model performs. The lower the loss value, the better the model's performance.

    tr: Modelin tahmin ettiği değerler ile gerçek değerler arasındaki farkı ölçen bir kayıp (loss) fonksiyonudur.
    ayrıca modelin ne kadar kötü veya ne kadar iyi performans gösterdiğni değerlendirir.
    Kayıp değeri ne kadar düşükse model o kadar iyi performans gösteriyor demektir.
    """
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    return (-np.mean(np.sum(Y * np.log(predictions), axis=1)))


