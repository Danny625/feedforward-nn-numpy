# Feedforward Neural Network (NumPy)

A single-hidden-layer neural network implemented entirely in **NumPy**, built to understand the mechanics of forward propagation, backpropagation, and stochastic gradient descent (SGD) from first principles.

Developed by **Danny Weng** as part of an educational project exploring the foundations of machine learning model design without external deep learning libraries.

---

## 🧠 Overview

This project implements a fully connected feedforward neural network with:
- One hidden layer using **Sigmoid** activation
- **Softmax with Cross-Entropy loss** for classification
- Manual **forward and backward propagation**
- Training via **Stochastic Gradient Descent (SGD)**

Everything—weight initialization, loss computation, and gradient updates—is implemented from scratch using only NumPy.

---

## ⚙️ Features

- **Zero or Random Weight Initialization**  
- **Fully Custom Layers:** `Linear`, `Sigmoid`, and `SoftmaxCrossEntropy`
- **Automatic Gradient Updates** using SGD  
- **Numerically Stable Softmax Implementation**  
- **Command-Line Interface** for configurable training and evaluation  
- **Deterministic Shuffling** for reproducibility per epoch

---

## 🚀 Usage

To train and evaluate the model, run:

```bash
python neuralnet.py train.csv valid.csv train_out.txt valid_out.txt metrics.txt 50 100 1 0.01

---

| Argument        | Description                                        |
| --------------- | -------------------------------------------------- |
| `train.csv`     | Path to the training dataset                       |
| `valid.csv`     | Path to the validation dataset                     |
| `train_out.txt` | File to save predictions on the training set       |
| `valid_out.txt` | File to save predictions on the validation set     |
| `metrics.txt`   | File to store training and validation losses       |
| `num_epoch`     | Number of epochs to train (e.g. 50)                |
| `hidden_units`  | Number of hidden layer units (e.g. 100)            |
| `init_flag`     | 1 = random initialization, 2 = zero initialization |
| `learning_rate` | SGD learning rate (e.g. 0.01)                      |

