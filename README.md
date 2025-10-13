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
'''

---
Arguments:

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

---
Example Output:

epoch=1 crossentropy(train): 2.302
epoch=1 crossentropy(validation): 2.298
epoch=2 crossentropy(train): 2.289
epoch=2 crossentropy(validation): 2.285
...
error(train): 0.124
error(validation): 0.132

---
Project Structure:

feedforward-nn-numpy/
│
├── neuralnet.py              # Core neural network implementation
├── README.md                 # Project overview and usage
├── requirements.txt          # Dependencies (NumPy)
├── train_sample.csv          # Example synthetic dataset (optional)
├── valid_sample.csv          # Example synthetic dataset (optional)
├── metrics_sample.txt        # Example output log (optional)
└── LICENSE                   # MIT License (optional)

---

Dependencies:

Python 3.9+
NumPy ≥ 1.20

Install dependencies:
pip install -r requirements.txt

---

Notes

This project was inspired by coursework from Carnegie Mellon University’s 10-301: Introduction to Machine Learning, but this version has been rewritten and documented for public educational use.

No CMU-provided materials, datasets, or autograder files are included.

---

Learning Outcomes

Through this implementation, I reinforced key machine learning fundamentals:

Deriving and coding the backpropagation algorithm manually

Understanding how activation functions and loss interact

Implementing SGD without frameworks

Visualizing training and validation losses over epochs

---

Empirical Summary:
Random initialization converged faster than zero initialization, and increasing hidden units improved training stability. Validation loss decreased over epochs, confirming correct gradient propagation.

---

Author:

Danny Weng
Carnegie Mellon University
