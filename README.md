# 🧠 Feedforward Neural Network from Scratch

A single-hidden-layer neural network implemented entirely with NumPy.

I built this to understand the mechanics behind neural networks without hiding everything behind PyTorch or TensorFlow. The project implements forward propagation, backpropagation, softmax cross-entropy loss, and stochastic gradient descent from scratch.

## 🛠 Tools

- Python
- NumPy
- argparse

## ✨ What it includes

- Custom `Linear` layer
- Sigmoid activation
- Softmax + cross-entropy loss
- Forward propagation
- Backpropagation
- Stochastic gradient descent
- Zero or random weight initialization
- Train/validation loss tracking
- Prediction and metrics output files

## 🚦 Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python neuralnet.py train_sample.csv valid_sample.csv train_out.txt valid_out.txt metrics.txt 10 5 1 0.1
```

This trains the network for 10 epochs with 5 hidden units, random initialization, and a learning rate of 0.1.

## 📄 Input format

The input CSV files should have the label in the first column and numeric features in the remaining columns.

Example:

```text
0,0.1,0.2,0.3,0.5,0.7
1,0.4,0.6,0.8,0.2,0.3
0,0.2,0.1,0.9,0.6,0.4
```

## ⚙️ Arguments

```text
python neuralnet.py <train_input> <validation_input> <train_out> <validation_out> <metrics_out> <num_epoch> <hidden_units> <init_flag> <learning_rate>
```

| Argument | Meaning |
|---|---|
| `train_input` | training CSV file |
| `validation_input` | validation CSV file |
| `train_out` | file for training predictions |
| `validation_out` | file for validation predictions |
| `metrics_out` | file for losses and error rates |
| `num_epoch` | number of training epochs |
| `hidden_units` | number of hidden layer units |
| `init_flag` | `1` = random init, `2` = zero init |
| `learning_rate` | SGD learning rate |

## 📤 Output

The model writes:

```text
train_out.txt      # predicted labels for training data
valid_out.txt      # predicted labels for validation data
metrics.txt        # train/validation losses and error rates
```

Example metrics output:

```text
epoch=1 crossentropy(train): 2.302
epoch=1 crossentropy(validation): 2.298
...
error(train): 0.124
error(validation): 0.132
```

## 🤖 How it works

The model uses a simple feedforward architecture:

```text
input features
   ↓
Linear layer
   ↓
Sigmoid activation
   ↓
Linear layer
   ↓
Softmax + cross-entropy loss
```

During training, each example is passed through the network, the loss gradient is backpropagated layer by layer, and the weights are updated using SGD.

## 📁 Project structure

```text
feedforward-nn-numpy/
├── README.md
├── neuralnet.py          # Core neural network implementation
├── requirements.txt      # NumPy dependency
├── train_sample.csv      # Small sample training file
├── valid_sample.csv      # Small sample validation file
└── LICENSE
```

## 📚 What I learned

- How gradients flow through a neural network
- How softmax and cross-entropy work together
- How backpropagation updates weights
- Why initialization affects training
- Why deep learning libraries are useful after understanding the basics

## 👤 Author

Danny Weng
