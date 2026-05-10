# 🧠 Feedforward Neural Network from Scratch

A small feedforward neural network built from scratch with NumPy.

I made this to understand what actually happens inside a neural network: linear layers, activation functions, softmax probabilities, cross-entropy loss, backpropagation, and stochastic gradient descent without using PyTorch, TensorFlow, or scikit-learn.

## 🛠 Tools

- Python
- NumPy
- argparse
- custom neural network layers
- stochastic gradient descent

## ✨ What it includes

- Single-hidden-layer neural network
- Custom `Linear` layer
- Sigmoid activation
- Softmax + cross-entropy loss
- Forward propagation
- Backpropagation
- SGD weight updates
- Train/validation loss tracking
- Prediction output files
- Error rate reporting

## 🚦 How to run

### 1. Clone the repo

```bash
git clone https://github.com/Danny625/feedforward-nn-numpy.git
cd feedforward-nn-numpy
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run training

```bash
python neuralnet.py train_sample.csv valid_sample.csv train_out.txt valid_out.txt metrics.txt 10 5 1 0.1
```

The arguments are:

```text
python neuralnet.py \
  <train_input> \
  <validation_input> \
  <train_out> \
  <validation_out> \
  <metrics_out> \
  <num_epoch> \
  <hidden_units> \
  <init_flag> \
  <learning_rate>
```

### Example

```bash
python neuralnet.py train_sample.csv valid_sample.csv train_out.txt valid_out.txt metrics.txt 10 5 1 0.1
```

This trains for 10 epochs with 5 hidden units, random initialization, and a learning rate of 0.1.

## 📄 Input format

The training and validation files should be CSV files where:

- the first column is the class label
- the remaining columns are numeric features

Example:

```text
0,0.1,0.2,0.3,0.5,0.7
1,0.4,0.6,0.8,0.2,0.3
0,0.2,0.1,0.9,0.6,0.4
```

## 📤 Outputs

The program writes three output files:

```text
train_out.txt      # predicted labels for training data
valid_out.txt      # predicted labels for validation data
metrics.txt        # train/validation losses and error rates
```

The metrics file includes cross-entropy loss after each epoch, followed by final train and validation error rates.

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

During training, the network processes one example at a time, computes the loss, backpropagates gradients through each layer, and updates weights using stochastic gradient descent.

## 📁 Project structure

```text
feedforward-nn-numpy/
├── README.md
├── neuralnet.py          # Main neural network implementation
├── requirements.txt      # Python dependencies
├── train_sample.csv      # Small sample training file
└── valid_sample.csv      # Small sample validation file
```

## 📚 What I learned

- How forward propagation works layer by layer
- How softmax and cross-entropy connect for classification
- How gradients flow backward through a network
- How stochastic gradient descent updates weights
- Why deep learning libraries are useful after understanding the basics

## 👤 Author

Danny Weng
