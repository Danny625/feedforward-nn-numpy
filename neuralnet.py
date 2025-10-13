"""
Feedforward Neural Network from Scratch
---------------------------------------
Implements a single-hidden-layer neural network using only NumPy.
Includes custom modules for Linear layers, Sigmoid activation,
and Softmax Cross-Entropy loss with stochastic gradient descent training.

Author: Danny Weng
Date: March 2024
"""

import numpy as np
import argparse
from typing import Callable, List, Tuple

# Command-line argument setup for training, validation, and output configuration
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    Parses command-line arguments and loads train/test data.
    Returns data arrays, output file paths, and training hyperparameters.
    """

    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    Shuffles the training data deterministically based on epoch number.
    Ensures reproducible random order for stochastic gradient descent.
    """

    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """Initialize weights with zeros."""

    return np.zeros(shape=shape)


def random_init(shape):
    """Initialize weights uniformly between [-0.1, 0.1]."""

    M, D = shape
    np.random.seed(M * D)  # Don't change this line
    return np.random.uniform(low=-0.1, high=0.1, size=shape)


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Softmax activation and Cross-Entropy loss layer.
        Provides forward and backward methods for loss computation and gradients.
        """
        # Compute softmax probabilities
        z = z - np.max(z)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # Compute cross-entropy loss
        return -np.log(y_hat[y] + 1e-15)

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        # Forward pass combining softmax and cross-entropy
        y_hat = self._softmax(z)
        loss = self._cross_entropy(y, y_hat)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        probabilities, we are directly computing gradient w.r.t. the softmax
        input.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # Compute gradient of loss with respect to softmax input
        grad = y_hat.copy()
        grad[y] -= 1
        return grad


class Sigmoid:
    def __init__(self):
        """
        Sigmoid activation function.
        Caches the output for use in backward gradient computation.
        """
        # Initialize cached output for backward pass
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        # Compute sigmoid output and cache for backward pass
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        # Compute gradient of loss with respect to sigmoid input
        return dz * self.out * (1 - self.out)


# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Fully connected linear layer (with bias folding).
        Includes forward pass, gradient computation, and SGD weight updates.
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate
        # Initialize weight matrix and bias
        self.w = weight_init_fn((output_size, input_size + 1))
        self.w[:, 0] = 0.0
        self.dw = np.zeros_like(self.w)
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (output_size,)
        """
        # Compute layer output and cache bias-augmented input for backprop
        x_bias = np.insert(x, 0, 1)
        self.cache['x_bias'] = x_bias
        z = np.dot(self.w, x_bias)
        return z

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear
        
        Note that this function should set self.dw
            (gradient of loss with respect to weights)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.
        """
        # Compute gradients for weights and input
        x_bias = self.cache['x_bias']
        self.dw = np.outer(dz, x_bias)
        dx_bias = np.dot(self.w.T, dz)
        dx = dx_bias[1:]
        return dx

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        # Update weights using SGD
        self.w -= self.lr * self.dw
        self.dw = np.zeros_like(self.w)


class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Single-hidden-layer feedforward neural network.
        Implements forward propagation, backpropagation, and SGD optimization.
        """

        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize network layers and activation/loss modules
        self.linear1 = Linear(input_size, hidden_size, weight_init_fn, learning_rate)
        self.sigmoid = Sigmoid()
        self.linear2 = Linear(hidden_size, output_size, weight_init_fn, learning_rate)
        self.softmax_ce = SoftMaxCrossEntropy()

    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation. 
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # Forward pass through all layers
        a = self.linear1.forward(x)
        z = self.sigmoid.forward(a)
        b = self.linear2.forward(z)
        y_hat, loss = self.softmax_ce.forward(b, y)
        self.cache = {'a': a, 'z': z, 'b': b}
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # Backward pass through all layers
        db = self.softmax_ce.backward(y, y_hat)
        dz = self.linear2.backward(db)
        da = self.sigmoid.backward(dz)
        _ = self.linear1.backward(da)

    def step(self):
        """
        Apply SGD update to weights.
        """
        # Apply SGD updates to all layers
        self.linear1.step()
        self.linear2.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # Compute mean cross-entropy loss across all samples
        total_loss = 0.0
        N = X.shape[0]
        for i in range(N):
            x = X[i]
            label = int(y[i])
            _, loss = self.forward(x, label)
            total_loss += loss
        return total_loss / N

    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # Train network for multiple epochs
        train_losses = []
        test_losses = []
        N = X_tr.shape[0]
        for epoch in range(n_epochs):
            # Shuffle training data at the beginning of each epoch
            X_shuffled, y_shuffled = shuffle(X_tr, y_tr, epoch)
            for i in range(N):
                x = X_shuffled[i]
                label = int(y_shuffled[i])
                y_hat, _ = self.forward(x, label)
                self.backward(label, y_hat)
                self.step()
            train_loss = self.compute_loss(X_tr, y_tr)
            test_loss = self.compute_loss(X_test, y_test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            print(f"epoch={epoch+1} crossentropy(train): {train_loss}")
            print(f"epoch={epoch+1} crossentropy(validation): {test_loss}")
        return train_losses, test_losses

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # Compute predictions and overall error rate
        N = X.shape[0]
        predictions = []
        num_errors = 0
        for i in range(N):
            x = X[i]
            label = int(y[i])
            a = self.linear1.forward(x)
            z = self.sigmoid.forward(a)
            b = self.linear2.forward(z)
            y_hat = self.softmax_ce._softmax(b)
            pred = int(np.argmax(y_hat))
            predictions.append(pred)
            if pred != label:
                num_errors += 1
        error_rate = num_errors / N
        return np.array(predictions), error_rate


if __name__ == "__main__":
    args = parser.parse_args()
    # Initialize and train neural network

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    # Call args2data to get all data + argument values
    # Define class labels
    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)

    nn = NN(
        input_size=X_tr.shape[-1],
        hidden_size=n_hid,
        output_size=len(labels),
        weight_init_fn=zero_init if init_flag == 2 else random_init,
        learning_rate=lr
    )

    # Train and evaluate model
    train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # test model and get predicted labels and errors
    train_labels, train_error_rate = nn.test(X_tr, y_tr)
    test_labels, test_error_rate = nn.test(X_test, y_test)

    # Write predicted label and error into file
    # Note that this assumes train_losses and test_losses are lists of floats
    # containing the per-epoch loss values.
    with open(out_tr, "w") as f:
        for label in train_labels:
            f.write(str(label) + "\n")
    with open(out_te, "w") as f:
        for label in test_labels:
            f.write(str(label) + "\n")
    with open(out_metrics, "w") as f:
        for i in range(len(train_losses)):
            cur_epoch = i + 1
            cur_tr_loss = train_losses[i]
            cur_te_loss = test_losses[i]
            f.write("epoch={} crossentropy(train): {}\n".format(
                cur_epoch, cur_tr_loss))
            f.write("epoch={} crossentropy(validation): {}\n".format(
                cur_epoch, cur_te_loss))
        f.write("error(train): {}\n".format(train_error_rate))
        f.write("error(validation): {}\n".format(test_error_rate))
