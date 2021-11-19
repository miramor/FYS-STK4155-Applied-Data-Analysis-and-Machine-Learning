import numpy as np
import importlib
import functions
importlib.reload(functions)
from functions import *

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=[20, 20],
            n_categories=1,
            epochs=1000,
            batch_size=10,
            eta=0.01,
            lmbd=0,
            activation_function = sigmoid,
            output_activation = linear):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories
        self.n_layers = len(n_hidden_neurons)+2

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.z_h = [0]*(self.n_layers-2)
        self.a_h = [0]*(self.n_layers-2)

        self.create_biases_and_weights()

        self.act_func = activation_function
        self.output_activation = output_activation

        self.performance = [] #For storing performance after each epoch

    def create_biases_and_weights(self):
        self.hidden_weights = []
        self.hidden_bias = []
        self.hidden_weights.append(np.random.randn(self.n_features, self.n_hidden_neurons[0]))
        for i in range(1, self.n_layers-2):
            self.hidden_weights.append(np.random.randn(self.n_hidden_neurons[i-1], self.n_hidden_neurons[i]))

        for i in range(self.n_layers-2):
            self.hidden_bias.append(np.zeros(self.n_hidden_neurons[i]) + 0.01)

        self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01


    def feed_forward(self):
        # feed-forward for training

        self.z_h[0] = np.matmul(self.X_data, self.hidden_weights[0]) + self.hidden_bias[0]
        self.a_h[0] = self.act_func(self.z_h[0])
        for i in range(1, self.n_layers-2):
            self.z_h[i] = np.matmul(self.a_h[i-1], self.hidden_weights[i]) + self.hidden_bias[i]
            self.a_h[i] = self.act_func(self.z_h[i])

        self.z_o = np.matmul(self.a_h[-1], self.output_weights) + self.output_bias
        self.z_o = self.output_activation(self.z_o)
        #exp_term = np.exp(self.z_o)
        #self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output

        z_h_out = np.matmul(X, self.hidden_weights[0]) + self.hidden_bias[0]
        a_h_out = self.act_func(z_h_out)
        for i in range(1, self.n_layers-2):
            z_h_out = np.matmul(a_h_out, self.hidden_weights[i]) + self.hidden_bias[i]
            a_h_out = self.act_func(z_h_out)

        z_o = np.matmul(a_h_out, self.output_weights) + self.output_bias

        return self.output_activation(z_o)

    def backpropagation(self):
        error_output = (self.z_o - self.Y_data.reshape(self.z_o.shape))# * self.act_func(self.z_o, derivative = True)

        error_hidden = [0]*(self.n_layers-2)
        error_hidden[0] = np.matmul(error_output, self.output_weights.T) *  self.act_func(self.z_h[-1], derivative = True)
        for i in range(1, self.n_layers-2):
            error_hidden[i] = np.matmul(error_hidden[i-1], self.hidden_weights[-i].T) * self.act_func(self.z_h[-i-1], derivative = True)


        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden[-1]) + self.lmbd * self.hidden_weights[0]
        self.hidden_weights[0] -= self.eta * self.hidden_weights_gradient
        self.hidden_bias_gradient = np.sum(error_hidden[-1], axis=0)
        self.hidden_bias[0] -= self.eta * self.hidden_bias_gradient

        for i in range(1, self.n_layers-2):
            self.hidden_weights_gradient = np.matmul(self.a_h[i-1].T, error_hidden[-i-1]) + self.lmbd * self.hidden_weights[i]
            self.hidden_bias_gradient = np.sum(error_hidden[-i-1], axis=0)

            self.hidden_weights[i] -= self.eta * self.hidden_weights_gradient
            self.hidden_bias[i] -= self.eta * self.hidden_bias_gradient


        self.output_weights_gradient = np.matmul(self.a_h[-1].T, error_output) + self.lmbd * self.output_weights
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient


    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def predict_reg(self, X):
        return self.feed_forward_out(X)

    def plot_mse(self, save = False, fn = "mse_train.pdf"):
        epochs = [i for i in range(len(self.performance))]
        plt.plot(epochs, self.performance)
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.title("Mean squared error as function of epoch")
        if save:
            plt.savefig(fn, dpi = 400, bbox_inches='tight')
        plt.show()


    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                #pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False)

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

            y_tilde = self.predict_reg(self.X_data_full)
            self.performance.append(mse(y_tilde[:,0], self.Y_data_full))
