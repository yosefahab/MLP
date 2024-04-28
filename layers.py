import numpy as np
import abc

class Layer():
    def __init__(self, bias, num_input, num_output, activation, activation_):
        if (self.__class__ is not OutputLayer):
            self.weights = np.random.normal(0, 1 / num_input, (num_input, num_output))
            self.bias = np.random.normal(0, 1 / num_input,
                (1, num_output)) if bias == True else np.zeros((1, num_output))
            self.next_sigma = None

        self.activation = activation
        self.activation_ = activation_

    @abc.abstractmethod
    def forward_propagation(self, input_x):
        return

    @abc.abstractmethod
    def backward_propagation(self, next_sigma):
        return

    @abc.abstractmethod
    def update_weights(self, learning_rate):
        raise NotImplementedError('Abstract layer does not implement methods')

class InputLayer(Layer):
    def __init__(self, bias, num_input, num_output, activation, activation_):
        # self.weights = np.random.rand(num_input, num_output)
        # self.bias = np.random.rand(1, num_output) if bias == True else np.zeros((1, num_output))

        super().__init__(bias, num_input, num_output, activation, activation_)
        self.X = None

    def forward_propagation(self, input_x):
        self.X = np.expand_dims(input_x, axis=1)
        next_net = np.dot(self.X.T, self.weights) + self.bias
        return next_net

    def backward_propagation(self, next_sigma):
        self.next_sigma = next_sigma
        return self.next_sigma

    def update_weights(self, learning_rate):
        self.weights += learning_rate * np.dot(self.X, self.next_sigma)
        self.bias += learning_rate * self.next_sigma

class HiddenLayer(Layer):
    def __init__(self, bias, num_input, num_output, activation, activation_):
        # weights and bias that connect current layer (i) to next layer (i+1)
        # self.weights = np.random.rand(num_input, num_output)
        # self.bias = np.random.rand(1, num_output) if bias == True else np.zeros((1, num_output))

        super().__init__(bias, num_input, num_output, activation, activation_)
        self.net = None  # input net
        self.next_net = None  # net of layer i+1
        self.activ = None  # activated net
        self.sigma = None  # error

    def forward_propagation(self, input_x):
        self.net = input_x
        self.activ = self.activation(self.net)
        next_net = np.dot(self.activ, self.weights) + self.bias
        return next_net

    def backward_propagation(self, next_sigma):
        self.next_sigma = next_sigma
        self.sigma = np.dot(self.next_sigma, self.weights.T) * self.activation_(self.net)
        return self.sigma

    def update_weights(self, learning_rate):
        self.weights += learning_rate * np.dot(self.activ.T, self.next_sigma)
        self.bias += learning_rate * self.next_sigma

class OutputLayer(Layer):
    def __init__(self, activation, activation_):
        super().__init__(None, None, None, activation, activation_)
        self.activ = None
        self.activation = activation
        self.activation_ = activation_

    def forward_propagation(self, input_x):
        self.net = input_x
        self.activ = self.activation(input_x)
        return self.activ

    def backward_propagation(self, next_sigma):
        return next_sigma

    def loss(self, y, y_pred):
        return y - y_pred

    def sigma(self, y):
        loss = self.loss(y, self.activ)
        return self.activation_(self.net) * loss

    def update_weights(self, _):
        return