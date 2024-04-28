# OOP implementation of MLP
import numpy as np

class MLP:
    def __init__(self, num_outputs):
        self.layers = []
        self.num_outputs = num_outputs

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        predictions = []

        for x in input_data:
            # output is input for every layer
            output = x
            for layer in self.layers:
                output = layer.forward_propagation(output)

            # final output : y_pred
            predictions.append(self.decode_label(output))

        return predictions

    def encode_label(self, y):
        encoded_y = np.zeros((self.num_outputs, 1))
        encoded_y[int(y)] = 1
        return encoded_y.reshape((-1,))

    def decode_label(self, y_pred):
        decoded_y = np.argmax(y_pred)
        return decoded_y

    def fit(self, x_train, y_train, epochs, learning_rate):
        for _ in range(epochs):

            for (x, y) in zip(x_train, y_train):
                y = self.encode_label(y)
                # output is input for every layer
                output = x
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                # final output = y_pred

                # backward propagation
                output_layer = self.layers[len(self.layers) - 1]
                prev_sigma = output_layer.sigma(y)
                for layer in reversed(self.layers):
                    prev_sigma = layer.backward_propagation(prev_sigma)

                for layer in self.layers:
                    layer.update_weights(learning_rate)
