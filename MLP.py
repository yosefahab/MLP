# Pure arrays implementation of MLP
import numpy as np
import mnist_preprocessing

layers, neurons, rate, epochs, bias, function = 2, 10, 0.1, 2, 1, "Sigmoid"

weights = []
biases = []
activations = []
sigmas = []


def init_weights_activations_sigmas():
    global weights, activations, sigmas

    weights.append(np.random.rand(num_inputs, neurons))
    activations.append(np.zeros((num_inputs, neurons)))
    sigmas.append(np.zeros((neurons,)))

    for _ in range(layers - 1):
        weights.append(np.random.rand(neurons, neurons))
        activations.append(np.zeros((neurons,)))
        sigmas.append(np.zeros((neurons,)))

    weights.append(np.random.rand(neurons, num_outputs))
    activations.append(np.zeros((num_outputs,)))
    sigmas.append(np.zeros((num_outputs,)))

    weights = np.array(weights, dtype="object")
    activations = np.array(activations, dtype="object")
    sigmas = np.array(sigmas, dtype="object")


def init_biases():
    global biases
    biases = [bias for _ in range(layers + 1)]


def sigmoid(net):
    return 1.0 / (1 + np.exp(-net))


def sigmoid_derivative(ac):
    return ac * (1.0 - ac)


def tanh(net):
    return np.tanh(net)
    # return (np.exp(net) - np.exp(-net)) / (np.exp(net) + np.exp(-net))


def tanh_derivative(net):
    return 1 - (tanh(net) ** 2)


features_resh = None


def forward_propagation(features):
    global features_resh
    features_resh = activations[0]

    ac = features
    activations[0] = features

    for i, w in enumerate(weights):
        net = np.dot(ac, w) + biases[i]
        if function == "Sigmoid":
            ac = sigmoid(net)
        else:
            ac = tanh(net)

        activations[i + 1] = ac
        if i == len(weights) - 2:
            break
    return ac


def encoded_y(y_true):
    encoded_y = np.zeros((num_outputs, 1))
    encoded_y[y_true] = 1
    return encoded_y.reshape((-1,))


def back_propagation(y_true, y_pred):

    y_true = encoded_y(y_true)
    prev_sigma = y_true - y_pred
    f_net = activations[layers]

    if function == "Sigmoid":
        f_ = sigmoid_derivative(f_net)
    else:
        f_ = tanh_derivative(f_net)

    sigmas[layers] = f_ * prev_sigma

    # 2 1 0 -> sigmas
    #   ^
    for i in list(reversed(range(len(sigmas) - 1))):
        f_net = activations[i + 1]

        if function == "Sigmoid":
            f_ = sigmoid_derivative(f_net)
        else:
            f_ = tanh_derivative(f_net)

        sigmas[i] = np.dot(weights[i + 1], sigmas[i + 1]) * f_


def gradient_descent(rate, inp):
    for i in range(len(weights)):
        old_weight = weights[i]

        # for j in range(len(features_resh)):
        # for k in range(len(features_resh)):
        # weights[j] = old_weight[j] + (rate * sigmas[] * features_resh[j])

        if i == 0:
            weights[i] = old_weight + (np.dot(features_resh, sigmas[i]) * rate)
            print(weights[i].shape)
            exit(0)
        else:
            weights[i] = old_weight + (sigmas[i] * rate * activations[i])


def train(X_train, Y_train, epochs, rate):
    init_biases()
    init_weights_activations_sigmas()
    for _ in range(epochs):
        for j, curr in enumerate(X_train):
            y_true = Y_train[j]
            y_pred = forward_propagation(curr)
            back_propagation(y_true, y_pred)
            gradient_descent(rate, curr)


data = mnist_preprocessing.preprocessing(train_file="mnist_train.csv", test_file="mnist_test.csv")
X_train, x_test, Y_train, y_test = data.get_training_test_data()

num_inputs = X_train.shape[1]
num_outputs = 10

train(X_train, Y_train, epochs, rate)
