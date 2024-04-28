from MLP_ import MLP
from layers import *
from loader import MNISTLoader, PenguinLoader
from tabulate import tabulate

def sigmoid(net):
    return 1.0 / (1.0 + np.exp(-net))

def sigmoid_derivative(net):
    ac = sigmoid(net)
    return ac * (1.0 - ac)

def tanh(net):
    return np.tanh(net)

def tanh_derivative(net):
    return 1.0 - (tanh(net) ** 2)

def softmax(net):
    exps = np.exp(net - net.max())
    return exps / np.sum(exps, axis=0)

def softmax_derivative(net):
    return net / np.sum(net, axis=0) * (1 - net / np.sum(net, axis=0))

################### Initialise MLP
def start(layers, neurons, learning_rate, epochs, bias, activation_function, dataset):
    loader = PenguinLoader() if (dataset == "Penguins") else MNISTLoader()
    x_train, y_train = loader.load_train()

    num_inputs, num_outputs = loader.get_inputs_outputs()

    function_dict = {
        "Sigmoid": (sigmoid, sigmoid_derivative),
        "Hyperbolic Tangent Sigmoid": (tanh, tanh_derivative),
        "Softmax": (softmax, softmax_derivative)
    }

    activation, activation_ = function_dict[activation_function]
    mlp = MLP(num_outputs)

    # Input layer
    mlp.add_layer(InputLayer(bias, num_inputs, neurons, activation, activation_))

    # Hidden layers
    for _ in range(layers - 1):
        mlp.add_layer(HiddenLayer(bias, neurons, neurons, activation, activation_))

    # final hidden layer
    mlp.add_layer(HiddenLayer(bias, neurons, num_outputs, activation, activation_))

    # Output layer
    mlp.add_layer(OutputLayer(activation, activation_))
    ###################


    ################### Start training
    print("Started Training...")
    mlp.fit(x_train, y_train, epochs, learning_rate)
    print("Training Finished.")
    ###################

    print()

    ################## Test network
    print("Testing Network...")
    x_test, y_test = loader.load_test()
    y_pred = mlp.predict(x_test)

    matrix = np.zeros((num_outputs, num_outputs))
    for i, j in zip(y_test, y_pred):
        matrix[int(i)][int(j)] += 1

    #print(matrix)
    print(tabulate(matrix, headers=[i for i in range(num_outputs)], showindex=True))
    print(np.sum(y_pred == y_test) / len(y_test))
    print("Finished.")
    ##################