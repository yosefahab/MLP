from tkinter import *
from tkinter import messagebox
import train

class GUI:
    def __init__(self, window):
        self.hidden_layers = 0
        self.neurons = 0
        self.learning_rate = 0
        self.epochs = 0
        self.bias = 0
        self.activation_function = ""
        self.dataset = ""
        self.datasets = ["Penguins", "MNIST"]
        self.activation_functions = ["Sigmoid", "Hyperbolic Tangent Sigmoid", "Softmax"]
        self.window = window
        self.initialize_window()

    def store_parameters(self, layers, neurons, rate, epochs, bias, function, dataset):
        self.hidden_layers = int(layers.get())
        self.neurons = int(neurons.get())
        self.learning_rate = float(rate.get())
        self.epochs = int(epochs.get())
        self.bias = bias.get()
        self.dataset = dataset.get()
        self.activation_function = function.get()

    def check(self, layers, neurons, rate, epochs):
        validation_dict = {
            (layers.get(), ''): "Hidden Layers Can not be empty",
            (neurons.get(), ''): "Number of neurons Can not be empty",
            (rate.get(), ''): "The Learning Rate Can not be empty",
            (epochs.get(), ''): "The Epochs Can not be empty",
        }

        for k in validation_dict:
            if k[0] == k[1]:
                messagebox.showwarning("WARNING...", validation_dict[k])
                return False
        return True
    
    def initialize_window(self):
        self.window.title("Task 3")

        w, h = (600, 650)
        ws = self.window.winfo_screenwidth()
        hs = self.window.winfo_screenheight()

        x = (ws / 2) - (w / 2)
        y = (hs / 2) - (h / 2)
        self.window.geometry("%dx%d+%d+%d" % (w, h, x, y))
        label_1 = Label(self.window, text="\nEnter number of hidden layer: ",
            justify="left", font=("Helvetica", 15), pady=5)
        label_1.pack()
        hidden_layer = Entry(self.window)
        hidden_layer.focus_set()
        hidden_layer.pack(pady=5)

        label_2 = Label(self.window, text="\nEnter number of neurons in each hidden layer: ",
            justify="left", font=("Helvetica", 15), pady=5)
        label_2.pack()
        neurons = Entry(self.window)
        neurons.pack(pady=5)

        label_3 = Label(self.window, text="\nEnter Learning Rate: ",
            justify="left", font=("Helvetica", 15), pady=5)
        label_3.pack()
        learning_rate = Entry(self.window)
        learning_rate.pack(pady=5)

        label_4 = Label(self.window, text="\nEnter Number of Epochs: ",
            justify="left", font=("Helvetica", 15), pady=5)
        label_4.pack()
        epochs = Entry(self.window)
        epochs.pack(pady=5)

        bias = IntVar()
        c1 = Checkbutton(self.window, text='bias', variable=bias, onvalue=1, offvalue=0)
        c1.pack(pady=5)

        label_5 = Label(self.window, text="Choose Activation Function: ",
            justify="left", font=("Helvetica", 15), pady=5)
        label_5.pack()

        activation_function = StringVar(self.window)
        activation_function.set(self.activation_functions[0])
        drop_down_menu1 = OptionMenu(self.window, activation_function, *self.activation_functions)
        drop_down_menu1.pack(pady=5)

        label_6 = Label(self.window, text="Choose Dataset:", justify="left",
            font=("Helvetica", 15), pady=5)
        label_6.pack()

        dataset = StringVar(self.window)
        dataset.set(self.datasets[0])
        drop_down_menu2 = OptionMenu(self.window, dataset, *self.datasets)
        drop_down_menu2.pack(pady=5)

        b = Button(self.window, text='Start', width=10, 
            command=lambda: self.start(hidden_layer, neurons, learning_rate,
                epochs, bias, activation_function, dataset))
        b.pack(pady=5)

    def start(self, layers, neurons, rate, epochs, bias, function, dataset):
        if self.check(layers, neurons, rate, epochs):
            self.store_parameters(layers, neurons, rate, epochs, bias, function, dataset)
            train.start(self.hidden_layers, self.neurons, self.learning_rate,
                self.epochs, self.bias, self.activation_function, self.dataset)