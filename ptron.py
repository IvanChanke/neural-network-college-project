"""
(C) Ivan Chanke 2020

Contains classes:
    Node - > Layer -> Network
The three constantly refer to one another and aren't supposed to work separately
For details on math model visit GitHub directory corresponding to the project
"""
import numpy as np
import pickle

bias_signal = 1

def load_model(file):
    """
    Loads network model from file
    """
    f = open(file, 'rb')
    model = pickle.load(f)
    f.close()

    return model

def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def s_df(x):

    return  sigmoid(x) * (1 - sigmoid(x))


class Node:
    """
    Stores:
        Its position in the network (position, layernum)
        A vector of weights going out of it.
        Local gradient
        Induced field
        Weights deltas vector
        Bias weight delta
        Its own bias weight; bias signal is always 1
    """
    def __init__(self, position):
        """
        position is a (node index, layer index) tuple
        """
        self.position = position[0]
        self.layernum = position[1]
        self.weights = None
        self.bias_weight = None
        self.local_grad = None
        self.induced_field = None
        self.weights_delta = None
        self.bias_weight_delta = 0

    def connect_node(self, layer):
        """
        Initializes a vector of weights for synapses this node has with the next layer
        Weights are initialized randomly
        """
        self.weights = (np.random.rand(layer.nnodes)) * 2 - 1
        self.weights_delta = np.zeros(layer.nnodes)

    def apply_deltas(self):

        self.weights += self.weights_delta
        self.weights_delta = np.zeros(self.weights_delta.shape)

    def modify_bias(self):

        self.bias_weight += self.bias_weight_delta
        self.bias_weight_delta = 0


class Layer:
    """
    Handles backpropagation and feeding forward.
    Stores:
        Layer index
        A list of nodes
        Number of nodes in the list (nnodes)
    """
    def __init__(self, nnodes, number):

        self.number = number
        self.nnodes = nnodes
        self.nodes = [Node((i, number)) for i in range(nnodes)]

    def connect_layer(self, other):
        """
        Connects layer self with the PREVIOUS layer other
        """
        for node in other.nodes:
            node.connect_node(self)

    def start_forward_propagation(self, other, vector):
        """
        Begins feeding forward
        This method works similarly to "propagate_forward" defined below
        The only difference is that the input vector doesn't go through the activation function
        """
        bias_weights_vector = np.array([node.bias_weight for node in other.nodes])

        weights_stack = [node.weights for node in self.nodes]
        weights_stack.append(bias_weights_vector)

        memory_matrix = np.vstack(weights_stack)
        memory_matrix = np.transpose(memory_matrix)

        for i in range(len(self.nodes)):
            self.nodes[i].induced_field = vector[i]

        vector = vector
        vector = np.append(vector, bias_signal)

        return memory_matrix.dot(vector)

    def propagate_forward(self, other, vector):
        """
        Maps input vector of layer i to input vector of layer i+1
        """
        bias_weights_vector = np.array([node.bias_weight for node in other.nodes]) # Composes a vector of bias weigths for layer other

        weights_stack = [node.weights for node in self.nodes]
        weights_stack.append(bias_weights_vector)

        memory_matrix = np.vstack(weights_stack) # Stacks up weight vectors of each node in self and bias weight vector of other
        memory_matrix = np.transpose(memory_matrix) # Each column is a weight vector; ncols = nnodes + 1, nrows + 1 = nsynapses;

        for i in range(len(self.nodes)):
            self.nodes[i].induced_field = vector[i] # Each node stores its induced field which is later used to compute local_grad

        vector = sigmoid(vector) # Vector goes through the activation function
        vector = np.append(vector, bias_signal) # Bias signal is added to a vector

        return memory_matrix.dot(vector) # Returns input vector for the next layer

    def start_backpropagation(self, e, learning_rate): # For output layer only; computes local_grad for each node in self; e - error vector
        """
        Computes a local gradient for each neuron in the output vector
        """
        for i in range(len(self.nodes)):
            self.nodes[i].local_grad = s_df(self.nodes[i].induced_field) * e[i]

            self.nodes[i].bias_weight_delta += learning_rate * self.nodes[i].local_grad * bias_signal

    def propagate_backward(self, other, learning_rate): # For layers except output only; connection scheme: self-other
        """
        Computes local gradients for nodes in self

        """
        for i in range(len(self.nodes)): # Local_grad for each node in layer self is computed
            lgv = np.array([node.local_grad for node in other.nodes]) # Local gradient vector for other
            d = s_df(self.nodes[i].induced_field) * np.dot(lgv, np.transpose(self.nodes[i].weights))
            self.nodes[i].local_grad = d
        """
        Computes deltas using local gradients
        """
        for node in self.nodes:
            deltas = []
            for i in range(len(node.weights)):
                delta = learning_rate * sigmoid(node.induced_field) * other.nodes[i].local_grad
                deltas.append(delta)

            bias_delta = learning_rate * node.local_grad * bias_signal

            node.weights_delta += np.array(deltas)
            node.bias_weight_delta += bias_delta


class Network:
    """
    Stores:
        Current signal
        A list of layers
        last mse
        Number of epoch trained
        Task it performs
    Initializing a network also initializes its layers and nodes in them
    """
    def __init__(self, structure, task): # Structure is a tuple of nnodes in each layer

        self.signal = None
        self.layers = []
        self.mse = None
        self.epochs = 0
        self.task = task

        for i in range(len(structure)): # Constructing layers
            self.layers.append(Layer(structure[i], i))

        for i in range(1, len(self.layers)):
            self.layers[i].connect_layer(self.layers[i - 1])

        for i in range(1, len(self.layers)): # Initializing bias weights
            for j in range(len(self.layers[i].nodes)):
                self.layers[i].nodes[j].bias_weight = (np.random.rand()) * 2 - 1

    def feed_forward(self, vector):
        """
        Maps input-output
        """
        self.signal = self.layers[0].start_forward_propagation(self.layers[1], vector)

        for i in range(1, len(self.layers) - 1): # Last layer does nothing, hence range(len - 1)
            self.signal = self.layers[i].propagate_forward(self.layers[i + 1], self.signal)

        for i in range(len(self.layers[-1].nodes)): # Stores last layer's nodes' induced fields
            self.layers[-1].nodes[i].induced_field = self.signal[i]

        return self.signal

    def feed_backward(self, error_vector, learning_rate):
        """
        Backpropagation
        Recursively computes deltas for weights; applies them
        """
        self.layers[-1].start_backpropagation(error_vector, learning_rate)

        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i - 1].propagate_backward(self.layers[i], learning_rate)


    def learning_iteration(self, batch_tuple, learning_rate):
        """
        One complete learning epoch
        batch_tuple is a tuple of tuples: ((in, desired_out), (...), ..., (...))
        """
        self.epochs += 1
        self.mse = 0
        for instance in batch_tuple:
            output = sigmoid(self.feed_forward(np.array(instance[0])))
            error_vector = instance[1] - output
            self.mse += sum(error_vector**2)

            self.feed_backward(error_vector, learning_rate)

            #print(instance[0], ':', output, "| e:", error_vector)
            #print((np.where(instance[1] == 1))[0], ':', np.around(output, decimals = 4))
            print(instance[0], ':', output)


        for node in self.layers[0].nodes:
            node.apply_deltas()

        for i in range(1, len(self.layers) - 1):
            for node in self.layers[i].nodes:
                node.apply_deltas()
                node.modify_bias()

        for node in self.layers[-1].nodes:
                node.modify_bias()

        print('MSE:', self.mse)
        print('----------------')

    def save_model(self, file):
        """
        Stores current model as a file
        """
        f = open(file, 'wb')
        pickle.dump(self, f)
        f.close()
