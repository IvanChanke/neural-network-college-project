import numpy as np
"""
ТЕКУЩИЕ ПАРАМЕТРЫ:
    ВЕСА ОТ -1 ДО 1
    СДВИГ ОТ -4 ДО 4

TODO: STOCHASTIC GRADIENT DESCENT. СДЕЛАТЬ ВИЗУАЛИЗАЦИЮ (?)

Найти инициализацию сдвига
Поэкспериментирповать с начальными значениями весов - установить их от -1 до 1
установить вес сдвига от -1 до 1
"""
"""
Старые комментарии на английском, которые писал "для себя", решил не удалять
Пусть будут.

Ниже определены сигмоида и её производная.
Затем классы нейрона -> слоя сети -> самой сети
Каждый нейрон в сети хранит вектор весов, из него исходящих, а также вес собственного сдвига.
Сигнал сдвига всегда равен 1.
"""

bias_signal = 1


def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def s_df(x):

    return  sigmoid(x) * (1 - sigmoid(x))


class Node:
    """
    Элементарная структурная единица сети.
    Хранит:
        координаты в сети
        вектор исходящих синаптических весов
        локальный градиент
        значение ошибки (для нейронов выходного слоя)
        индуцированное поле (сигнал выхода до активации сигмоидой)
        вектор поправки к собственным синаптическим весам
        значение поправки к собственному весу сдвига (сигнал сдвига всегда = 1)
    """
    def __init__(self, position):
        """
        position - кортеж вида (номер нейрона в слое, номер слоя)
        """
        self.position = position[0]
        self.layernum = position[1]
        self.weights = None
        self.bias_weight = None
        self.local_grad = None
        self.error = None
        self.induced_field = None
        self.weights_delta = None
        self.bias_weight_delta = 0

    def connect_node(self, layer): # Initializes a vector of weights for synapses this node has with the next layer
        """
        Соединяет нейрон с идущим следующим после собственного слоем
        Вызывается при инициализации сети
        Инициализирует случайный вектор весов, нулевой вектор поправки
        """
        self.weights = (np.random.rand(layer.nnodes)) * 2 - 1
        #print('Начальные веса', self.weights)
        self.weights_delta = np.zeros(layer.nnodes)

    def apply_deltas(self):
        """
        Применяет поправки к весам
        Обнуляет вектор поправок
        """
        self.weights += self.weights_delta
        #print('Веса', self.weights)
        self.weights_delta = np.zeros(self.weights_delta.shape)

    def modify_bias(self):
        """
        Применяет поправку к весу сдвига
        Обнуляет значение поправки
        """
        #print('Сдвиг', self.bias_weight)
        self.bias_weight += self.bias_weight_delta

        self.bias_weight_delta = 0



class Layer:
    """
    Слой нейронной сети
    Отвечает за распространение сигнала вперед-обратное распространение
    Хранит:
        собственный номер в сети
        число нейронов в себе
        список объектов-нейронов
    """
    def __init__(self, nnodes, number):
        """
        Инициализация слоя number. Инициализация nnodes нейронов слоя.
        """
        self.number = number
        self.nnodes = nnodes
        self.nodes = [Node((i, number)) for i in range(nnodes)]

    def connect_layer(self, other):
        """
        Соединяет слой self с ПРЕДЫДУЩИМ слоем other
        Вызывает метод соединения для каждого нейрона предыдущего слоя
        """
        for node in other.nodes:
            node.connect_node(self)

    def start_forward_propagation(self, other, vector):
        """
        Начинает распространение сигнала вперёд.
        Работа во многом аналогична методу propagate_forward, cм. ниже
        Единственное отличие - не пропускает через сигмоиду входной вектор
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
        Отображает входной вектор слоя self во входной вектор для СЛЕДУЮЩЕГО слоя other
        Также отвечает за применение сдвига в нейронах слоя other

        Составляет вектор весов сдвига для слоя other
        """
        bias_weights_vector = np.array([node.bias_weight for node in other.nodes])

        """
        Собирает матрицу весов из весов каждого собственного нейрона
        Добавляет вектор весов сдвига следующего слоя
        Транспонирует матрицу так, что каждая колонка - вектор весов
        """
        weights_stack = [node.weights for node in self.nodes]
        weights_stack.append(bias_weights_vector)
        memory_matrix = np.vstack(weights_stack) # Stacks up weight vectors of each node in the self and bias weight vector of other
        memory_matrix = np.transpose(memory_matrix) # Each column is a weight vector; ncols = nnodes + 1, nrows + 1 = nsynapses;
        """
        Каждый нейрон запоминает своё индуцированное поле
        Оно пригодится во время обратного распространения
        """
        for i in range(len(self.nodes)):
            self.nodes[i].induced_field = vector[i] # Each node stores its induced field later used to compute local_grad

        """
        Добавляем входному вектору сигнал от сдвига
        """
        vector = vector
        vector = np.append(vector, bias_signal) # Bias signal is added to a vector

        return sigmoid(memory_matrix.dot(vector)) # Returns input vector for the next layer

    def start_backpropagation(self, e, learning_rate): # For output layer only; computes local_grad for each node in self; e - error vector
        """
        Начинает обратное распространение
        Вычисляет локальные градиенты нейронов выходного слоя
        """
        for i in range(len(self.nodes)):
            self.nodes[i].local_grad = s_df(self.nodes[i].induced_field) * e[i]

            self.nodes[i].bias_weight_delta += learning_rate * self.nodes[i].local_grad * bias_signal



    def propagate_backward(self, other, learning_rate): # For layers except output only; connection scheme: self-other
        """
        Обратное распространение через один слой в направлении от other
        к self.
        """
        # This method computes local grads for nodes in self. Then it uses local grads in other to compute deltas.
        """
        Подсчёт локального градиента каждого нейрона слоя self
        """
        for i in range(len(self.nodes)): # Local_grad for each node in layer self is computed
            lgv = np.array([node.local_grad for node in other.nodes]) # Local gradient vector for other
            d = s_df(self.nodes[i].induced_field) * np.dot(lgv, np.transpose(self.nodes[i].weights))
            self.nodes[i].local_grad = d

        """
        Вычисляются поправки к весам и сдвигу
        """
        #print('Проход через слой {}'.format(self.number))
        for node in self.nodes:
            deltas = []
            for i in range(len(node.weights)):
                delta = learning_rate * sigmoid(node.induced_field) * other.nodes[i].local_grad
                deltas.append(delta)
            #print(node.bias_weight)
            bias_delta = learning_rate * node.local_grad * bias_signal
            """
            Вычисленные поправки учитываются в общей поправке эпохи
            """
            node.weights_delta += np.array(deltas)
            node.bias_weight_delta += bias_delta





class Network:
    """
    Класс сети
    """
    def __init__(self, structure): # Structure is a tuple of nnodes in each layer
        """
        Сеть хранит:
            Текущий сигнал, который через неё проходит
            Список слоёв
        При инициализации сети инициализируются слои, в каждом слое
        инициализируются нейроны. Слои соединяются. Инициализируются сдвиги.
        """
        self.signal = None # The vector currently going through the network
        self.layers = []

        for i in range(len(structure)): # Constructing layers
            self.layers.append(Layer(structure[i], i))

        for i in range(1, len(self.layers)): # Connecting layers; first "other" - layer[0], last "self" - layer[-1]
            self.layers[i].connect_layer(self.layers[i - 1])

        for i in range(1, len(self.layers)): # Initializing bias weights
            for j in range(len(self.layers[i].nodes)):
                self.layers[i].nodes[j].bias_weight = (np.random.rand()) * 2 - 1
                print('Слой {} нейрон {} сдвиг инициализирован'.format(i, j))



    def feed_forward(self, vector): # Maps input-output. Last layer is handled separately
        """
        Отображает входной вектор в выходной
        Метод ссылается на методы прогонки сигнала вперед класса "Слой"
        """
        self.signal = self.layers[0].start_forward_propagation(self.layers[1], vector)

        for i in range(1, len(self.layers) - 1): # Last layer does nothing, hence range(len - 1)
            self.signal = self.layers[i].propagate_forward(self.layers[i + 1], self.signal)

        for i in range(len(self.layers[-1].nodes)): # Stores last layer's nodes' induced fields
            self.layers[-1].nodes[i].induced_field = self.signal[i]

        return self.signal

    def feed_backward(self, error_vector, learning_rate): # Recursively computes deltas for weights; applies them
        """
        Обратное распространение ошибки
        Также ссылается на методы класса "Слой"
        """
        self.layers[-1].start_backpropagation(error_vector, learning_rate)

        for i in range(len(self.layers) - 1, 0, -1):
            self.layers[i - 1].propagate_backward(self.layers[i], learning_rate)


    def learning_iteration(self, batch_tuple, learning_rate): # batch_tuple is a tuple of tuples: ((in, desired_out), (...), ..., (...))
        """
        Одна эпоха обучения.
        Прогоняет вперёд каждый пример, состоящий из пары векторов вход-выход
        Для каждого примера прогоняет назад ошибку.
        В конце применяет поправки в весам и сдвигам.
        """
        mse = 0
        for instance in batch_tuple: # Each instance in the batch has its effect on the total error because it wants different outputs
            output = self.feed_forward(np.array(instance[0])) # Feeds forward the "in" vector
            error_vector = instance[1] - output
            mse += sum(error_vector**2)

            self.feed_backward(error_vector, learning_rate)

            #print(instance[0], ':', output, "| e:", error_vector) # Печатает полученный на выходе сигнал
            print(instance[0], ':', output)


        for node in self.layers[0].nodes:
            node.apply_deltas()
        for i in range(1, len(self.layers) - 1):
            for node in self.layers[i].nodes:
                #print('bias', node.bias_weight)
                node.apply_deltas()
                node.modify_bias()
        for node in self.layers[-1].nodes:
                #print('bias_last', node.bias_weight)
                node.modify_bias()

        #print('MSE:', mse)
