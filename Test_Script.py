import Network_code as net
import numpy as np


a = net.Network((2, 2, 2, 1)) #Имеем сеть 2-2-1. Сейчас запрогана задача ИЛИ

i = 0
while True:
    i += 1
    batch = (
        (np.array([0, 0]), np.array([0])),
        #(np.array([0, 1]), np.array([1])),
        #(np.array([1, 0]), np.array([1])),
        #(np.array([1, 1]), np.array([0])),
        # (np.array([1, 0, 0]), np.array([1])),
        # (np.array([1, 0, 1]), np.array([1])),
        # (np.array([1, 1, 0]), np.array([1])),
        # (np.array([1, 1, 1]), np.array([0])),
    )
    a.learning_iteration(batch, 0.7) #Вот это число - скорость обучения
    batch = (
        (np.array([0, 1]), np.array([1])),
    )
    a.learning_iteration(batch, 0.7) #Вот это число - скорость обучения
    batch = (
        (np.array([1, 0]), np.array([1])),
    )
    a.learning_iteration(batch, 0.7) #Вот это число - скорость обучения
    batch = (
        (np.array([1, 1]), np.array([0])),

    )
    a.learning_iteration(batch, 0.7) #Вот это число - скорость обучения
    print('Epoch {}'.format(i))
    print('----------------------')
