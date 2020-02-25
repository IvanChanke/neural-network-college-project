import network_code_eng as net
import numpy as np
"""
XOR CURRENTLY PROGRAMMED IN
"""

a = net.Network((3, 2, 1))
a.save_model('test')
a = net.load_model('test')
i = 0
while i < 2000:
    i += 1
    batch = (
        (np.array([0, 0, 0]), np.array([0])),
        (np.array([0, 0, 1]), np.array([1])),
        (np.array([0, 1, 0]), np.array([1])),
        (np.array([0, 1, 1]), np.array([0])),
        (np.array([1, 0, 0]), np.array([1])),
        (np.array([1, 0, 1]), np.array([0])),
        (np.array([1, 1, 0]), np.array([0])),
        (np.array([1, 1, 1]), np.array([1])),
    )
    a.learning_iteration(batch, 0.7)
    print('Epoch {}'.format(a.epochs))
    print('----------------------')
