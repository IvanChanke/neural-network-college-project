import numpy as np

batch_xor3 = (
    (np.array([0, 0, 0]), np.array([0])),
    (np.array([0, 0, 1]), np.array([1])),
    (np.array([0, 1, 0]), np.array([1])),
    (np.array([0, 1, 1]), np.array([0])),
    (np.array([1, 0, 0]), np.array([1])),
    (np.array([1, 0, 1]), np.array([0])),
    (np.array([1, 1, 0]), np.array([0])),
    (np.array([1, 1, 1]), np.array([1])),
)
batch_xor2 = (
    (np.array([0, 0]), np.array([0])),
    (np.array([0, 1]), np.array([1])),
    (np.array([1, 0]), np.array([1])),
    (np.array([1, 1]), np.array([0])),
)
batch_and3 = (
    (np.array([0, 0, 0]), np.array([0])),
    (np.array([0, 0, 1]), np.array([0])),
    (np.array([0, 1, 0]), np.array([0])),
    (np.array([0, 1, 1]), np.array([0])),
    (np.array([1, 0, 0]), np.array([0])),
    (np.array([1, 0, 1]), np.array([0])),
    (np.array([1, 1, 0]), np.array([0])),
    (np.array([1, 1, 1]), np.array([1])),
)
batch_or3 = (
    (np.array([0, 0, 0]), np.array([0])),
    (np.array([0, 0, 1]), np.array([1])),
    (np.array([0, 1, 0]), np.array([1])),
    (np.array([0, 1, 1]), np.array([1])),
    (np.array([1, 0, 0]), np.array([1])),
    (np.array([1, 0, 1]), np.array([1])),
    (np.array([1, 1, 0]), np.array([1])),
    (np.array([1, 1, 1]), np.array([1])),
)


data = {
    'XOR3' : batch_xor3,
    'XOR2' : batch_xor2,
    'AND3' : batch_and3,
    'OR3' : batch_or3
}
