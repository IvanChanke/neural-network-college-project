import numpy as np
import network_code_eng as net
from PIL import Image

def whiteness(pixel):

    return sum(pixel[:-1]) / 765


"""
1 is totally white
0 is pitch black
"""
network = net.Network((784, 32, 32, 1))

img = Image.open('MNIST_number7.png')
pix_white = Image.open('white.png').load()
pix_black = Image.open('black.png').load()

white_vector = []
for i in range(28):
    for j in range(28):
        white_vector.append(whiteness(pix_white[i, j]))
white_vector = np.array(white_vector)

black_vector = []
for i in range(28):
    for j in range(28):
        black_vector.append(whiteness(pix_black[i, j]))
black_vector = np.array(black_vector)

batch = (
    (white_vector, np.array([0])),
    (black_vector, np.array([1])),
)
while True:
    network.learning_iteration(batch, 0.5)
