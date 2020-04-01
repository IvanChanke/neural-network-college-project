import numpy as np
import ptron as net
from PIL import Image

def darkness(pixel):

    return 1 - pixel / 255

def image_to_vector(image):
    vector = []
    for i in range(28):
        for j in range(28):
            vector.append(darkness(image[i, j]))

    return np.array(vector)

"""
0 is totally white
1 is pitch black
"""
network = net.Network((784, 64, 64, 32, 10))

zero = Image.open('zero.jpg').load()
one = Image.open('one.jpg').load()
two = Image.open('two.jpg').load()
three = Image.open('three.jpg').load()
four = Image.open('four.jpg').load()
five = Image.open('five.jpg').load()
six = Image.open('six.jpg').load()
seven = Image.open('seven.jpg').load()
eight = Image.open('eight.jpg').load()
nine = Image.open('nine.jpg').load()

zero_vector = image_to_vector(zero)
one_vector = image_to_vector(one)
two_vector = image_to_vector(two)
three_vector = image_to_vector(three)
four_vector = image_to_vector(four)
five_vector = image_to_vector(five)
six_vector = image_to_vector(six)
seven_vector = image_to_vector(seven)
eight_vector = image_to_vector(eight)
nine_vector = image_to_vector(nine)



batch = (
    (zero_vector, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])),
    (one_vector, np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])),
    (two_vector, np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])),
    (three_vector, np.array([0, 0, 0, 1, 0, 0, 0 ,0 ,0 ,0])),
    (four_vector, np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0])),
    (five_vector, np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])),
    (six_vector, np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])),
    (seven_vector, np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])),
    (eight_vector, np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
    (nine_vector, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])),
)
batch1 = (
    (zero_vector, np.array([0, 1, 0, 0, 0, 0])),
)
batch2 = (
    (zero_vector, np.array([0, 0, 1, 0, 0, 0])),
)
batch3 = (
    (zero_vector, np.array([1, 0, 0, 0, 0, 0])),
)
batch4 = (
    (zero_vector, np.array([1, 0, 0, 0, 0, 0])),
)

i = 0
while True:
    i += 1
    network.learning_iteration(batch, 0.25)
    print('----------------------------------')
    print('Epoch', i)
