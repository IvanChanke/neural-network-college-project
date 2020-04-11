import numpy as np
import ptron as net
import mnist
import joblib


network = net.Network((784, 64, 64, 32, 10))
network.set_printing_option(2)

b_0 = mnist.train_images[0][:1]
b_1 = mnist.train_images[1][:1]
b_2 = mnist.train_images[2][:1]
b_3 = mnist.train_images[3][:1]
b_4 = mnist.train_images[4][:1]
b_5 = mnist.train_images[5][:1]
b_6 = mnist.train_images[6][:1]
b_7 = mnist.train_images[7][:1]
b_8 = mnist.train_images[8][:1]
b_9 = mnist.train_images[9][:1]

train_b = np.array(
    b_0 + b_1 + b_2 + b_3 + b_4 + b_5 + b_6 + b_7 + b_8 + b_9
)

print(len(train_b))
np.random.shuffle(train_b)

for i in range(450):
    print('Iteration', i)
    network.learning_iteration(train_b, 0.35)

joblib.dump(network, 'model_testing_shuffled_2_samples')
