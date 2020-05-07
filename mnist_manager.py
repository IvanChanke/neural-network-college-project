import numpy as np
from PIL import Image
import os
import joblib

label_map = {
    0: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    1: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    2: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
    3: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
    4: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    5: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
    6: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
    7: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
    8: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
    9: np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
}

def darkness(pixel):

    return 1 - pixel / 255

def image_to_vector(image):
    vector = []
    for i in range(28):
        for j in range(28):
            vector.append(darkness(image[i, j]))

    return np.array(vector)


print('mnist is importing...')
print('Gathering train data...')

images_path = []
for i in range(10):
    images_path.append(
        os.listdir(path = 'mnist_png_small\\training\\{}'.format(str(i)))
    )
# Now images_path contains lists of train file names for numbers 0-9
train_images = []
for i in range(10): # For each number
    train_images.append([]) # Initialize a list for (vector, label) pairs
    for img in images_path[i]: # For each file in the list for the number
        train_images[i].append(
            (
                image_to_vector(Image.open('mnist_png\\training\\{}\\{}'.format(i, img)).load()), label_map[i]
            ) # A (vector, label) tuple
        )
print('Done')
# Now train_images contains 10 lists (for each digit 0-9) of (vector, label) tuples

print('Gathering test data...')

images_path = []
for i in range(10):
    images_path.append(
        os.listdir(path = 'mnist_png\\testing\\{}'.format(str(i)))
    )
# Now images_path contains lists of test file names for numbers 0-9
test_images = []
for i in range(10): # For each number
    test_images.append([]) # Initialize a list for (vector, label) pairs
    for img in images_path[i]: # For each file in the list for the number
        test_images[i].append(
            (
                image_to_vector(Image.open('mnist\\testing\\{}\\{}'.format(i, img)).load()), label_map[i]
            ) # A (vector, label) tuple
        )
# Now test_images contains 10 lists (for each digit 0-9) of (vector, label) tuples

print('Done')
print('Successfully imported mnist')

f = open('train_data', 'wb')
joblib.dump(train_images, f)
f.close()

f = open('test_data', 'wb')
joblib.dump(test_images, f)
f.close()
