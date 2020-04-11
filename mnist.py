import joblib

print('MNIST is loading...')
train_images = joblib.load('train_data')
test_images = joblib.load('test_data')
print('Successfully imported MNIST')
