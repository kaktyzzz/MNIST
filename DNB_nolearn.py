from nolearn import dbn


import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score, r2_score
from scipy.ndimage import rotate, convolve
from matplotlib import pyplot as plt



def train_test_prep():
    '''
    This function will load the MNIST data, scale it to a 0 to 1 range, and split it into test/train sets.
    '''

    image_data = fetch_mldata('MNIST Original') # Get the MNIST dataset.

    basic_x = image_data.data
    basic_y = image_data.target # Separate images from their final classification.

    min_max_scaler = MinMaxScaler() # Create the MinMax object.
    basic_x = min_max_scaler.fit_transform(basic_x.astype(float)) # Scale pixel intensities only.

    x_train, x_test, y_train, y_test = train_test_split(basic_x, basic_y,
                            test_size = 0.2, random_state = 0) # Split training/test.
    return x_train, x_test, y_train, y_test

def random_image_generator(image):
    '''
    This function will randomly translate and rotate an image, producing a new, altered version as output.
    '''

    # Create our movement vectors for translation first.

    move_up = [[0, 1, 0],
               [0, 0, 0],
               [0, 0, 0]]

    move_left = [[0, 0, 0],
                 [1, 0, 0],
                 [0, 0, 0]]

    move_right = [[0, 0, 0],
                  [0, 0, 1],
                  [0, 0, 0]]

    move_down = [[0, 0, 0],
                 [0, 0, 0],
                 [0, 1, 0]]

    # Create a dict to store these directions in.

    dir_dict = {1:move_up, 2:move_left, 3:move_right, 4:move_down}

    # Pick a random direction to move.

    direction = dir_dict[np.random.randint(1,5)]

    # Pick a random angle to rotate (10 degrees clockwise to 10 degrees counter-clockwise).

    angle = np.random.randint(-10,11)

    # Move the random direction and change the pixel data back to a 2D shape.

    moved = convolve(image.reshape(28,28), direction, mode = 'constant')

    # Rotate the image

    rotated = rotate(moved, angle, reshape = False)

    return rotated



x_train, x_test, y_train, y_test = train_test_prep()

dbn_model = dbn.DBN([x_train.shape[1], 500, 100, 50, 10],
                learn_rates = 0.09,
                learn_rate_decays = 0.9,
                epochs = 50,
                verbose = 1)

dbn_model.fit(x_train, y_train)
y_true, y_pred = y_test, dbn_model.predict(x_test) # Get our predictions
print(classification_report(y_true, y_pred)) # Classification on each digit
print('ACC: %s' % accuracy_score(y_true, y_pred))
print('R^2: %s' % r2_score(y_true, y_pred))

sample = np.reshape(x_train[0], ((28,28))) # Get the training data back to its original form.
sample = sample*255. # Get the original pixel values.
plt.imshow(sample, cmap = plt.cm.gray)

distorted_example = random_image_generator(x_train[0]*255.)
plt.imshow(distorted_example, cmap = plt.cm.gray)


