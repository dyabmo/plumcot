from keras.models import load_model
import numpy as np
from keras.utils.np_utils import to_categorical
import h5py
import sys
import scipy.misc
import os
import matplotlib.pyplot as plt

DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3
nb_epoch = 100
batch_size=32
number_of_nodes=12
training_no_samples=0
development_no_samples=0
test_no_samples=0
WORKERS=10
IMAGE_GENERATOR=False
VISUALIZE=True


def visualize(x_train,y_train,i):

    index = batch_size // 8
    for j in range(0, index):
        plt.subplot(index // 2, index // 2, j + 1)

        # Speaking person will show in RGB
        scale = 1
        if (not y_train[j * 8][1]):
            scale = 255

        plt.imshow(x_train[j * 8] * scale)
    plt.savefig("/vol/work1/dyab/training_models/samples_visualization/batch_" + str(i) + ".png")

def process_arguments(arguments):

    assert len(arguments) == 5, "Error with number of arguments: <model path> <image size> <batch_size> <output_path>."
    assert (os.path.isfile(arguments[1])), "Error in model: file doesn't exist."
    assert (arguments[2] != 56 or arguments[2] != 112 or arguments[2] != 224), "Error in Image size: must be either 56:(56*112), 112:(112*112) or 224:(224*224)"
    assert (int(arguments[3]) % 32 == 0), "Error in batch size."
    assert (os.path.isdir(arguments[4])), "Error in output folder: folder doesn't exist."

    model_path = arguments[1]
    image_size = int(arguments[2])
    batch_size = int(arguments[3])
    output_path = arguments[4]
    if (output_path[-1] != "/"):
        output_path = output_path + "/"

    return model_path, image_size, batch_size, output_path

def set_paths():

    training_path = "/vol/work1/dyab/training_set/"
    training_file_name = "/train_dataset.h5"
    training_file = training_path + training_file_name

    development_path = "/vol/work1/dyab/development_set/"
    development_file_name = "/develop_dataset.h5"
    development_file = development_path + development_file_name

    return training_file, development_file

def load_as_numpy_array(dir,type):

    x_dataset,y_dataset = np.empty((0)),np.empty((0))
    file = h5py.File(dir, 'r')  # 'r' means that hdf5 file is open in read-only mode

    if (type == "training"):
        x_dataset = np.array(file['training_input'])
        y_dataset = np.array(file['training_labels'])

    elif (type == "development"):
        x_dataset = np.array(file['development_input'])
        y_dataset = np.array(file['development_labels'])

    elif (type == "test"):
        x_dataset = np.array(file['test_input'])
        y_dataset = np.array(file['test_labels'])

    file.close()

    return x_dataset,y_dataset

def visualize(x_train,y_train,i):

    index = batch_size // 8
    for j in range(0, index):
        plt.subplot(index // 2, index // 2, j + 1)

        # Speaking person will show in RGB
        scale = 1
        if (not y_train[j * 8][1]):
            scale = 255

        plt.imshow(x_train[j * 8] * scale)
    plt.savefig("/vol/work1/dyab/training_models/samples_visualization/evaluation_56/batch_" + str(i) + ".png")

def preprocess(x_np,y_np,image_size=DEFAULT_IMAGE_SIZE):

    #If image size is 112*112 or 56*112: first I must resize 224*224 to 112*112
    if (image_size == IMAGE_SIZE_112 ):

        x_np_temp = np.empty((x_np.shape[0], IMAGE_SIZE_112, IMAGE_SIZE_112, INPUT_CHANNEL))
        for j in range(0, x_np.shape[0]):
            x_np_temp[j] = scipy.misc.imresize(x_np[j], (IMAGE_SIZE_112, IMAGE_SIZE_112))

    #If the requested image size was originally 56*112, then crop lower part of image, hopefully capturing the mouth, discard the upper one.
    elif(image_size == IMAGE_SIZE_56 ):

        x_np_temp = np.empty((x_np.shape[0], IMAGE_SIZE_56, IMAGE_SIZE_112, INPUT_CHANNEL))
        for j in range(0, x_np.shape[0]):
            temp = scipy.misc.imresize(x_np[j], (IMAGE_SIZE_112, IMAGE_SIZE_112))
            x_np_temp[j] = temp[IMAGE_SIZE_56: IMAGE_SIZE_112 , :, :]

    elif (image_size == DEFAULT_IMAGE_SIZE):
        x_np_temp = x_np

    # Perform simple normalization
    x_eval = np.divide(x_np_temp, 255.0)

    #Change y to categorical
    y_eval = to_categorical(y_np, num_classes=2)

    return x_eval,y_eval

if __name__ == "__main__":

    model_path, image_size, batch_size, output_path = process_arguments(sys.argv)

    model = load_model(model_path)
    model.summary()

    _, development_file = set_paths()

    x, y = load_as_numpy_array(development_file,type = "development")

    x_val, y_val = preprocess(x, y, image_size = image_size )

    # Visualize 1/8 images out of each batch
    if VISUALIZE:
        num_batches = int( x_val.shape[0] / batch_size ) - batch_size
        for i in range(num_batches):
            visualize(x_val[i: i+batch_size], y_val[i: i+batch_size], i)


    score = model.evaluate(x_val, y_val,batch_size=32, verbose=1)
    print('Validation Loss:' + str(score[0]))
    print('Validation Accuracy:' + str(score[1]))
    print('Validation Mean Absolute error:' + str(score[2]))