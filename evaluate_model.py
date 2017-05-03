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
BATCH_SIZE=32
IMAGE_GENERATOR=False
NORMALIZE=True
VISUALIZE=False
GREYSCALE=False
VALIDATION_SIZE = 50000
VALIDATION_START = 60000

def process_arguments():

    assert len(sys.argv) == 5, "Error with number of arguments: <Model Path> <Type> <Evaluated Dataset Path> <Image Size>"
    assert (os.path.isfile(sys.argv[1])), "Error in model: file doesn't exist."
    assert (sys.argv[2]=="validation" or sys.argv[2]=="development" or sys.argv[2]=="test" )
    assert (os.path.isfile(sys.argv[3])), "Error in Dataset: file doesn't exist."
    assert (sys.argv[4] != 56 or sys.argv[4] != 112 or sys.argv[4] != 224), "Error in Image size: must be either 56:(56*112), 112:(112*112) or 224:(224*224)"

    model_path = sys.argv[1]
    type = sys.argv[2]
    evaluated_dataset_path = sys.argv[3]
    image_size = int(sys.argv[4])

    return model_path,type, evaluated_dataset_path, image_size

def load_as_numpy_array(dir,type):

    x_dataset,y_dataset = np.empty((0)),np.empty((0))
    file = h5py.File(dir, 'r')  # 'r' means that hdf5 file is open in read-only mode

    if (type == "validation"):
        x_dataset = np.array(file['training_input'][VALIDATION_START: VALIDATION_START + VALIDATION_SIZE])
        y_dataset = np.array(file['training_labels'][VALIDATION_START: VALIDATION_START + VALIDATION_SIZE])

    elif (type == "development"):
        x_dataset = np.array(file['development_input'])
        y_dataset = np.array(file['development_labels'])

    elif (type == "test"):
        x_dataset = np.array(file['test_input'])
        y_dataset = np.array(file['test_labels'])

    file.close()

    return x_dataset,y_dataset

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
    if NORMALIZE:
        x_np_temp = np.divide(x_np_temp, 255.0)

    # Change to greyscale if needed
    if GREYSCALE:
        x_np_temp = rgb2grey(x_np_temp)

    #Change y to categorical
    y_eval = to_categorical(y_np, num_classes=2)

    return x_np_temp,y_eval

def rgb2grey(x):

    r, g, b = x[ : , : , : , 0 ] , x[ : , : , : , 1 ], x[ : , : , : , 2 ]
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grey_reshaped = grey.reshape((x.shape[0],x.shape[1],x.shape[2],1))

    return grey_reshaped

if __name__ == "__main__":

    model_path,type, evaluated_dataset_path, image_size =  process_arguments()

    model = load_model(model_path)
    model.summary()

    x, y = load_as_numpy_array(evaluated_dataset_path,type = type)
    x_val, y_val = preprocess(x, y, image_size = image_size )

    score = model.evaluate(x_val, y_val,BATCH_SIZE=32, verbose=1)
    print('Validation Loss:' + str(score[0]))
    print('Validation Accuracy:' + str(score[1]))
    print('Validation Mean Absolute error:' + str(score[2]))