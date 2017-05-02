from glob import glob
import numpy as np
import h5py
import random
import sys
import os

INPUT_WIDTH=224
INPUT_HEIGHT=224
INPUT_CHANNEL=3
input_shape=(INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNEL)
DEFAULT_IMAGE_SIZE=224
nb_epoch = 100
batch_size=32
number_of_nodes=12
training_no_samples=0
development_no_samples=0
test_no_samples=0
SHUFFLE_SAMPLES_INPLACE=False

def validate_dataset_shape(dir):

    validate = True

    #Validate input shape
    x_fnames = glob(dir + "/*.Xv.npy")
    x_fnames.sort()
    x_size=0
    for f in x_fnames:
        array = np.load(f)
        x_size = x_size + array.shape[0]

        if (array.shape[1] != INPUT_WIDTH or array.shape[2] != INPUT_HEIGHT or array.shape[3] != INPUT_CHANNEL):
            print(f + ":  " + str(array.shape))
            validate=False
    print("x_size: " + str(x_size))
    #Validate output shape
    y_fnames = glob(dir + "/*.Y.npy")
    y_fnames.sort()
    y_size=0
    for f in y_fnames:
        array = np.load(f)
        y_size = y_size + array.shape[0]

    #validate that both are equal in shape
    if(x_size != y_size):
        print("x_size: " + str(x_size) + ", y_size: "+str(y_size))
        validate = False

    return validate

def get_file_names(dir):

    file = open(dir, "r")
    result = file.readlines()
    names_list = list(map(lambda x : x[0:-1] ,result))
    file.close()

    return names_list

def get_numpy_arrays(dir,names_list):

    Xv_array=list()
    Y_array=list()

    for item in names_list:
        Xv_array.append(glob(dir + "/" + item + "*.Xv.npy"))
        Y_array.append(glob(dir + "/" + item + "*.Y.npy"))

    #remove empty entries
    Xv_array=list(filter(lambda x:len(x)!=0 ,Xv_array) )
    Y_array = list(filter(lambda x: len(x) != 0, Y_array))

    #flatten list
    flattened_Xv_array = [val for sublist in Xv_array for val in sublist]
    flattened_Y_array =  [val for sublist in Y_array  for val in sublist]

    return flattened_Xv_array, flattened_Y_array

def save_to_hdf5(dir,output_file,type="training"):

    #Set training and validation video names, to be able to add the validation videos in the end
    validation_list = get_file_names("/vol/work1/dyab/validation_video_list")
    all_list = get_file_names("/vol/work1/dyab/train_video_list")

    #Exclude validation file names from list of all names to get training names
    training_only_list = [val for val in all_list if val not in validation_list]

    #Shuffle training set with respect to videos
    random.shuffle(training_only_list)

    #get numpy arrays for training and validation
    train_x_fnames, train_y_fnames = get_numpy_arrays(dir,training_only_list)
    validation_x_fnames, validation_y_fnames = get_numpy_arrays(dir,validation_list)

    #concatenate training and validation input
    x_train_arrays=[np.load(f) for f in train_x_fnames]
    x_train_arrays=np.concatenate(x_train_arrays)

    x_validate_arrays=[np.load(f) for f in validation_x_fnames]
    x_validate_arrays=np.concatenate(x_validate_arrays)

    x_dataset = np.concatenate((x_train_arrays,x_validate_arrays))

    #concatenate training and validation output
    y_train_arrays = [np.load(f) for f in train_y_fnames]
    y_train_arrays = np.concatenate(y_train_arrays)

    y_validate_arrays = [np.load(f) for f in validation_y_fnames]
    y_validate_arrays = np.concatenate(y_validate_arrays)

    y_dataset = np.concatenate((y_train_arrays, y_validate_arrays))

    print(x_dataset.shape)
    print(y_dataset.shape)

    if SHUFFLE_SAMPLES_INPLACE:
        #Shuffle samples inplace: not useful! must shuffle videos themselves
        index = np.arange(x_dataset.shape[0])
        np.random.shuffle(index)
        x_dataset=x_dataset[index]
        y_dataset = y_dataset[index]

    f = h5py.File(output_file, 'w')

    if(type == "training"):

        f.attrs['train_size'] = x_dataset.shape[0]
        f.attrs['validation_size'] = x_validate_arrays.shape[0]
        print(f.attrs['validation_size'])
        # Creating dataset to store features
        f.create_dataset('training_input', data=x_dataset)

        # Creating dataset to store labels
        f.create_dataset('training_labels', data=y_dataset)

    elif (type == "development"):

        f.attrs['dev_size'] = x_dataset.shape[0]

        # Creating dataset to store features
        f.create_dataset('development_input', data=x_dataset)

        # Creating dataset to store labels
        f.create_dataset('development_labels', data=y_dataset)

    elif (type == "test"):

        f.attrs['test_size'] = x_dataset.shape[0]

        # Creating dataset to store features
        f.create_dataset('test_input', data=x_dataset)

        # Creating dataset to store labels
        f.create_dataset('test_labels', data=y_dataset)

    f.flush()
    f.close()

def process_arguments():

    assert len(sys.argv) == 4, "Error with number of arguments: <type> <directory> <outputfile name>"
    assert (sys.argv[1]=="training" or sys.argv[1]=="development" or sys.argv[1]=="test" )
    assert (os.path.isdir(sys.argv[2])),"Error, directory doesn't exist"
    assert (not os.path.isfile(sys.argv[3])), "Error: file already exists, can't overwrite it"

    type = sys.argv[1]
    path = sys.argv[2]
    outputfile = sys.argv[3]

    return type, path, outputfile

if __name__ == "__main__":

    type, path, outputfile = process_arguments()

    save_to_hdf5(path, outputfile, type=type)