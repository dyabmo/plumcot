from glob import glob
import numpy as np
import h5py

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

def save_to_hdf5(dir,output_file,type="training"):

    #Get x
    x_fnames = glob(dir + "/*.Xv.npy")
    x_fnames.sort()
    x_arrays = [np.load(f) for f in x_fnames]
    x_dataset = np.concatenate(x_arrays)

    #Get y
    y_fnames = glob(dir + "/*.Y.npy")
    y_fnames.sort()
    y_arrays = [np.load(f) for f in y_fnames]
    y_dataset = np.concatenate(y_arrays)

    f = h5py.File(output_file, 'w')

    if(type == "training"):

        f.attrs['train_size'] = x_dataset.shape[0]
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

if __name__ == "__main__":

    training_path = "/vol/work1/dyab/training_set/"
    training_file_name = "/train_dataset2.h5"
    input_file = training_path + training_file_name

    development_path = "/vol/work1/dyab/development_set/"
    development_file_name = "/develop_dataset2.h5"
    development_file = development_path + development_file_name

    test_path = "/vol/work1/dyab/test_set/"
    test_file_name = "/test_dataset.h5"
    test_file = test_path + test_file_name

    if(validate_dataset_shape(test_path)):
        save_to_hdf5(test_path,test_file,type="test")