from glob import glob
import numpy as np
import h5py
import random
import sys
import os
import matplotlib.pyplot as plt

INPUT_WIDTH=224
INPUT_HEIGHT=224
INPUT_CHANNEL=3
input_shape=(INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNEL)
SEQUENCE_LENGTH=25
PRINT_HISTOGRAM=False
SHUFFLE_SAMPLES=False

def get_file_names(dir):

    file = open(dir, "r")
    result = file.readlines()
    names_list = list(map(lambda x : x[0:-1] ,result))
    file.close()

    return names_list

def get_numpy_files(dir, names_list):

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

    #make sure they are the same lists:
    x_names = list(map(lambda x: x.split("/")[-1].split(".")[0] , flattened_Xv_array))
    y_names = list(map( lambda x: x.split("/")[-1].split(".")[0]      , flattened_Y_array))
    difference1 = [ item for item in x_names if item not in y_names]
    difference2 = [ item for item in y_names if item not in x_names]
    #Should print nothing!
    print(difference1)
    print(difference2)

    return flattened_Xv_array, flattened_Y_array

def save_to_hdf5(dir,output_file,type="training"):

    # validate shape along saving file to save time
    validate = True

    #If type is training, must do a special handling to split to validation set
    if(type=="training"):
        #Set training and validation video names, to be able to add the validation videos in the end
        validation_list = get_file_names("/vol/work1/dyab/training_set/validation_video_list_last_videos")
        #validation_list = get_file_names("/vol/work1/dyab/training_set/validation_video_list")
        all_list = get_file_names("/vol/work1/dyab/training_set/train_video_list")

        #Exclude validation file names from list of all names to get training names
        training_only_list = [val for val in all_list if val not in validation_list]

        #Shuffle training set with respect to videos
        random.shuffle(training_only_list)

        #get numpy arrays for training and validation
        train_x_fnames, train_y_fnames = get_numpy_files(dir, training_only_list)
        validation_x_fnames, validation_y_fnames = get_numpy_files(dir, validation_list)

        # concatenate training and validation output
        y_train_arrays = [np.load(f) for f in train_y_fnames]
        index_arr_train = [len(array) for array in y_train_arrays]
        y_train_arrays = np.concatenate(y_train_arrays)

        y_validate_arrays = [np.load(f) for f in validation_y_fnames]
        index_arr_validate = [len(array) for array in y_validate_arrays]
        y_validate_arrays = np.concatenate(y_validate_arrays)

        #concatenate training and validation input
        x_train_arrays=[np.load(f) for f in train_x_fnames]
        x_train_arrays=np.concatenate(x_train_arrays)

        if SHUFFLE_SAMPLES:
            #shuffle training video samples inplace
            index = np.arange(x_train_arrays.shape[0])
            np.random.shuffle(index)
            x_train_arrays = x_train_arrays[index]
            y_train_arrays = y_train_arrays[index]

        x_validate_arrays=[np.load(f) for f in validation_x_fnames]
        x_validate_arrays=np.concatenate(x_validate_arrays)

        x_dataset = np.concatenate((x_train_arrays,x_validate_arrays))
        y_dataset = np.concatenate((y_train_arrays, y_validate_arrays))
        index_arr = np.concatenate((index_arr_train,index_arr_validate))

        print(len(index_arr))
        print(sum(index_arr))
        print(index_arr)

    #If type is development or test, handle normally
    elif (type == "development" or type == "test" or type == "training_old"):
        # Get x
        x_fnames = glob(dir + "/*.Xv.npy")
        x_fnames.sort()
        # Get y
        y_fnames = glob(dir + "/*.Y.npy")
        y_fnames.sort()

        x_arrays = [np.load(f) for f in x_fnames]
        x_dataset = np.concatenate(x_arrays)

        y_arrays = [np.load(f) for f in y_fnames]
        y_dataset = np.concatenate(y_arrays)

        if SHUFFLE_SAMPLES:
            # shuffle training video samples inplace but keep validation set intact.
            size_to_shuffle = int(x_dataset.shape[0] * 0.8)
            index_to_shuffle = np.arange(size_to_shuffle)
            np.random.shuffle(index_to_shuffle)
            x_dataset[0:size_to_shuffle] = x_dataset[index_to_shuffle]
            y_dataset[0:size_to_shuffle] = y_dataset[index_to_shuffle]
            print("Shape should still be: "+str(x_dataset.shape[0]))

    #validate x_dataset shape
    if (x_dataset.shape[1] != INPUT_WIDTH or x_dataset.shape[2] != INPUT_HEIGHT or x_dataset.shape[3] != INPUT_CHANNEL):
        validate = False
        return validate

    if (y_dataset.shape[0] != x_dataset.shape[0]):
        print("x_dataset size: " + str(x_dataset.shape[0]) + ", y_dataset size: " + str(y_dataset.shape[0]))
        validate = False
        return validate

    print(x_dataset.shape)
    print(y_dataset.shape)

    f = h5py.File(output_file, 'w')

    if(type == "training"):

        f.create_dataset('index_array', data=index_arr)

        f.attrs['train_size'] = x_dataset.shape[0]
        f.create_dataset('index_array_train', data=index_arr_train)
        print(index_arr_train)

        f.attrs['validation_size'] = x_validate_arrays.shape[0]
        f.attrs['validation_start'] = x_train_arrays.shape[0]
        print(f.attrs['validation_start'])
        print(f.attrs['validation_size'])
        f.create_dataset('index_array_validate', data=index_arr_validate)
        print(index_arr_validate)

        # Creating dataset to store features
        f.create_dataset('training_input', data=x_dataset)

        # Creating dataset to store labels
        f.create_dataset('training_labels', data=y_dataset)

    elif (type == "training_old"):

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
    return validate

def process_arguments():

    assert len(sys.argv) == 4, "Error with number of arguments: <type> <directory> <outputfile name>"
    assert (sys.argv[1]=="training" or sys.argv[1]=="development" or sys.argv[1]=="test" or sys.argv[1]=="training_old" )
    assert (os.path.isdir(sys.argv[2])),"Error, directory doesn't exist"
    assert (not os.path.isfile(sys.argv[3])), "Error: file already exists, can't overwrite it"

    type = sys.argv[1]
    path = sys.argv[2]
    outputfile = sys.argv[3]

    return type, path, outputfile

def plot_histogram(dir,type="training"):

    #If type is training, must do a special handling to split to validation set
    if(type=="training"):
        #Set training and validation video names, to be able to add the validation videos in the end
        all_list = get_file_names("/vol/work1/dyab/training_set/train_video_list_subset")

        #get numpy arrays for training and validation
        train_x_fnames, train_y_fnames = get_numpy_files(dir, all_list)

        frames_lengths = [len(np.load(f)) for f in train_y_fnames]

        plt.hist(frames_lengths,bins=500)
        plt.show()

        histogram = dict()
        discard=0
        for f in train_y_fnames:
            array= np.load(f)
            length = len(array)

            if length in histogram:
                histogram[length]+=1
            else:
                histogram[length]=1

            if(length < SEQUENCE_LENGTH):
                discard+=1
        print("Percentage of numpy arrays to be discarded(because they are smaller than 25)\n")
        percentage = discard/len(train_x_fnames) * 100.
        print(percentage)
        print("Histogram\n")
        print(len(histogram))
        print(histogram)

if __name__ == "__main__":

    type, path, outputfile = process_arguments()

    if PRINT_HISTOGRAM:
        plot_histogram(path, type=type)

    validate = save_to_hdf5(path, outputfile, type=type)
    if(not validate):
        print("Error in dataset shape...")