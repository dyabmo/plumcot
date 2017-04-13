from keras.models import load_model
from glob import glob
import numpy as np
from keras.utils.io_utils import HDF5Matrix
import h5py
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from visual_callbacks import AccLossPlotter, ConfusionMatrixPlotter
import sys
import timeit
import scipy.misc
import os

DEFAULT_IMAGE_SIZE=224
INPUT_CHANNEL=3
nb_epoch = 100
batch_size=32
number_of_nodes=12
training_no_samples=0
development_no_samples=0
test_no_samples=0

def process_arguments(arguments):

    assert len(arguments) == 5, "Error with number of arguments: <model path> <image size> <batch_size> <output_path>."
    assert (os.path.isfile(arguments[1])), "Error in model: file doesn't exist."
    assert (arguments[2] != 112 or arguments[2] != 224), "Error in Image size: must be either 224 or 112."
    assert (int(arguments[3]) % 32 == 0), "Error in batch size."
    assert (os.path.isdir(arguments)), "Error in output folder: folder doesn't exist."

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

class TimeLogger(Callback):

    def __init__(self, path):
        self.validation_data = None
        self.fd = open(path, 'w')
        self.fd.write("Epoch , Duration in minutes\n")
        self.fd.flush()

    def on_train_begin(self, logs={}):

        self.start_time = timeit.default_timer()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = (timeit.default_timer() - self.start_time)/60.0
        self.fd.write(str(epoch) + ", " + '{:.2f}'.format(elapsed) + "\n")
        self.fd.flush()

    def on_train_end(self, logs=None):
        self.fd.flush()
        self.fd.close()

def set_no_samples(train_dir,dev_dir=None,test_dir=None):

    #Set number of samples to calculate: steps_per_epoch automatically
    global training_no_samples
    global development_no_samples
    global test_no_samples

    f = h5py.File(train_dir, 'r')
    training_no_samples = f.attrs['train_size']
    print("Training number of samples: "+str(training_no_samples))

    if (dev_dir):
        f = h5py.File(dev_dir, 'r')
        development_no_samples = f.attrs['dev_size']
        print("Development number of samples: " + str(development_no_samples))

    if (test_dir):
        f = h5py.File(test_dir, 'r')
        test_no_samples = f.attrs['test_size']
        print("Test number of samples: " + str(test_no_samples))

def load_from_hdf5(dir,type,start=0,end=None):

    X_train, y_train = 0,0

    if(type=="training"):
        X_train = HDF5Matrix(dir, 'training_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'training_labels',start=start,end=end)

    elif(type=="development"):
        X_train = HDF5Matrix(dir, 'development_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'development_labels',start=start,end=end)

    elif (type == "test"):
        X_train = HDF5Matrix(dir, 'test_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'test_labels',start=start,end=end)

    return X_train,y_train

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

def random_shuffle_2_arrays(X_train,y_train):

    index = np.arange(X_train.shape[0])
    #Shuffle inplace
    np.random.shuffle(index)

    X_train=X_train[index]
    y_train = y_train[index]

    return X_train, y_train

def random_shuffle_subset( x_train,ratio=1):

    size = x_train.shape[0]
    # Shuffle inplace
    np.random.shuffle(x_train)

    subset = int(size*ratio)
    x_subset = x_train[0: subset]

    return x_subset

def generate_imges_from_hdf5(file,image_size,type="training",):

    index=0

    if(type=="training"):
        index = training_no_samples

    elif(type=="development"):
        index = development_no_samples

    while 1:

        for i in range(0,index-batch_size,batch_size):

            x, y = load_from_hdf5(file, type=type,start=i,end=i+batch_size)
            x_train, y_train = preprocess_batch(x,y,image_size=image_size)

            yield (x_train, y_train)

def preprocess_batch(x,y,image_size=DEFAULT_IMAGE_SIZE):
    # Convert to numpy array
    x_np = np.array(x)
    y_np = np.array(y)

    # Resize if needed
    x_np_temp = np.empty((x_np.shape[0], image_size, image_size, INPUT_CHANNEL))
    if (image_size != DEFAULT_IMAGE_SIZE):
        for j in range(0, x_np.shape[0]):
            x_np_temp[j] = scipy.misc.imresize(x_np[j], (image_size, image_size))
        del x_np
        x_np = x_np_temp
    elif (image_size == DEFAULT_IMAGE_SIZE):
        del x_np_temp

    # Shuffle
    x_train, y_train = random_shuffle_2_arrays(x_np, y_np)

    # Perform simple normalization
    x_train = np.divide(x_train, 255.0)

    return x_train,y_train

def calculate_steps_per_epoch():

    steps_per_epoch_train = int(training_no_samples/batch_size)
    development_steps = int(development_no_samples / batch_size )

    return steps_per_epoch_train, development_steps

if __name__ == "__main__":

    model_path, image_size, batch_size, output_path = process_arguments(sys.argv)

    training_file, development_file = set_paths()

    #Set global variables
    set_no_samples(training_file, development_file)
    steps_per_epoch_train, development_steps = calculate_steps_per_epoch();

    model = load_model(model_path)
    model.summary()

    #list of callbacks:
    plotter     = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True,path= output_path, name='graph_Epoch.')
    csv_logger  = CSVLogger(output_path+"csv_logger.csv")
    time_logger = TimeLogger(output_path+"time_logger.csv")
    checkpoint  = ModelCheckpoint(output_path+"Epoch.{epoch:02d}_Val_Acc.{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [plotter, csv_logger, time_logger, checkpoint]

    #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
    training_generator = generate_imges_from_hdf5(file=training_file, type="training", image_size= image_size)
    development_generator = generate_imges_from_hdf5(file=development_file,type="development",image_size=image_size)

    model.fit_generator(training_generator,verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=nb_epoch, validation_data = development_generator, validation_steps=development_steps ,callbacks= callbacks_list,pickle_safe=True,workers=10)

    #Plot confusion matrix
    X_val,y_val =  load_from_hdf5(development_file, "development")
    ConfusionMatrixPlotter(X_val=X_val, classes=("Talking","Not Talking"), Y_val=y_val,path=output_path)
    score = model.evaluate_generator(development_generator, steps=development_steps)

    print('Validation Accuracy:' + str(score[0]))
    print('Validation Mean Square error:' + str(score[1]))