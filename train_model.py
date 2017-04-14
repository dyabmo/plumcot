from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.utils.io_utils import HDF5Matrix
from keras.utils.np_utils import to_categorical
import h5py
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from visual_callbacks import AccLossPlotter, ConfusionMatrixPlotter
import sys
import timeit
import scipy.misc
import os

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
IMAGE_GENERATOR=False

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

    #Randomize which batch to get
    rand_index = np.arange(start=0, stop = index-batch_size, step = batch_size)
    np.random.shuffle(rand_index)

    while 1:

        for i in range(0, rand_index.shape[0] ):

            #Choose a random batch
            x,y = load_from_hdf5(file, type=type,start=rand_index[i],end=rand_index[i]+batch_size)
            #Proprocess random batch: shuffle samples, rescale values, resize if needed
            x_train, y_train = preprocess_batch(x,y,image_size=image_size)

            yield (x_train, y_train)

def preprocess_batch(x,y,image_size=DEFAULT_IMAGE_SIZE):
    # Convert to numpy array
    x_np = np.array(x)
    y_np = np.array(y)

    #If image size is 112*112 or 56*112: first I must resize 224*224 to 112*112
    if (image_size == IMAGE_SIZE_112 ):

        x_np_temp = np.empty((batch_size, IMAGE_SIZE_112, IMAGE_SIZE_112, INPUT_CHANNEL))
        for j in range(0, batch_size):
            x_np_temp[j] = scipy.misc.imresize(x_np[j], (IMAGE_SIZE_112, IMAGE_SIZE_112))

    #If the requested image size was originally 56*112, then crop lower part of image, hopefully capturing the mouth, discard the upper one.
    elif(image_size == IMAGE_SIZE_56 ):

        x_np_temp = np.empty((batch_size, IMAGE_SIZE_56, IMAGE_SIZE_112, INPUT_CHANNEL))
        for j in range(0, batch_size):
            temp = scipy.misc.imresize(x_np[j], (IMAGE_SIZE_112, IMAGE_SIZE_112))
            x_np_temp[j] = temp[IMAGE_SIZE_56: IMAGE_SIZE_112 , :, :]

    elif (image_size == DEFAULT_IMAGE_SIZE):
        x_np_temp = x_np

    # Shuffle
    x_train, y_train = random_shuffle_2_arrays(x_np_temp, y_np)

    # Perform simple normalization
    x_train = np.divide(x_train, 255.0)

    #Change y to categorical
    y_train = to_categorical(y_train, num_classes=2)

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

    if(IMAGE_GENERATOR):
        x_train, y_train = load_as_numpy_array(dir=training_file, type="training")
        x_dev, y_dev = load_as_numpy_array(dir=development_file, type="development")

        datagen = ImageDataGenerator(rescale=1./255, data_format="channels_last", featurewise_center=True , featurewise_std_normalization=True)

        #fit for featurewise center and std normalization
        datagen.fit( random_shuffle_subset(x_train, 0.1 ))

        training_generator = datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
        development_generator = datagen.flow(x_dev, y_dev, batch_size=batch_size, shuffle=True)

    else:
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