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
import matplotlib.pyplot as plt
import models.MT_IM56_NODROP as mt

DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3
WORKERS=10
NB_EPOCH = 100
BATCH_SIZE=32
training_no_samples=0
validation_no_samples=0
validation_start=0
development_no_samples=0
test_no_samples=0
index_arr = np.zeros((0))
IMAGE_GENERATOR=False
TRAINING_FIT_RATIO= 0.1
NORMALIZE=True
VISUALIZE=False
GREYSCALE=False
SHUFFLE_BATCHES=False
USE_VALIDATION = True
TRAINING_RATIO = 0.9
MODEL_MT=True
SEQUENCE_LENGTH=25
STEP=5

#To be able to visualize correctly
if VISUALIZE:
    WORKERS = 1
    NB_EPOCH = 1

def process_arguments():

    assert len(sys.argv) == 7, "Error with number of arguments: <training set path> <development set path>  <model path> <image size> <BATCH_SIZE> <output_path>."
    assert (os.path.isfile(sys.argv[1])), "Error in training set: file doesn't exist."
    assert (os.path.isfile(sys.argv[2])), "Error in development set: file doesn't exist."
    assert (os.path.isfile(sys.argv[3])), "Error in model: file doesn't exist."
    assert (sys.argv[4] != 56 or sys.argv[4] != 112 or sys.argv[4] != 224), "Error in Image size: must be either 56:(56*112), 112:(112*112) or 224:(224*224)"
    assert (int(sys.argv[5]) % 2 == 0), "Error in batch size."
    assert (os.path.isdir(sys.argv[6])), "Error in output folder: folder doesn't exist."

    training_file = sys.argv[1]
    development_file = sys.argv[2]
    model_path = sys.argv[3]
    image_size = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    output_path = sys.argv[6]
    if (output_path[-1] != "/"):
        output_path = output_path + "/"

    return training_file,development_file, model_path, image_size, batch_size, output_path

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
    global validation_no_samples
    global validation_start
    global development_no_samples
    global test_no_samples
    global index_arr

    f = h5py.File(train_dir, 'r')

    if MODEL_MT:
        index_arr = np.array(f['index_array'])
        print(index_arr)

    total_training_size = int(f.attrs['train_size'])
    training_no_samples = int(f.attrs['train_size'] * TRAINING_RATIO )
    print("Training file:" + train_dir)
    print("Total number of training samples: "+str(total_training_size) )
    print("Training number of samples used(and training end): "+str(training_no_samples))

    if USE_VALIDATION:
        validation_no_samples = int(f.attrs['validation_size'])
        validation_start = total_training_size - validation_no_samples - 1
        print("Validation start: " + str(validation_start))
        print("Validation number of samples: " + str(validation_no_samples))

    if (dev_dir and not USE_VALIDATION):
        f = h5py.File(dev_dir, 'r')
        print("Development file:" + dev_dir)
        development_no_samples = f.attrs['dev_size']
        print("Development number of samples: " + str(development_no_samples))

    if (test_dir):
        f = h5py.File(test_dir, 'r')
        test_no_samples = f.attrs['test_size']
        print("Test number of samples: " + str(test_no_samples))

def load_from_hdf5(dir,type,start=0,end=None):

    X_train, y_train = 0,0

    if(type=="training" or type == "validation"):
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
        x_dataset = np.array(file['training_input'][0:training_no_samples-1])
        y_dataset = np.array(file['training_labels'][0:training_no_samples-1])

    elif (type == "validation"):
        x_dataset = np.array(file['training_input'][validation_start: training_no_samples-1])
        y_dataset = np.array(file['training_labels'][validation_start: training_no_samples-1])

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

def compute_y_mt(y):

    #Compute y_mt using the majority of labels in y
    majority = sum(y[:,1])
    if majority >= int(SEQUENCE_LENGTH/2):
        y_mt=[0,1]
    else:
        y_mt=[1,0]

    return y_mt

def prepare_mt(x,y):

    # If model is multi_tower, change batch size to (32,25,heigh,width,3) (32,2)

    length = len(x)
    length = length - (length % STEP)

    no_samples = int( ((length - SEQUENCE_LENGTH)/STEP) ) + 1
    x = x[0:length]
    print(x.shape)

    x_mt = x[0:SEQUENCE_LENGTH]
    x_mt=x_mt.reshape((1,SEQUENCE_LENGTH, x_mt.shape[1], x_mt.shape[2], x_mt.shape[3]))

    #Compute y_mt using the majority of labels in y
    y_mt = compute_y_mt(y[0:SEQUENCE_LENGTH])

    for i in range(no_samples - 1):

        start = ((i+1) * STEP)
        x_mt_next = x[start :SEQUENCE_LENGTH]
        x_mt_next = x_mt_next.reshape((1, SEQUENCE_LENGTH, x_mt_next.shape[1], x_mt_next.shape[2], x_mt_next.shape[3]))
        x_mt = np.concatenate((x_mt,x_mt_next))

        # Compute y_mt using the majority of labels in y
        y_mt_next = compute_y_mt(y[start:SEQUENCE_LENGTH])
        y_mt = np.concatenate((y_mt,y_mt_next))

        print(x_mt.shape)
        print(y_mt.shape)

    return x_mt,y_mt

def generate_images_hdf5_mt(file,image_size,type="training" ):

    #TODO: Calculate steps per epoch!!

    i=0
    facetrack_stopping_index=0

    while 1:

        while i < (len(index_arr) -1 ):

            #check if currrent facetrack's length is bigger than SEQUENCE_LENGTH or not
            #Only facetracks of length bigger than SEQUENCE_LENGTH will be used
            if index_arr[i] < SEQUENCE_LENGTH:
                i=i+1

            else:

                #Only in the initial case
                if i==0:
                    start=0
                    end = index_arr[i]
                else:
                    start=index_arr[i] + facetrack_stopping_index
                    end = index_arr[i+1] + sum(index_arr[0:i])

                #Do once at the beginning
                x, y = load_from_hdf5(file, type=type, start=start, end=end)
                x_train_processed, y_train_processed = preprocess(x, y, image_size=image_size)
                x_train,y_train = prepare_mt(x_train_processed,y_train_processed)

                exit(0)
                #Keep doing until batch size is reached
                while len(x_train) < (BATCH_SIZE * STEP):

                    i = i + 1
                    start = index_arr[i]
                    end = index_arr[i + 1] + sum(index_arr[0:i])

                    x, y = load_from_hdf5(file, type=type, start=start, end=end)
                    x_train_processed, y_train_processed = preprocess(x, y, image_size=image_size)
                    x_train_next, y_train_next = prepare_mt(x_train_processed, y_train_processed)

                    x_train = np.concatenate((x_train,x_train_next))
                    y_train = np.concatenate((y_train,y_train_next))

                #to save the place where we stopped in the middle of a facetrack, to use the remaining in next batch
                facetrack_stopping_index = len(x_train) - (BATCH_SIZE * STEP)


            yield (x_train, y_train)

def generate_imges_from_hdf5(file,image_size,type="training"):

    index=0
    offset=0

    if(type=="training"):
        index = training_no_samples

    elif (type == "validation"):
        index = validation_no_samples
        offset= validation_start

    elif(type=="development"):
        index = development_no_samples

    #Randomize which batch to get
    rand_index = np.arange(start=0, stop =index - BATCH_SIZE, step = BATCH_SIZE)

    if SHUFFLE_BATCHES:
        np.random.shuffle(rand_index)

    while 1:

        for i in range(rand_index.shape[0]):

            #Choose a random batch
            x,y = load_from_hdf5(file, type=type, start=rand_index[i] + offset, end=rand_index[i] + BATCH_SIZE + offset)
            #Proprocess random batch: shuffle samples, rescale values, resize if needed
            x_train, y_train = preprocess(x,y,image_size=image_size)

            #Visualize 1/8 images out of each batch
            if VISUALIZE: visualize(x_train, y_train,i,type)

            yield (x_train, y_train)

def visualize(x_train,y_train,i,type):

    index = BATCH_SIZE // 8
    for j in range(0, index):
        plt.subplot(index // 2, index // 2, j + 1)

        image = x_train[j * 8]

        if GREYSCALE:
            image = image.reshape((x_train.shape[1], x_train.shape[2]))
            cmap = plt.cm.gray
            if (not y_train[j * 8][1]):
                cmap=None
            plt.imshow(image, cmap=cmap)

        else:
        # Speaking person will show in RGB, but not inverted colors
            scale = 1
            if (not y_train[j * 8][1]):
                scale = 255
            plt.imshow( image * scale)

    if(type == "training"):
        plt.savefig("/vol/work1/dyab/samples_visualization/cluster_training/batch_" + str(i) + ".png")
    elif(type == "development"):
        plt.savefig("/vol/work1/dyab/samples_visualization/cluster_eval/batch_" + str(i) + ".png")
    elif (type == "validation"):
        plt.savefig("/vol/work1/dyab/samples_visualization/cluster_validation/batch_" + str(i) + ".png")

def preprocess(x,y,image_size=DEFAULT_IMAGE_SIZE):
    # Convert to numpy array
    x_np = np.array(x)
    y_np = np.array(y)

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

    # Shuffle
    x_train, y_train = random_shuffle_2_arrays(x_np_temp, y_np)

    # Perform simple normalization
    if NORMALIZE:
        x_train = np.divide(x_train, 255.0)

    #Change to greyscale if needed
    if GREYSCALE:
        x_train = rgb2grey(x_train)

    #Change y to categorical
    y_train = to_categorical(y_train, num_classes=2)
    return x_train,y_train


# Change to greyscale
def rgb2grey(x):

    r, g, b = x[ : , : , : , 0 ] , x[ : , : , : , 1 ], x[ : , : , : , 2 ]
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grey_reshaped = grey.reshape((x.shape[0],x.shape[1],x.shape[2],1))

    return grey_reshaped

def calculate_steps_per_epoch():

    steps_per_epoch_train = int(training_no_samples / BATCH_SIZE)
    validation_steps = int(validation_no_samples / BATCH_SIZE)
    development_steps = int(development_no_samples / BATCH_SIZE)

    return steps_per_epoch_train,validation_steps, development_steps

if __name__ == "__main__":

    training_file, development_file, model_path, image_size, BATCH_SIZE, output_path = process_arguments()

    #Set global variables
    set_no_samples(training_file, development_file)
    steps_per_epoch_train,validation_steps, development_steps = calculate_steps_per_epoch();

    if MODEL_MT:
        model = mt.get_model()
    else:
        model = load_model(model_path)
    #model.summary()

    #list of callbacks:
    plotter     = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True,path= output_path, name='graph_Epoch')
    csv_logger  = CSVLogger(output_path+"csv_logger.csv")
    time_logger = TimeLogger(output_path+"time_logger.csv")
    checkpoint  = ModelCheckpoint(output_path+"Epoch.{epoch:02d}_Training_Acc.{acc:.2f}.hdf5", verbose=1, save_best_only=False)
    callbacks_list = [plotter, csv_logger, time_logger, checkpoint]

    if(IMAGE_GENERATOR):
        print("Using Image Generator")
        x_train, y_train = load_as_numpy_array(dir=training_file, type="training")
        x_dev, y_dev = load_as_numpy_array(dir=development_file, type="development")

        datagen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True,data_format="channels_last")
        #, rotation_range = 30., width_shift_range = 0.3, height_shift_range = 0.3, zoom_range = 0.3, horizontal_flip = True, vertical_flip = True
        size = int(TRAINING_FIT_RATIO*training_no_samples)
        print("Fit Size: "+str(size))
        datagen.fit(x_train[0:size])
        training_generator = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
        development_generator = datagen.flow(x_dev, y_dev, batch_size=BATCH_SIZE, shuffle=True)

    else:

        if(MODEL_MT):
            training_generator = generate_images_hdf5_mt(file=training_file, type="training", image_size= image_size)
        else:
            #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
            training_generator = generate_imges_from_hdf5(file=training_file, type="training", image_size= image_size)

        validation_generator = generate_imges_from_hdf5(file=training_file, type="validation", image_size=image_size)
        development_generator = generate_imges_from_hdf5(file=development_file,type="development",image_size=image_size)

        if(USE_VALIDATION):
            dev_val_generator = validation_generator
            dev_val_steps = validation_steps
        else:
            dev_val_generator = development_generator
            dev_val_steps = development_steps


    model.fit_generator(training_generator, verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=NB_EPOCH, validation_data = dev_val_generator, validation_steps=dev_val_steps, callbacks= callbacks_list, pickle_safe=True, workers=WORKERS)