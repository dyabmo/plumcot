import numpy as np
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import timeit
from keras.callbacks import Callback
from keras.utils.io_utils import HDF5Matrix
import scipy.misc
from keras.utils.np_utils import to_categorical

DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3

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

#Copied from: https://github.com/chasingbob/keras-visuals, with some modifications
class AccLossPlotter(Callback):
    #Plot training Accuracy and Loss values on a Matplotlib graph.

    def __init__(self, graphs=['acc', 'loss'], save_graph=False,path='/vol/work1/dyab/training_models',name='training_acc_loss',percentage=0):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph
        self.name = name
        self.path = path
        self.percentage = percentage

    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0
        plt.ioff()


    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0

        if 'acc' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accuracy')
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')

            red_patch = mpatches.Patch(color='red', label='Val. +ve label '+str('{:.2f}'.format(self.percentage)) +'%' )
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if 'loss' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Loss')
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Val. +ve label '+str('{:.2f}'.format(self.percentage))+'%' )
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if self.save_graph:
            plt.savefig(self.path+'/'+self.name+'.png')

    def on_train_end(self, logs={}):
        if self.save_graph:
            plt.savefig(self.path+'/'+self.name+'.png')

def compute_majority_class(dir,type='validation',start=0,end=None):

    _, y=load_from_hdf5(dir, type, start=start, end=end, labels_only=True)

    positive_label_percentage =  (np.sum(y) / len(y)) * 100

    return positive_label_percentage

def load_from_hdf5(dir,type,start=0,end=None,labels_only=False):

    X_train, y_train = 0,0

    if(type=="training" or type == "validation"):

        if(labels_only):
            y_train = HDF5Matrix(dir, 'training_labels', start=start, end=end)
        else:
            X_train = HDF5Matrix(dir, 'training_input',start=start,end=end)
            y_train = HDF5Matrix(dir, 'training_labels',start=start,end=end)

    elif(type=="development"):
        if(labels_only):
            y_train = HDF5Matrix(dir, 'development_labels', start=start, end=end)
        else:
            X_train = HDF5Matrix(dir, 'development_input',start=start,end=end)
            y_train = HDF5Matrix(dir, 'development_labels',start=start,end=end)

    elif (type == "test"):
        if(labels_only):
            y_train = HDF5Matrix(dir, 'test_labels', start=start, end=end)
        else:
            X_train = HDF5Matrix(dir, 'test_input',start=start,end=end)
            y_train = HDF5Matrix(dir, 'test_labels',start=start,end=end)

    return X_train,y_train


def load_as_numpy_array(dir,type,validation_start=0,validation_size=None):

    x_dataset,y_dataset = np.empty((0)),np.empty((0))
    file = h5py.File(dir, 'r')  # 'r' means that hdf5 file is open in read-only mode

    if (type == "validation"):

        if( not validation_size ):
            raise Exception("Must set validation_size")
        else:
            x_dataset = np.array(file['training_input'][validation_start: validation_start + validation_size])
            y_dataset = np.array(file['training_labels'][validation_start: validation_start + validation_size])

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

# Change to greyscale
def rgb2grey(x):

    r, g, b = x[ : , : , : , 0 ] , x[ : , : , : , 1 ], x[ : , : , : , 2 ]
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grey_reshaped = grey.reshape((x.shape[0],x.shape[1],x.shape[2],1))

    return grey_reshaped

def visualize(x_train,y_train,i,type,batch_size,greyscale):

    index = batch_size // 8
    for j in range(0, index):
        plt.subplot(index // 2, index // 2, j + 1)

        image = x_train[j * 8]

        if greyscale:
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

def visualize_mt(input_list, y_train_batch, i, type):
    raise NotImplementedError("Visualization not implemented yet")

def preprocess(x,y,image_size=DEFAULT_IMAGE_SIZE,normalize=True,greyscale=False):
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
    if normalize:
        x_train = np.divide(x_train, 255.0)

    #Change to greyscale if needed
    if greyscale:
        x_train = rgb2grey(x_train)

    #Change y to categorical
    y_train = to_categorical(y_train, num_classes=2)
    return x_train,y_train

def random_shuffle_subset( x_train,ratio=1):

    size = x_train.shape[0]
    # Shuffle inplace
    np.random.shuffle(x_train)

    subset = int(size*ratio)
    x_subset = x_train[0: subset]

    return x_subset