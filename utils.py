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
from keras.callbacks import ModelCheckpoint, CSVLogger
import sys
from glob import glob


DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3

def log_description(model,path,training_ratio,validation_ratio,model_path,dataset_path,validation_used,batch_size=32,data_augmentation=False,optimizer=None):

    description = open(path+"/description" ,'w')
    description.write("Training set used: "+str(dataset_path)+"\n")
    description.write("Training size: {:.1f}%\n".format(training_ratio*100))
    if(validation_used):
        description.write("Validation set is used. Size: {:.1f}%\n".format(validation_ratio * 100))
    else:
        description.write("Development set is used\n")
    description.write(("Batch size: {}\n".format(batch_size)))

    if(data_augmentation):
        description.write("Data Augmentation is used\n")

    description.write("Optimizer: {}".format(optimizer))
    description.write("===============================================\n")
    description.write("Model used: "+str(model_path)+"\n")
    description.write("Model Summary:\n")
    description.flush()

    #redirect output of model.summary() to description file
    sys.stdout = open(path+"/description" , "a")
    model.summary()
    sys.stdout = sys.__stdout__
    #end of redirection

    description.flush()
    description.close()

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

    def __init__(self, graphs=['acc', 'loss'], save_graph=True,path='/vol/work1/dyab/training_models',name='graph_Epoch',percentage=0,training_percentage=0):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_graph
        self.name = name
        self.path = path
        self.percentage = percentage
        self.training_percentage = training_percentage

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

            red_patch = mpatches.Patch(color='red', label='Val. +ve label {:.2f}%'.format(self.percentage) )
            blue_patch = mpatches.Patch(color='blue', label='Train +ve label {:.2f}%'.format(self.training_percentage) )

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if 'loss' in self.graphs:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Loss')
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Validation')
            blue_patch = mpatches.Patch(color='blue', label='Training')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if self.save_graph:
            plt.savefig(self.path+'/'+self.name+'.png')

    def on_train_end(self, logs={}):
        if self.save_graph:
            plt.savefig(self.path+'/'+self.name+'.png')

def compute_samples_majority_class(dir, type='validation', start=0, end=None):

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


def load_as_numpy_array(dir,type=None,validation_start=0,validation_size=None):

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

    else:
        x_dataset = np.array(file['input'])
        y_dataset = np.array(file['labels'])

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
        plt.savefig("/vol/work1/dyab/samples_visualization/Delete/batch_" + str(i) + ".png")
    elif(type == "development"):
        plt.savefig("/vol/work1/dyab/samples_visualization/cluster_eval/batch_" + str(i) + ".png")
    elif (type == "validation"):
        plt.savefig("/vol/work1/dyab/samples_visualization/validation_training_shuffled_samples_shuffled_videos/batch_" + str(i) + ".png")

def visualize_mt(input_list, y_train_batch, i, type):
    raise NotImplementedError("Visualization not implemented yet")

def compute_y_mt(y,sequence_length):

    #Compute y_mt using the majority of labels in y
    majority = np.sum(y[:,1])
    if majority >= int(sequence_length/2):
        y_mt=np.array([0,1])
    else:
        y_mt=np.array([1,0])

    return y_mt

def consume_generator(generator):

    x_val, y_val = list(), list()
    for x, y in generator:
        x_val.append(x)
        y_val.append(y)

    # Flatten the list (because of batches)
    x_val = [val for sublist in x_val for val in sublist]
    y_val = [val for sublist in y_val for val in sublist]

    x_np = np.array(x_val)
    y_np = np.array(y_val)

    return x_np,y_np

def sequence_samples(x, y, sequence_length, step, seq2seq):

    # Group samples in form of sequences, change:
    #   From: (size, x.shape[0], x.shape[1], ...) (size,2)
    #   To:   (size/sequence_length, sequence_length, x.shape[0], x.shape[1], ...)  (size/sequence_length, sequence_length , 2)

    #create generic reshaping
    new_shape_list = [1,sequence_length]
    for i in range(1,x.ndim):
        new_shape_list.append(x.shape[i])

    #create new shape tuple
    new_shape_x = tuple(new_shape_list)
    new_shape_y = (1,sequence_length,2)

    #To represent samples as sequences, get rid of the modulus of the step
    length = len(x)

    length = length - ( (length % sequence_length) % step)

    #Trim list to be equal to calculated length
    x = x[0:length]
    y = y[0:length]

    #create list of indices that they numpy array will be split on
    new_indices = np.arange(0, length, sequence_length)

    #Incorporate the step size in the sequence indices
    new_indices_list = [(new_indices + z) for z in range(0, sequence_length, step)]

    #Actually split the array according to the indices
    #will return empty lists for exceeding length .. that can be removed later
    new_seq_x = [np.split(x, new_indices) for new_indices in new_indices_list]
    new_seq_y = [np.split(y, new_indices) for new_indices in new_indices_list]

    #Flatten the list
    new_seq_x = [val for sublist in new_seq_x for val in sublist]
    new_seq_y = [val for sublist in new_seq_y for val in sublist]

    #remove arrays of length that is less than sequence length, those are the boundaries of splitting
    new_seq_x = list(filter(lambda x: len(x) >= sequence_length, new_seq_x))
    new_seq_y = list(filter(lambda y: len(y) >= sequence_length, new_seq_y))

    #Only reshape and concatenate arrays if they are more than one ...
    if(len(new_seq_x) > 1 and len(new_seq_y) > 1):
        #Actually reshape each sequence
        new_seq_x = [x.reshape(new_shape_x) for x in new_seq_x]
        new_seq_y = [y.reshape(new_shape_y) for y in new_seq_y]

        new_seq_x = np.vstack(new_seq_x)
        new_seq_y = np.vstack(new_seq_y)

    else:
        new_seq_x = np.array(new_seq_x)
        new_seq_y = np.array(new_seq_y)

    #If one output label is needed for the sequence, instead of a sequence of outputs
    if(not seq2seq):
        raise NotImplementedError("Not implemented")
        #Compute y_mt using the majority of labels in y
        #y_mt = compute_y_mt(y[0:sequence_length,:],sequence_length=sequence_length)

    return new_seq_x,new_seq_y

def seq2eq(seq_y,sequence_length):

    for i in range(len(seq_y)):
        clean_value = np.zeros((sequence_length, 2))
        majority = np.sum(seq_y[i, :, 1])
        if majority >= int(sequence_length / 2):
            clean_value[:, 1] = 1
        else:
            clean_value[:, 0] = 1

        seq_y[i] = clean_value

    return seq_y

def preprocess_lstm(x,y,normalize=False,first_derivative=False,second_derivative=False):

    # Convert to numpy array
    x_np = np.array(x)
    y_np = np.array(y)

    #Change y to categorical
    y_train = to_categorical(y_np, num_classes=2)

    if normalize:
        raise NotImplementedError("Not implemented")

    if first_derivative:

        x_np_delta = np.zeros((x_np.shape[0],x_np.shape[1]*2))
        for i in range(len(x_np)):

            delta_x = np.zeros(x_np.shape[1])
            # ignore first x, first y and last x, last y
            for j in range(2,x_np.shape[1]-2):

                #step = 2 because: x,y,x,y,x,y,x,y
                delta_x[j] = x_np[i][j+2] - x_np[i][j-2]

            x_np_delta[i] = np.concatenate((x_np[i],delta_x))

        #second derivative is computed as a result of the first
        if second_derivative:

            x_np_delta2 = np.zeros((x_np.shape[0], x_np.shape[1] * 3))
            for i in range(len(x_np)):
                delta2_x = np.zeros(x_np.shape[1])

                # ignore first and second x, first and second y / last and before last x, last and before last y
                for j in range(4, x_np.shape[1] - 4):

                    #step = 2 because: x,y,x,y,x,y,x,y
                    #40 because we are computing using 1st derivative input
                    delta2_x[j] = x_np_delta[i][40 + j + 2] - x_np_delta[i][40 + j - 2]

                x_np_delta2[i] = np.concatenate((x_np_delta[i], delta2_x))

            return x_np_delta2,y_train

        #If second derivative is false, return value from 1st derivative
        else:
            return x_np_delta, y_train

    #If not 1st derivative, just return the values
    else:
        return x_np,y_train

#yield only one sequence each time, then use batchify..
#TODO: Generalize on sequences of images too
def lstm_generator(file,type,validation_start, index_arr_train_dev,index_arr_validate,sequence_length=25,step=2,first_derivative=True,second_derivative=True,forever=True):

    if(type=="training"):
        validation_offset = 0
        index_arr = index_arr_train_dev

    elif (type == "validation"):
        validation_offset= validation_start
        index_arr = index_arr_validate

    elif (type == "development"):
        index_arr = index_arr_train_dev

    #Generator has to loop forever for keras
    first_loop = True
    while forever or first_loop:

        facetrack_index = 0
        # Only facetracks of length bigger than SEQUENCE_LENGTH will be used
        while (facetrack_index < len(index_arr) - 1):

            if (index_arr[facetrack_index] >= sequence_length):

                start = np.sum(index_arr[0:facetrack_index]) #will return zero if facetrack_index is zero
                end = np.sum(index_arr[0:facetrack_index + 1])

                if (type == "validation"):
                    start = start + validation_offset
                    end = end + validation_offset

                # load the concened facetrack
                x, y = load_from_hdf5(file, type=type,start=start, end=end)

                # preprocess the facetrack
                x_processed, y_processed = preprocess_lstm(x, y,first_derivative=first_derivative,second_derivative=second_derivative)

                #Group facetrack samples as sequences
                x_train , y_train = sequence_samples(x_processed, y_processed,sequence_length=sequence_length, step=step,seq2seq=True)

                #Yield one sequence only each time
                for item_x,item_y in zip(x_train,y_train):
                    #import matplotlib.animation as animation
                    #visualize_mouth(item_x,item_y)
                    yield item_x,item_y

            #Go to next facetrack
            facetrack_index = facetrack_index + 1

        first_loop=False

def preprocess_cnn(x, y, image_size=DEFAULT_IMAGE_SIZE, normalize=True, greyscale=False, flatten=False):
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

    # Perform simple normalization
    if normalize:
        x_np_temp = np.divide(x_np_temp, 255.0)

    #Change to greyscale if needed
    if greyscale:
        x_np_temp = rgb2grey(x_np_temp)

    if flatten:
        x_np_temp = np.vstack([x.flatten() for x in x_np_temp])

    #Change y to categorical
    y_train = to_categorical(y_np, num_classes=2)
    return x_np_temp,y_train

def random_shuffle_subset( x_train,ratio=1):

    size = x_train.shape[0]
    # Shuffle inplace
    np.random.shuffle(x_train)

    subset = int(size*ratio)
    x_subset = x_train[0: subset]

    return x_subset

#Return number of samples for specific sequence_length and step based on facetracks size
def return_sequence_size(index_arr, sequence_length, step):

    sum_samples = 0
    for i in range(len(index_arr)):

        if (index_arr[i] >= sequence_length):

            face_track_length = index_arr[i]
            no_samples = int(((face_track_length - sequence_length) / step)) + 1
            sum_samples = sum_samples + no_samples

    return sum_samples

#TODO: Divide it into two versions, according to: either sue individual samples or sequences of samples.. use facetrack ratio instead of training ratio
def set_no_samples(train_dir,dev_dir,use_seq_model,use_validation,training_ratio,validation_ratio,sequence_length,step):

    #Set number of samples to calculate: steps_per_epoch automatically
    training_no_samples = 0
    training_sequence_no_samples = 0
    validation_no_samples = 0
    validation_sequence_no_samples = 0
    validation_start = 0
    development_no_samples = 0
    development_sequence_no_samples = 0
    index_arr_train = np.zeros((0))
    index_arr_validate = np.zeros((0))
    index_array_dev = np.zeros((0))

    f = h5py.File(train_dir, 'r')
    print("Training file:" + train_dir)

    if not use_seq_model:
        total_training_size = int(f.attrs['train_size'])
        print("Total number of training samples: "+str(total_training_size) )

        training_no_samples = int(f.attrs['train_size'] * training_ratio )
        print("Training number of samples used(and training end): "+str(training_no_samples))

        if use_validation:
            try:
                validation_no_samples = int(f.attrs['validation_size'])
                validation_start = int(f.attrs['validation_start'])
            except Exception:
                print("Validation set params not found in HDF5 file, computing according to VALIDATION_RATIO...")
                validation_no_samples = int(validation_ratio * total_training_size)
                validation_start = total_training_size - validation_no_samples - 1
                print("Validation start: " + str(validation_start))
                print("Validation number of samples: " + str(validation_no_samples))

        elif not use_validation:
            f = h5py.File(dev_dir, 'r')
            print("Development file:" + dev_dir)
            development_no_samples = f.attrs['dev_size']
            print("Development number of samples: " + str(development_no_samples))

    elif use_seq_model:

        if(use_validation):
            #set the training set without validation part
            index_arr_train = np.array(f['index_array_train'])

            index_arr_validate = np.array(f['index_array_validate'])
            total_validation_size = np.sum(index_arr_validate)
            print("Total number of validation samples: "+str(total_validation_size) )

            validation_index_arr_size = int((len(index_arr_validate) * validation_ratio))
            index_arr_validate = index_arr_validate[0:validation_index_arr_size]
            validation_no_samples = np.sum(index_arr_validate)
            print("Validation number of samples used: " + str(validation_no_samples))

            validation_sequence_no_samples = return_sequence_size(index_arr_validate, sequence_length=sequence_length, step=step)
            print("Number of validation samples for sequence based samples: " + str(validation_sequence_no_samples))

            validation_start = int(f.attrs['validation_start'])
            print("Validation start: " + str(validation_start))

        elif not use_validation:

            #use the whole training set including validation part, since dev set is going to be used
            index_arr_train = np.array(f['index_array'])

            f = h5py.File(dev_dir, 'r')
            print("Development file:" + dev_dir)
            development_no_samples = f.attrs['dev_size']
            print("Total number of development samples: "+str(development_no_samples) )
            print("Development number of samples: " + str(development_no_samples))
            index_array_dev = np.array(f['index_array'])

            development_sequence_no_samples = return_sequence_size(index_array_dev, sequence_length=sequence_length, step=step)
            print("Number of Developmet samples for sequence based samples: " + str(development_sequence_no_samples))

        #set variables related to training set
        total_training_size = np.sum(index_arr_train)
        print("Total number of training samples: "+str(total_training_size) )

        target_index_arr_size = int((len(index_arr_train) * training_ratio))
        index_arr_train = index_arr_train[0:target_index_arr_size]
        training_no_samples = np.sum(index_arr_train)
        print("Training number of samples used(and training end): " + str(training_no_samples))

        training_sequence_no_samples = return_sequence_size(index_arr_train, sequence_length=sequence_length, step=step)
        print("Number of training samples for sequence based samples: " + str(training_sequence_no_samples))

    return training_no_samples, training_sequence_no_samples, validation_no_samples, validation_sequence_no_samples, \
           validation_start, development_no_samples, development_sequence_no_samples, index_arr_train, index_arr_validate, index_array_dev

def calculate_steps_per_epoch(training_samples,validation_samples,development_samples,batch_size=32, image_generator=False,image_generator_factor=1):

    steps_per_epoch_train = int(training_samples / batch_size )
    validation_steps = int(validation_samples / batch_size)
    development_steps = int(development_samples / batch_size)

    if image_generator:
        steps_per_epoch_train*=image_generator_factor

    return steps_per_epoch_train,validation_steps, development_steps

def get_callabcks_list(output_path,percentage,training_percentage):
    # list of callbacks:
    plotter = AccLossPlotter(graphs=['acc', 'loss'], path=output_path, percentage=percentage,training_percentage=training_percentage)
    csv_logger = CSVLogger(output_path + "csv_logger.csv")
    time_logger = TimeLogger(output_path + "time_logger.csv")
    checkpoint = ModelCheckpoint(output_path + "Epoch.{epoch:02d}_Training_Acc.{acc:.2f}.hdf5", verbose=1)
    return [plotter, csv_logger, time_logger, checkpoint]

def save_sequences(output_path, generator_train, generator_val):

    models_all = glob(output_path + "/*.hdf5")
    models_all.sort()

    x_val,y_val = consume_generator(generator_val)
    x_train, y_train = consume_generator(generator_train)

    f1 = h5py.File("lstm_trainingset_complete.hdf5", 'w')
    f2 = h5py.File("lstm_validationset_complete.hdf5", 'w')

    f1.attrs['size'] = x_train.shape[0]
    # Creating dataset to store features
    f1.create_dataset('input', data=x_train)
    # Creating dataset to store labels
    f1.create_dataset('labels', data=y_train)

    f2.attrs['size'] = x_val.shape[0]
    # Creating dataset to store features
    f2.create_dataset('input', data=x_val)
    # Creating dataset to store labels
    f2.create_dataset('labels', data=y_val)

    f1.flush()
    f1.close()

    f2.flush()
    f2.close()