from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from glob import glob
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils.io_utils import HDF5Matrix
import h5py
from keras.callbacks import ModelCheckpoint, CSVLogger
from visual_callbacks import AccLossPlotter

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf
#Params: http://www.robots.ox.ac.uk/%7Evgg/publications/2016/Chung16/chung16.pdf

INPUT_WIDTH=224
INPUT_HEIGHT=224
INPUT_CHANNEL=3
input_shape=(INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNEL)
nb_epoch = 200
batch_size=32
number_of_nodes=8
steps_per_epoch_train=2300
validation_steps = 580

def VGG_M():
    model = Sequential()
    #ignoring normalization
    #First conv layer
    model.add(Conv2D(filters=96, kernel_size=(7, 7), activation='relu',strides=(2, 2), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Second layer
    model.add(Conv2D(filters=256, kernel_size=(7, 7), activation='relu',strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Third layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu',strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1)))
    #no max pooling

    #Fourth layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1)))

    # Fifth layer
    model.add(Conv2D(filters=512, kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #6th layer, fully connected with dropout
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 7th layer, fully connected with dropout
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    #Softmax layer
    #Only 1 value: talking or not talking!
    model.add(Dense(1, activation='softmax'))

    return model

def save_to_pickle(dir,output_file):

    x_fnames = glob(dir+"/*.Xv.npy")
    x_fnames.sort()
    print(x_fnames)
    x_arrays = [np.load(f) for f in x_fnames]
    x_train = np.concatenate(x_arrays)
    print(x_train.shape)

    y_fnames = glob(dir+"/*.Y.npy")
    y_fnames.sort()
    print(y_fnames)
    y_arrays = [np.load(f) for f in y_fnames]
    y_train = np.concatenate(y_arrays)
    print(y_train.shape)

    with open(output_file, 'wb') as f:
        data = x_train, y_train
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_from_pickle(dir):

    with open(dir, 'rb') as f:
        x_train, y_train = pickle.load(f)
    return x_train, y_train

def validate_dataset_shape(dir):

    #validate_dataset_shape('/vol/work1/dyab/training_set')

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

def save_to_hdf5(dir,output_file):

    #save_to_hdf5('/vol/work1/dyab/training_set/sample_dataset','sample.h5')

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
    print(y_dataset.shape)
    print(x_dataset.shape)

    x_train, x_val, y_train, y_val = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

    f = h5py.File(dir+"/"+output_file, 'w')
    # Creating dataset to store features
    f.create_dataset('training_input',data=x_train)
    f.create_dataset('training_validation_input', data=x_val)

    # Creating dataset to store labels
    f.create_dataset('training_labels', data=y_train)
    f.create_dataset('training_validation_labels', data=y_val)
    f.flush()
    f.close()

def load_from_hdf5(dir,type):

    X_train, y_train = 0,0

    if(type=="training"):
        X_train = HDF5Matrix(dir, 'training_input')
        y_train = HDF5Matrix(dir, 'training_labels')

    elif (type == "training_validation"):
        X_train = HDF5Matrix(dir, 'training_validation_input')
        y_train = HDF5Matrix(dir, 'training_validation_labels')

    elif(type=="development"):
        X_train = HDF5Matrix(dir, 'development_input')
        y_train = HDF5Matrix(dir, 'development_labels')

    elif (type == "test"):
        X_train = HDF5Matrix(dir, 'test_input')
        y_train = HDF5Matrix(dir, 'test_labels')

    return X_train, y_train

def generate_training_images(file):

    while 1:

        x_dataset, y_dataset = load_from_pickle(file)
        x_train, _, y_train, _ = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)
        print(x_train.shape[0])

        for i in range(0,x_train.shape[0],batch_size):
            yield (x_train[i:i+batch_size], y_train[i:i+batch_size])

def generate_training_images_hdf5(file):

    while 1:
        x_train, y_train = load_from_hdf5(file,type="training")
        print(x_train.shape[0])

        for i in range(0,x_train.shape[0],batch_size):
            yield (x_train[i:i+batch_size], y_train[i:i+batch_size])

def generate_validation_images(file):

    while 1:
        x_dataset, y_dataset = load_from_pickle(file)
        _, x_val, _, y_val = train_test_split(x_dataset, y_dataset, test_size=0.1, random_state=0)

        for i in range(0, x_val.shape[0], batch_size):
            yield (x_val[i:i+batch_size], y_val[i:i+batch_size])

def generate_validation_images_hdf5(file):

    while 1:
        x_val, y_val = load_from_pickle(file,type="development")

        for i in range(0, x_val.shape[0], batch_size):
            yield (x_val[i:i+batch_size], y_val[i:i+batch_size])

def generate_imges_from_hdf5(file,type="training"):

    while 1:
        x, y = load_from_hdf5(file,type=type)
        print(x.shape[0])

        for i in range(0,x.shape[0],batch_size):
            yield (x[i:i+batch_size], y[i:i+batch_size])

if __name__ == "__main__":

    training_path="/vol/work1/dyab/training_set/"
    file_name="/medium_train_val_dataset.h5"
    input_file = training_path+file_name

    model =VGG_M()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    #list of callbacks:
    plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True,name='medium_graph')
    csv_logger = CSVLogger('/vol/work1/dyab/training_models/medium_training.log')
    checkpoint = ModelCheckpoint("/vol/work1/dyab/training_models/medium_weights.{epoch:02d}-{loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [csv_logger,checkpoint,plotter]

    #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
    model.fit_generator(generate_imges_from_hdf5(input_file,type="training"),verbose=2, steps_per_epoch=steps_per_epoch_train, epochs=nb_epoch, validation_data =generate_imges_from_hdf5(input_file,type="training_validation"),validation_steps=validation_steps ,callbacks= callbacks_list,pickle_safe=True, workers=number_of_nodes)

    score = model.evaluate_generator(generate_imges_from_hdf5(input_file,type="development"), steps=2, pickle_safe=True, workers=number_of_nodes)
    print('Validation Accuracy:' + str(score[0]))
    print('Validation Mean Square error:'+str( score[1]))
