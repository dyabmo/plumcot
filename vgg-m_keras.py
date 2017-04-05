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
from keras.callbacks import ModelCheckpoint

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf
#Params: http://www.robots.ox.ac.uk/%7Evgg/publications/2016/Chung16/chung16.pdf

input_shape=(224,224,3)
nb_epoch = 30
batch_size=32
number_of_nodes=4

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

def save_to_hdf5(dir,output_file):

    #save_to_hdf5('/vol/work1/dyab/training_set/sample_dataset','sample.h5')

    #Get x
    x_fnames = glob(dir + "/*.Xv.npy")
    x_fnames.sort()
    x_arrays = [np.load(f) for f in x_fnames]
    x_train = np.concatenate(x_arrays)
    print(x_train.shape)

    #Get y
    y_fnames = glob(dir + "/*.Y.npy")
    y_fnames.sort()
    y_arrays = [np.load(f) for f in y_fnames]
    y_train = np.concatenate(y_arrays)
    print(y_train.shape)

    f = h5py.File(dir+"/"+output_file, 'w')
    # Creating dataset to store features
    X_dset = f.create_dataset('training_input',data=x_train)

    # Creating dataset to store labels
    y_dset = f.create_dataset('training_labels', data=y_train)
    f.flush()
    f.close()

def load_from_hdf5(dir,type):

    X_train, y_train = 0,0

    if(type=="training"):
        X_train = HDF5Matrix(dir, 'training_input')
        y_train = HDF5Matrix(dir, 'training_labels')

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

if __name__ == "__main__":

    x_train,y_train=load_from_hdf5('/vol/work1/dyab/training_set/sample_dataset/sample.h5',type="training")
    steps_per_epoch_train = int(x_train.shape[0]/batch_size)

    model =VGG_M()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    # checkpoint
    filepath = "/vol/work1/dyab/training_models/weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
    model.fit_generator(generate_training_images_hdf5('/vol/work1/dyab/training_set/sample_dataset/sample.h5'), steps_per_epoch=steps_per_epoch_train, epochs=nb_epoch, callbacks= callbacks_list,pickle_safe=True, workers=number_of_nodes)

    score = model.evaluate_generator(generate_validation_images_hdf5('/vol/work1/dyab/training_set/sample_dataset/sample.h5'), steps=2, pickle_safe=True, workers=number_of_nodes)
    print('Validation Accuracy:' + str(score[0]))
    print('Validation Mean Square error:'+str( score[1]))
