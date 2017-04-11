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
from visual_callbacks import AccLossPlotter, ConfusionMatrixPlotter

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf
#Params: http://www.robots.ox.ac.uk/%7Evgg/publications/2016/Chung16/chung16.pdf

INPUT_WIDTH=224
INPUT_HEIGHT=224
INPUT_CHANNEL=3
input_shape=(INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNEL)
nb_epoch = 100
batch_size=32
number_of_nodes=12
training_no_samples=0
training_validation_no_samples=0
development_no_samples=0
test_no_samples=0

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
    print(x_dataset.shape[0])

    train_ratio = 0.8
    train_size = int(x_dataset.shape[0] * train_ratio)

    x_train = x_dataset[0:train_size]
    y_train = y_dataset[0:train_size]

    x_val = x_dataset[train_size+1 : x_dataset.shape[0] ]
    y_val = y_dataset[train_size+1 : x_dataset.shape[0] ]

    del x_dataset

    print(x_train.shape)
    print(y_train.shape)
    print(x_val.shape)
    print(y_val.shape)

    f = h5py.File(dir+"/"+output_file, 'w')
    f.attrs['train_size'] = x_train.shape[0]
    f.attrs['val_size'] = x_val.shape[0]
    # Creating dataset to store features
    f.create_dataset('training_input',data=x_train)

    f.create_dataset('training_validation_input', data=x_val)

    # Creating dataset to store labels
    f.create_dataset('training_labels', data=y_train)

    f.create_dataset('training_validation_labels', data=y_val)

    f.flush()
    f.close()

def set_no_samples(dir):

    #Set number of samples to calculate: steps_per_epoch automatically
    global training_no_samples
    global training_validation_no_samples
    global development_no_samples
    global test_no_samples

    f = h5py.File(dir, 'r')
    training_no_samples = f.attrs['train_size']
    training_validation_no_samples = f.attrs['val_size']

    print(training_no_samples)
    print(training_validation_no_samples)

    #y_develop= HDF5Matrix(dir, 'development_labels')
    #development_no_samples = y_develop.shape[0]

    #y_test= HDF5Matrix(dir, 'test_labels')
    #test_no_samples = y_test.shape[0]

def load_from_hdf5(dir,type,start=0,end=None):

    X_train, y_train = 0,0

    if(type=="training"):
        X_train = HDF5Matrix(dir, 'training_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'training_labels',start=start,end=end)

    elif (type == "training_validation"):
        X_train = HDF5Matrix(dir, 'training_validation_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'training_validation_labels',start=start,end=end)

    elif(type=="development"):
        X_train = HDF5Matrix(dir, 'development_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'development_labels',start=start,end=end)

    elif (type == "test"):
        X_train = HDF5Matrix(dir, 'test_input',start=start,end=end)
        y_train = HDF5Matrix(dir, 'test_labels',start=start,end=end)

    return X_train,y_train

def random_shuffle_2_arrays(X_train,y_train):

    index = np.arange(X_train.shape[0])
    #Shuffle inplace
    np.random.shuffle(index)

    X_train=X_train[index]
    y_train = y_train[index]

    return X_train, y_train

def generate_imges_from_hdf5(file,type="training"):

    while 1:

        for i in range(0,training_no_samples-batch_size,batch_size):
            x, y = load_from_hdf5(file, type=type,start=i,end=i+batch_size)
            yield (x, y)

def calculate_steps_per_epoch():

    steps_per_epoch_train = int(training_no_samples/batch_size)
    training_validation_steps = int(training_validation_no_samples/batch_size)
    development_steps = int(development_no_samples / batch_size )

    return steps_per_epoch_train, training_validation_steps,development_steps

if __name__ == "__main__":

    training_path="/vol/work1/dyab/training_set/"
    training_file_name = "/train_val_dataset.h5"
    input_file = training_path + training_file_name

    development_path="/vol/work1/dyab/development_set/"
    development_file_name="/develop_dataset.h5"
    development_file = development_path+development_file_name

    output_path="/vol/work1/dyab/training_models/GPU_BS-32_DA-NO_EP100_OP-SGD/"

    #Set global variables
    set_no_samples(input_file)
    steps_per_epoch_train, training_validation_steps, development_steps = calculate_steps_per_epoch();

    model =VGG_M()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    #list of callbacks:
    plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True,path= output_path, name='graph_Epoch.')
    csv_logger = CSVLogger(output_path+"csv_logger")
    checkpoint = ModelCheckpoint(output_path+"Epoch.{epoch:02d}_Val_Acc.{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [csv_logger,checkpoint,plotter]

    #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
    training_generator = generate_imges_from_hdf5(input_file,type="training")
    training_validation_generator = generate_imges_from_hdf5(input_file,type="training_validation")

    model.fit_generator(training_generator,verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=nb_epoch, validation_data =training_validation_generator,validation_steps=training_validation_steps ,callbacks= callbacks_list)

    exit(0)
    development_generator = generate_imges_from_hdf5(development_file,type="development")
    score = model.evaluate_generator(development_generator, steps=development_steps)

    #Plot confusion matrix
    X_val,y_val =  load_from_hdf5(development_file, "development")
    ConfusionMatrixPlotter(X_val=X_val, classes=("Talking","Not Talking"), Y_val=y_val,path=output_path)

    print('Validation Accuracy:' + str(score[0]))
    print('Validation Mean Square error:'+str( score[1]))