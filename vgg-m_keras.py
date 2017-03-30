from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from glob import glob
import cv2, numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf

#HELP:
#https://keras.io/getting-started/sequential-model-guide/
#https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

#Think about 224*224
input_shape=(128,128,3)
nb_epoch = 2
batch_size=16

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

if __name__ == "__main__":

    x_dataset,y_dataset = load_from_pickle('/vol/work1/dyab/training_set/sample_training.pickle')
    x_train, x_val, y_train, y_val = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=0)

    model =VGG_M()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])
    model.fit(x_train, y_train, verbose=1, batch_size=batch_size, epochs=nb_epoch,validation_data=(x_val, y_val))

    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation Accuracy:' + str(score[0]))
    print('Validation Mean Square error:'+str( score[1]))

    #For prediction
    #out = model.predict(im)
    #print np.argmax(out)