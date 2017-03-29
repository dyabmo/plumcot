from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from glob import glob
import cv2, numpy as np

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf

#HELP:
#https://keras.io/getting-started/sequential-model-guide/
#https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

nb_filters = 32
#Think about 224*224
nb_epoch = 10
batch_size=32

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
    # Only 2 classes: talking or not talking!
    model.add(Dense(2, activation='softmax'))

    return model

if __name__ == "__main__":


    x_fnames = glob("/vol/work1/dyab/training_set/*.Xv.npy")
    x_fnames.sort()
    print(x_fnames)
    x_arrays = [np.load(f) for f in x_fnames]
    x_train = np.concatenate(x_arrays)
    print(x_train)

    y_fnames = glob("/vol/work1/dyab/training_set/*.Y.npy")
    y_fnames.sort()
    print(y_fnames)
    y_arrays = [np.load(f) for f in y_fnames]
    y_train = np.concatenate(y_arrays)

    exit(0)
    x_train = np.loadtxt("/vol/work1/dyab/training_set/BFMTV_CultureEtVous_2012-04-16_065040.36.Xv.npy")
    y_train = np.loadtxt("/vol/work1/dyab/training_set/BFMTV_CultureEtVous_2012-04-16_065040.36.Y.npy")
    x_val = 0
    y_va = 0

    model =VGG_M()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    model.fit(x_train, y_train, verbose=1, batch_size=batch_size, epochs=nb_epoch,validation_data=(x_val, y_val))

    score = model.evaluate(x_val, y_val, verbose=0)
    print('Validation score:', score[0])
    print('Validation accuracy:', score[1])

    #For prediction
    #out = model.predict(im)
    #print np.argmax(out)