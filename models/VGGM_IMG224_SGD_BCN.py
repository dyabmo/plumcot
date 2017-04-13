from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras import optimizers

INPUT_WIDTH=224
INPUT_HEIGHT=224
INPUT_CHANNEL=3
input_shape=(INPUT_WIDTH,INPUT_HEIGHT,INPUT_CHANNEL)

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf
#Params: http://www.robots.ox.ac.uk/%7Evgg/publications/2016/Chung16/chung16.pdf

def VGG_M():
    model = Sequential()

    #First conv layer
    model.add(Conv2D(filters=96, kernel_size=(7, 7), activation='relu',strides=(2, 2), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3) , strides=2))

    #Second layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu',strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

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
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    #6th layer, fully connected with dropout
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    # 7th layer, fully connected with dropout
    #Makes sense ?
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    #Softmax layer
    #Only 1 value: talking or not talking!
    model.add(Dense(1, activation='softmax'))

    return model

def save_model():

    model =VGG_M()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    model.save('/vol/work1/dyab/training_models/VGGM_IMG224_SGD_BCN.h5')
    model.summary()

if __name__ == "__main__":

    save_model()