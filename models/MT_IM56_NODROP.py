from keras.models import Sequential, save_model
from keras.layers import Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization, Merge
from keras.layers.merge import Concatenate
from keras import optimizers

INPUT_HEIGHT=56
INPUT_WIDTH=112
INPUT_CHANNEL=3
input_shape=(INPUT_HEIGHT,INPUT_WIDTH,INPUT_CHANNEL)

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf

#Params: http://www.robots.ox.ac.uk/%7Evgg/publications/2016/Chung16/chung16.pdf

def add_tower():

    model = Sequential()
    # First conv layer
    model.add(Conv2D(filters=45, kernel_initializer="he_normal", kernel_size=(3, 3), activation='relu', strides=(2, 2),
                      input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    return model

def get_model():

    towers=list()
    for i in range(25):
        model = add_tower()
        towers.append(model)

    model = Sequential()
    model.add(Merge(towers, mode='concat'))

    #add conv1d layer
    model.add(Conv2D(filters=96,kernel_size=(1,1),activation='relu'))

    #Second layer
    model.add(Conv2D(filters=256,kernel_initializer="he_normal", kernel_size=(3, 3), activation='relu',strides=(2, 2)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    #Third layer
    model.add(Conv2D(filters=512,kernel_initializer="he_normal", kernel_size=(3, 3), activation='relu',strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1)))
    #no max pooling

    #Fourth layer
    model.add(Conv2D(filters=512,kernel_initializer="he_normal", kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1)))

    # Fifth layer
    model.add(Conv2D(filters=512,kernel_initializer="he_normal", kernel_size=(3, 3), activation='relu', strides=(1, 1)))
    model.add(ZeroPadding2D((1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    #6th layer, fully connected
    model.add(Flatten())
    model.add(Dense(4096,kernel_initializer="he_normal", activation='relu'))

    # 7th layer, fully connected
    model.add(Dense(128,kernel_initializer="he_normal", activation='relu'))

    #Softmax layer
    #2 classes: talking or not talking!
    model.add(Dense(2, activation='softmax'))

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    return model

def save_model_():

    model =get_model()

    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    model.save('/vol/work1/dyab/training_models/models/MT_IM56_NODROP_INIT.h5')
    print(model.inputs)
    model.summary()


if __name__ == "__main__":

    save_model_()