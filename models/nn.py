from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D, BatchNormalization
from keras import optimizers

INPUT_HEIGHT=56
INPUT_WIDTH=112
INPUT_CHANNEL=1
input_shape=(INPUT_HEIGHT,INPUT_WIDTH,INPUT_CHANNEL)

#Using VGG_M presented here:
#   https://gist.github.com/ksimonyan/f194575702fae63b2829#file-readme-md
#   https://arxiv.org/pdf/1405.3531.pdf

#Params: http://www.robots.ox.ac.uk/%7Evgg/publications/2016/Chung16/chung16.pdf

def VGG_M():
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu',kernel_initializer= "he_normal"))

    #model.add(Dense(1024, activation='relu',kernel_initializer= "he_normal"))


    #Softmax layer
    #2 classes: talking or not talking!
    model.add(Dense(2, activation='softmax'))

    return model

def save_model():

    model =VGG_M()
    sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc', 'mae'])

    model.save('/vol/work1/dyab/training_models/models/NN_128.h5')
    model.summary()

if __name__ == "__main__":

    save_model()
