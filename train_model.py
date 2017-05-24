from keras.models import load_model
import numpy as np
import h5py
import sys
import os
import models.MT_IM56_NODROP as mt
import models.vgg_pretrained as pretrained
import utils
from keras.preprocessing.image import ImageDataGenerator

DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3
NB_EPOCH = 100
BATCH_SIZE=32
IMAGE_GENERATOR = False
IMAGE_GENERATOR_FACTOR = 5
NORMALIZE=True
VISUALIZE=False
GREYSCALE=False
FLATTEN=False
USE_VALIDATION = True
TRAINING_RATIO = 0.8
VALIDATION_RATIO = 0.1

MODEL_PRETRAINED=False
MODEL_VGG16=False
MODEL_MT=False
SEQUENCE_LENGTH=25
STEP=5
#To be able to visualize correctly

if USE_VALIDATION==False:
    TRAINING_RATIO = 1

if VISUALIZE:
    NB_EPOCH = 1
    GREYSCALE = False
    IMAGE_GENERATOR=False
    TRAINING_RATIO = 0.001
    VALIDATION_RATIO = 0.1

def process_arguments():

    assert len(sys.argv) == 7, "Error with number of arguments: <training set path> <development set path>  <model path> <image size> <BATCH_SIZE> <output_path>."
    assert (os.path.isfile(sys.argv[1])), "Error in training set: file doesn't exist."
    assert (os.path.isfile(sys.argv[2])), "Error in development set: file doesn't exist."
    assert (os.path.isfile(sys.argv[3])), "Error in model: file doesn't exist."
    assert (sys.argv[4] != 56 or sys.argv[4] != 112 or sys.argv[4] != 224), "Error in Image size: must be either 56:(56*112), 112:(112*112) or 224:(224*224)"
    assert (int(sys.argv[5]) % 2 == 0), "Error in batch size."
    assert (not os.path.isdir(sys.argv[6])), "Error in output folder: folder already exists, can't overwrite."

    training_file = sys.argv[1]
    development_file = sys.argv[2]
    model_path = sys.argv[3]
    image_size = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    output_path = sys.argv[6]
    if (output_path[-1] != "/"):
        output_path = output_path + "/"

    os.mkdir(output_path)

    return training_file,development_file, model_path, image_size, batch_size, output_path

def generate_images_hdf5_mt(file,image_size,type,validation_start,index_arr_train,index_arr_validate):

    if(type=="training"):
        validation_offset = 0
        index_arr = index_arr_train

    elif (type == "validation"):
        validation_offset= validation_start
        index_arr = index_arr_validate

    elif (type == "development"):
        raise NotImplementedError("NOT IMPLEMENTED")

    while 1:

        i = 0
        remaining_x_train = np.zeros((0))
        remaining_y_train = np.zeros((0))
        remaining_samples = 0

        while i < (len(index_arr) -1 ):
            #print("\ni is :"+str(i))
            #check if currrent facetrack's length is bigger than SEQUENCE_LENGTH or not
            #Only facetracks of length bigger than SEQUENCE_LENGTH will be used
            if index_arr[i] < SEQUENCE_LENGTH:
                i=i+1

            else:
                #Only in the initial case
                if i==0:
                    start=0
                    end = index_arr[i]
                elif i==1:
                    start = index_arr[0]
                    end = index_arr[0] + index_arr[1]
                else:
                    start= np.sum(index_arr[0:i])
                    end =  np.sum(index_arr[0:i+1])

                if (type == "validation"):
                    start = start + validation_offset
                    end = end + validation_offset

                #Do once at the beginning
                x, y = utils.load_from_hdf5(file, type=type, start=start, end=end)
                x_train_processed, y_train_processed = utils.preprocess_cnn(x, y, image_size=image_size, normalize=NORMALIZE, greyscale=GREYSCALE)
                x_train,y_train = utils.sequence_samples(x_train_processed, y_train_processed, sequence_length=SEQUENCE_LENGTH, step = STEP,seq2seq=False)

                #concatenate remaining samples from previous iteration if they exist
                if(remaining_samples>0):
                    x_train = np.concatenate((remaining_x_train, x_train))
                    y_train = np.vstack((remaining_y_train, y_train))
                    remaining_samples=0

                i = i + 1
                #Keep doing until batch size is reached
                while len(x_train) < BATCH_SIZE and i < (len(index_arr) -1 ):

                    #print("Entered while loop")

                    if index_arr[i] < SEQUENCE_LENGTH:
                        i = i + 1
                        #print("Entered the If")

                    else:
                        #print("Entered the else")

                        #i can be at least 1, so this case must be handled
                        if i == 1:
                            start = index_arr[0]
                            end = index_arr[0] + index_arr[1]
                        else:
                            start = np.sum(index_arr[0:i])
                            end = np.sum(index_arr[0:i+1])

                        if (type == "validation"):
                            start = start + validation_offset
                            end = end + validation_offset

                        x, y = utils.load_from_hdf5(file, type=type, start=start, end=end)
                        x_train_processed, y_train_processed = utils.preprocess_cnn(x, y, image_size=image_size, normalize=NORMALIZE, greyscale=GREYSCALE)
                        x_train_next, y_train_next = utils.sequence_samples(x_train_processed, y_train_processed, sequence_length=SEQUENCE_LENGTH, step=STEP,seq2seq=False)

                        x_train = np.concatenate((x_train,x_train_next))
                        y_train = np.vstack((y_train,y_train_next))

                        i = i + 1

                length = len(x_train)
                batches = int(length / BATCH_SIZE)

                #use the remaining samples for next batch
                #only get them if the previous ones were consumed
                if remaining_samples==0:
                    remaining_samples = length % BATCH_SIZE

                if remaining_samples > 0:
                    remaining_x_train = x_train[ length-remaining_samples : length ,:,:,:,:]
                    remaining_y_train = y_train[ length-remaining_samples : length ,:]

                #print(batches)
                for j in range(batches):

                    #yield each batch:
                    x_train_batch = x_train[j*BATCH_SIZE:j*BATCH_SIZE + BATCH_SIZE ,:,:,:,:]
                    y_train_batch = y_train[j * BATCH_SIZE:j * BATCH_SIZE + BATCH_SIZE, :]

                    input_list=list()
                    for k in range(SEQUENCE_LENGTH):
                        input_list.append(x_train_batch[:,k,:,:,:])

                    if VISUALIZE: utils.visualize_mt(input_list, y_train_batch, i, type)

                    yield (input_list, y_train_batch)

def generate_imges_from_hdf5(file,image_size,type,training_no_samples,validation_no_samples,validation_start,development_no_samples):

    index=0
    offset=0

    if(type=="training"):
        index = training_no_samples

        if IMAGE_GENERATOR:
            datagen = ImageDataGenerator(rotation_range=20., width_shift_range=0.3, height_shift_range=0.3,
                                         zoom_range=0.3,shear_range=0.2, horizontal_flip=False, vertical_flip=False,
                                         data_format="channels_last")

    elif (type == "validation"):
        index = validation_no_samples
        offset= validation_start

    elif(type=="development"):
        index = development_no_samples

    #Randomize which batch to get
    rand_index = np.arange(start=0, stop =index - BATCH_SIZE, step = BATCH_SIZE)

    while 1:

        #Shuffle batches each epoch
        np.random.shuffle(rand_index)

        for i in range(rand_index.shape[0]):
            #Choose a random batch
            x,y = utils.load_from_hdf5(file, type=type, start=rand_index[i] + offset, end=rand_index[i] + BATCH_SIZE + offset)
            #Proprocess random batch: shuffle samples, rescale values, resize if needed
            x_train, y_train = utils.preprocess_cnn(x, y, image_size=image_size, normalize=NORMALIZE, greyscale=GREYSCALE, flatten=FLATTEN)

            #Visualize 1/8 images out of each batch
            if VISUALIZE: utils.visualize(x_train, y_train,i,type,batch_size=BATCH_SIZE,greyscale=False)

            if IMAGE_GENERATOR and type=="training":
                j=-1
                for x_train_datagen,y_train_datagen in datagen.flow(x_train,y_train, batch_size=BATCH_SIZE):
                    j+=1
                    yield (x_train_datagen,y_train_datagen)

                    if j == IMAGE_GENERATOR_FACTOR-1:
                        break;

            else:
                yield (x_train, y_train)

        print("\ni is: " + str(i) +" ("+type+")")

if __name__ == "__main__":

    training_file, development_file, model_path, image_size, BATCH_SIZE, output_path = process_arguments()

    training_no_samples, training_no_samples_mt, validation_no_samples, validation_no_samples_mt, \
    validation_start, development_no_samples, development_no_samples_mt, index_arr_train, index_arr_validate,index_array_dev = utils.set_no_samples(training_file, development_file,MODEL_MT,
                                                                                                   USE_VALIDATION,TRAINING_RATIO,VALIDATION_RATIO,SEQUENCE_LENGTH,STEP)

    if MODEL_MT:
        model = mt.get_model(towers_no=SEQUENCE_LENGTH)
    elif MODEL_PRETRAINED:
        model = pretrained.vgg_pretrained()
    elif MODEL_VGG16:
        model = pretrained.vgg_pretrained(weights=None,trainable=True)
    else:
        model = load_model(model_path)

    if(MODEL_MT):
        steps_per_epoch_train, validation_steps, development_steps = \
            utils.calculate_steps_per_epoch(training_no_samples_mt,validation_no_samples_mt,development_no_samples_mt,batch_size=BATCH_SIZE,
                                      image_generator=IMAGE_GENERATOR,image_generator_factor=IMAGE_GENERATOR_FACTOR);

        training_generator = generate_images_hdf5_mt(training_file, image_size, "training",validation_start, index_arr_train,index_arr_validate)
        validation_generator =  generate_images_hdf5_mt(training_file, image_size, "validation",validation_start, index_arr_train,index_arr_validate)
        development_generator = generate_images_hdf5_mt(training_file, image_size, "development",validation_start, index_arr_train,index_arr_validate)
    else:
        steps_per_epoch_train, validation_steps, development_steps = \
            utils.calculate_steps_per_epoch(training_no_samples, validation_no_samples, development_no_samples,batch_size=BATCH_SIZE,
                                      image_generator=IMAGE_GENERATOR, image_generator_factor=IMAGE_GENERATOR_FACTOR);
        #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
        training_generator = generate_imges_from_hdf5(training_file, image_size, "training",training_no_samples, validation_no_samples, validation_start, development_no_samples)
        validation_generator = generate_imges_from_hdf5(training_file, image_size,"validation",training_no_samples, validation_no_samples, validation_start, development_no_samples)
        development_generator = generate_imges_from_hdf5(development_file, image_size,"development",training_no_samples, validation_no_samples, validation_start, development_no_samples)



    if(USE_VALIDATION):
        dev_val_generator = validation_generator
        dev_val_steps = validation_steps
        percentage = utils.compute_samples_majority_class(training_file, type="validation", start=validation_start,
                                                          end=validation_start + validation_no_samples)
        print("Validation set +ve label percentage: "+str(percentage))

    else:
        dev_val_generator = development_generator
        dev_val_steps = development_steps
        percentage = utils.compute_samples_majority_class(development_file, type="development", start=0,
                                                          end=development_no_samples)
        print("Development set +ve label percentage: " + str(percentage))

    callbacks_list = utils.get_callabcks_list(output_path=output_path,percentage=percentage)
    utils.log_description(model=model,path=output_path,training_ratio=TRAINING_RATIO,validation_ratio=VALIDATION_RATIO,
                          model_path=model_path,dataset_path=training_file,validation_used=USE_VALIDATION,batch_size=BATCH_SIZE,data_augmentation=IMAGE_GENERATOR)

    model.fit_generator(training_generator, verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=NB_EPOCH, validation_data = dev_val_generator, validation_steps=dev_val_steps, callbacks= callbacks_list)