from keras.models import load_model
import numpy as np
import h5py
from keras.callbacks import ModelCheckpoint, CSVLogger
import sys
import os
import models.MT_IM56_NODROP as mt
import models.vgg_pretrained as pretrained
import utils
from utils import AccLossPlotter, TimeLogger

DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3
WORKERS=1
NB_EPOCH = 100
BATCH_SIZE=32
training_no_samples=0
training_no_samples_mt=0
validation_no_samples=0
validation_no_samples_mt=0
validation_start=0
development_no_samples=0
test_no_samples=0
index_arr_train = np.zeros((0))
index_arr_validate = np.zeros((0))
TRAINING_FIT_RATIO= 0.1
NORMALIZE=True
VISUALIZE=False
GREYSCALE=False
SHUFFLE_BATCHES=True
USE_VALIDATION = True
TRAINING_RATIO = 0.8
VALIDATION_RATIO = 0.1

MODEL_PRETRAINED=False
MODEL_VGG16=False
MODEL_MT=False
TRAINING_RATIO_MT = 0.2
SEQUENCE_LENGTH=25
STEP=5
#To be able to visualize correctly

if USE_VALIDATION==False:
    TRAINING_RATIO = 1

if VISUALIZE:
    WORKERS = 1
    NB_EPOCH = 1

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

def set_no_samples(train_dir,dev_dir=None,test_dir=None):

    #Set number of samples to calculate: steps_per_epoch automatically
    global training_no_samples
    global training_no_samples_mt
    global validation_no_samples
    global validation_no_samples_mt
    global validation_start
    global development_no_samples
    global test_no_samples
    global index_arr_train
    global index_arr_validate

    f = h5py.File(train_dir, 'r')

    total_training_size = int(f.attrs['train_size'])
    training_no_samples = int(f.attrs['train_size'] * TRAINING_RATIO )
    print("Training file:" + train_dir)
    print("Total number of training samples: "+str(total_training_size) )
    print("Training number of samples used(and training end): "+str(training_no_samples))

    if MODEL_MT:
        index_arr_train = np.array(f['index_array_train'])
        target_index_arr_size = int((len(index_arr_train) * TRAINING_RATIO_MT))
        index_arr_train = index_arr_train[0:target_index_arr_size]

        index_arr_validate = np.array(f['index_array_validate'])
        print(index_arr_train)
        print(len(index_arr_train))

        training_no_samples_mt = return_sum_samples(index_arr_train)
        print("Number of training samples for multiple tower model: "+str(training_no_samples_mt))

        validation_no_samples_mt = return_sum_samples(index_arr_validate)
        print("Number of validation samples for multiple tower model: " + str(validation_no_samples_mt))

    if USE_VALIDATION:
        try:
            validation_no_samples = int(f.attrs['validation_size'])
            validation_start = int(f.attrs['validation_start'])
        except Exception:
            print("Validation set params not found in HDF5 file, computing according to VALIDATION_RATIO...")
            validation_no_samples = int(VALIDATION_RATIO * total_training_size )
            validation_start = total_training_size - validation_no_samples -1

        print("Validation start: " + str(validation_start))
        print("Validation number of samples: " + str(validation_no_samples))

    if (dev_dir and not USE_VALIDATION):
        f = h5py.File(dev_dir, 'r')
        print("Development file:" + dev_dir)
        development_no_samples = f.attrs['dev_size']
        print("Development number of samples: " + str(development_no_samples))

    if (test_dir):
        f = h5py.File(test_dir, 'r')
        test_no_samples = f.attrs['test_size']
        print("Test number of samples: " + str(test_no_samples))

def compute_y_mt(y):

    #Compute y_mt using the majority of labels in y
    majority = np.sum(y[:,1])
    if majority >= int(SEQUENCE_LENGTH/2):
        y_mt=np.array([0,1])
    else:
        y_mt=np.array([1,0])

    return y_mt

def prepare_mt(x,y):

    # If model is multi_tower, change batch size to (32,25,heigh,width,3) (32,2)

    length = len(x)
    length = length - (length % STEP)

    no_samples = int( ((length - SEQUENCE_LENGTH)/STEP) ) + 1
    x = x[0:length]

    x_mt = x[0:SEQUENCE_LENGTH,:,:,:]
    x_mt=x_mt.reshape((1,SEQUENCE_LENGTH, x_mt.shape[1], x_mt.shape[2], x_mt.shape[3]))

    #Compute y_mt using the majority of labels in y
    y_mt = compute_y_mt(y[0:SEQUENCE_LENGTH,:])

    for i in range(no_samples - 1):

        start = ((i+1) * STEP)
        x_mt_next = x[start:start+SEQUENCE_LENGTH,:,:,:]
        x_mt_next = x_mt_next.reshape((1, SEQUENCE_LENGTH, x_mt_next.shape[1], x_mt_next.shape[2], x_mt_next.shape[3]))
        x_mt = np.concatenate((x_mt,x_mt_next))

        # Compute y_mt using the majority of labels in y
        y_mt_next = compute_y_mt(y[start:start+SEQUENCE_LENGTH,:])
        y_mt = np.vstack((y_mt,y_mt_next))

    return x_mt,y_mt

def generate_images_hdf5_mt(file,image_size,type="training" ):

    if(type=="training"):
        validation_offset = 0
        index_arr = index_arr_train

    elif (type == "validation"):
        validation_offset= validation_start
        index_arr = index_arr_validate

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
                x_train_processed, y_train_processed = utils.preprocess(x, y, image_size=image_size,normalize=NORMALIZE,greyscale=GREYSCALE)
                x_train,y_train = prepare_mt(x_train_processed,y_train_processed)

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
                        x_train_processed, y_train_processed = utils.preprocess(x, y, image_size=image_size,normalize=NORMALIZE,greyscale=GREYSCALE)
                        x_train_next, y_train_next = prepare_mt(x_train_processed, y_train_processed)

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

def generate_imges_from_hdf5(file,image_size,type="training"):

    index=0
    offset=0

    if(type=="training"):
        index = training_no_samples

    elif (type == "validation"):
        index = validation_no_samples
        offset= validation_start

    elif(type=="development"):
        index = development_no_samples

    #Randomize which batch to get
    rand_index = np.arange(start=0, stop =index - BATCH_SIZE, step = BATCH_SIZE)

    while 1:

        if SHUFFLE_BATCHES:
            np.random.shuffle(rand_index)

        for i in range(rand_index.shape[0]):
            #Choose a random batch
            x,y = utils.load_from_hdf5(file, type=type, start=rand_index[i] + offset, end=rand_index[i] + BATCH_SIZE + offset)
            #Proprocess random batch: shuffle samples, rescale values, resize if needed
            x_train, y_train = utils.preprocess(x,y,image_size=image_size)

            #Visualize 1/8 images out of each batch
            if VISUALIZE: utils.visualize(x_train, y_train,i,type,batch_size=BATCH_SIZE,greyscale=False)

            yield (x_train, y_train)

        print("\ni is: " + str(i) +" ("+type+")")

def return_sum_samples(index_arr):

    sum_samples = 0
    for i in range(len(index_arr)):

        if (index_arr[i] >= SEQUENCE_LENGTH):

            face_track_length = index_arr[i]
            no_samples = int(((face_track_length - SEQUENCE_LENGTH) / STEP)) + 1
            sum_samples = sum_samples + no_samples

    return sum_samples

def calculate_steps_per_epoch():

    if MODEL_MT:

        steps_per_epoch_train = int(training_no_samples_mt / BATCH_SIZE)
        validation_steps = int(validation_no_samples_mt / BATCH_SIZE)
        development_steps=0

    else:

        steps_per_epoch_train = int(training_no_samples / BATCH_SIZE)
        validation_steps = int(validation_no_samples / BATCH_SIZE)
        development_steps = int(development_no_samples / BATCH_SIZE)

    return steps_per_epoch_train,validation_steps, development_steps

if __name__ == "__main__":

    training_file, development_file, model_path, image_size, BATCH_SIZE, output_path = process_arguments()

    #Set global variables
    set_no_samples(training_file, development_file)
    steps_per_epoch_train,validation_steps, development_steps = calculate_steps_per_epoch();

    utils.log_description(path=output_path,training_ratio=TRAINING_RATIO,validation_ratio=VALIDATION_RATIO,
                          model_path=model_path,dataset_path=training_file,validation_used=USE_VALIDATION,batch_size=BATCH_SIZE)

    if MODEL_MT:
        model = mt.get_model(towers_no=SEQUENCE_LENGTH)
    elif MODEL_PRETRAINED:
        model = pretrained.vgg_pretrained()
    elif MODEL_VGG16:
        model = pretrained.vgg_pretrained(weights=None,trainable=True)
    else:
        model = load_model(model_path)

    if(MODEL_MT):
        training_generator = generate_images_hdf5_mt(file=training_file, type="training", image_size= image_size)
        validation_generator =  generate_images_hdf5_mt(file=training_file, type="validation", image_size=image_size)
    else:
        #Each time, the generator returns a batch of 32 samples, each epoch represents approximately the whole training set
        training_generator = generate_imges_from_hdf5(file=training_file, type="training", image_size= image_size)
        validation_generator = generate_imges_from_hdf5(file=training_file, type="validation",image_size=image_size)

    development_generator = generate_imges_from_hdf5(file=development_file,type="development",image_size=image_size)

    if(USE_VALIDATION):
        dev_val_generator = validation_generator
        dev_val_steps = validation_steps
        percentage = utils.compute_majority_class(training_file, type="validation", start=validation_start,
                                            end=validation_start + validation_no_samples)
        print("Validation set +ve label percentage: "+str(percentage))

    else:
        dev_val_generator = development_generator
        dev_val_steps = development_steps
        percentage = utils.compute_majority_class(development_file, type="development", start=0,
                                        end=development_no_samples)
        print("Development set +ve label percentage: " + str(percentage))

    #list of callbacks:
    plotter     = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True,path= output_path, name='graph_Epoch',percentage=percentage)
    csv_logger  = CSVLogger(output_path+"csv_logger.csv")
    time_logger = TimeLogger(output_path+"time_logger.csv")
    checkpoint  = ModelCheckpoint(output_path+"Epoch.{epoch:02d}_Training_Acc.{acc:.2f}.hdf5", verbose=1, save_best_only=False)
    callbacks_list = [plotter, csv_logger, time_logger, checkpoint]

    model.fit_generator(training_generator, verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=NB_EPOCH, validation_data = dev_val_generator, validation_steps=dev_val_steps, callbacks= callbacks_list)