from pyannote.generators.batch import batchify
from models import stacked_lstm
from models import optimizers
import keras
import os
import sys
import utils
import numpy as np
from keras.models import load_model

NB_EPOCH = 100
BATCH_SIZE=32
INPUT_DIMS=40
SEQUENCE_LENGTH=25
STEP = 2
USE_VALIDATION = False
TRAINING_RATIO = 0.8
VALIDATION_RATIO = 0.5

#TODO: Count number of sequential samples

def process_arguments():

    assert len(sys.argv) == 6, "Error with number of arguments: <training set path> <development set path> <BATCH_SIZE> <Optimizer> <output_path>."
    assert (os.path.isfile(sys.argv[1])), "Error in training set: file doesn't exist."
    assert (os.path.isfile(sys.argv[2])), "Error in development set: file doesn't exist."
    assert (int(sys.argv[3]) % 2 == 0), "Error in batch size."
    assert (sys.argv[4]== val for val in ["ssmorm3","rmsprop","adadelta","adagrad","adam","adamax","sgd" ] )
    assert (not os.path.isdir(sys.argv[5])), "Error in output folder: folder already exists, can't overwrite."

    training_file = sys.argv[1]
    development_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    optimizer_name = sys.argv[4]
    output_path = sys.argv[5]
    if (output_path[-1] != "/"):
        output_path = output_path + "/"

    os.mkdir(output_path)

    return training_file,development_file, batch_size,optimizer_name, output_path

#yield only one sequence each time, then use batchify..
#TODO: Generalize on sequences of images too
def lstm_generator(file,type,validation_start, index_arr_train_dev,index_arr_validate):

    if(type=="training"):
        validation_offset = 0
        index_arr = index_arr_train_dev

    elif (type == "validation"):
        validation_offset= validation_start
        index_arr = index_arr_validate

    elif (type == "development"):
        index_arr = index_arr_train_dev

    #Generator has to loop forever for keras
    while 1:

        facetrack_index = 0
        # Only facetracks of length bigger than SEQUENCE_LENGTH will be used
        while (index_arr[facetrack_index] >=SEQUENCE_LENGTH) and (facetrack_index < len(index_arr) - 1):

            start = np.sum(index_arr[0:facetrack_index]) #will return zero if facetrack_index is zero
            end = np.sum(index_arr[0:facetrack_index + 1])

            if (type == "validation"):
                start = start + validation_offset
                end = end + validation_offset

            # load the concened facetrack
            x, y = utils.load_from_hdf5(file, type=type,start=start, end=end)

            # preprocess the facetrack
            x_processed, y_processed = utils.preprocess_lstm(x, y)

            #Group facetrack samples as sequences
            x_train , y_train = utils.sequence_samples(x_processed, y_processed,sequence_length=SEQUENCE_LENGTH, step=STEP,seq2seq=True)

            #Yield one sequence only each time
            for item_x,item_y in zip(x_train,y_train):
                yield item_x,item_y

            #After yield is finished, go to next facetrack
            facetrack_index = facetrack_index + 1

if __name__ == "__main__":

    training_file, development_file, BATCH_SIZE,optimizer_name, output_path = process_arguments()

    training_no_samples, training_sequence_no_samples, validation_no_samples, validation_sequence_no_samples, \
    validation_start, development_no_samples, development_sequence_no_samples, index_arr_train, index_arr_validate,index_array_dev  = utils.set_no_samples(training_file,development_file,True,USE_VALIDATION,TRAINING_RATIO,VALIDATION_RATIO,SEQUENCE_LENGTH,STEP)

# create batch generator
    signature = ({'type': 'ndarray'}, {'type': 'ndarray'})

    training_generator = lstm_generator(training_file,"training",validation_start,index_arr_train,index_arr_validate)
    batch_training_generator = batchify(training_generator, signature, batch_size=BATCH_SIZE)

    validation_generator = lstm_generator(training_file, "validation",validation_start, index_arr_train, index_arr_validate)
    batch_validation_generator = batchify(validation_generator, signature, batch_size=BATCH_SIZE)

    development_generator = lstm_generator(development_file, "development",validation_start, index_array_dev, None)
    batch_development_generator = batchify(development_generator, signature, batch_size=BATCH_SIZE)

    steps_per_epoch_train, validation_steps, development_steps = \
        utils.calculate_steps_per_epoch(training_sequence_no_samples, validation_sequence_no_samples, development_sequence_no_samples,
                                        batch_size=BATCH_SIZE,
                                        image_generator=False, image_generator_factor=0);

    training_percentage = utils.compute_samples_majority_class(training_file, type="training", start=0,end=training_no_samples)
    print("Training set +ve label percentage: " + str(training_percentage))

    if (USE_VALIDATION):
        dev_val_generator = batch_validation_generator
        dev_val_steps = validation_steps
        percentage = utils.compute_samples_majority_class(training_file, type="validation", start=validation_start,
                                                          end=validation_start + validation_no_samples)
        print("Validation set +ve label percentage: " + str(percentage))

    else:
        dev_val_generator = batch_development_generator
        dev_val_steps = development_steps
        percentage = utils.compute_samples_majority_class(development_file, type="development", start=0, end=development_no_samples)
        print("Development set +ve label percentage: " + str(percentage))

    model_callable = stacked_lstm.StackedLSTM(lstm=[128,128],mlp=[128])
    model = model_callable(input_shape = (SEQUENCE_LENGTH, INPUT_DIMS) )


    if(optimizer_name=="ssmorm3"):
        optimizer = optimizers.SSMORMS3()
    elif(optimizer_name=="rmsprop"):
        optimizer = keras.optimizers.RMSprop()
    elif(optimizer_name=="adam"):
        optimizer = keras.optimizers.adam()
    elif (optimizer_name == "adadelta"):
        optimizer = keras.optimizers.adadelta()
    elif (optimizer_name == "adamax"):
        optimizer = keras.optimizers.adamax()
    elif (optimizer_name == "sgd"):
        optimizer = keras.optimizers.sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])

    callbacks_list = utils.get_callabcks_list(output_path=output_path, percentage=percentage,training_percentage = training_percentage)
    utils.log_description(model=model, path=output_path, training_ratio=TRAINING_RATIO,
                          validation_ratio=VALIDATION_RATIO,
                          model_path="Stacked LSTM", dataset_path=training_file, validation_used=USE_VALIDATION,
                          batch_size=BATCH_SIZE,optimizer=optimizer_name)

    model.fit_generator(batch_training_generator, verbose=1, steps_per_epoch=steps_per_epoch_train//80, epochs=NB_EPOCH, validation_data = dev_val_generator, validation_steps=dev_val_steps, callbacks= callbacks_list)

    #model.load_weights("/vol/work1/dyab/training_models/lstm_ssmorm3_layers_2_128_step_2_tiny_epoch/Epoch.93_Training_Acc.1.00.hdf5")
    #score_dev = model.evaluate_generator(batch_development_generator, development_steps)
    #print("dev"+"_acc,"+"dev"+"_loss,"+"dev"+"_mean_absolute_error")
    #print(str(score_dev[1]) + "," + str(score_dev[0]) + "," + str(score_dev[2]))

