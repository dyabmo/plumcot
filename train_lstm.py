from pyannote.generators.batch import batchify
from models import stacked_lstm
from models import optimizers
import os
import sys
import utils
import numpy as np

NB_EPOCH = 100
BATCH_SIZE=32
INPUT_DIMS=40
SEQUENCE_LENGTH=25
STEP = 5
USE_VALIDATION = True
TRAINING_RATIO = 0.8
VALIDATION_RATIO = 0.1
FACETRACK_RATIO = 1

def process_arguments():

    assert len(sys.argv) == 5, "Error with number of arguments: <training set path> <development set path> <BATCH_SIZE> <output_path>."
    assert (os.path.isfile(sys.argv[1])), "Error in training set: file doesn't exist."
    #assert (os.path.isfile(sys.argv[2])), "Error in development set: file doesn't exist."
    assert (int(sys.argv[3]) % 2 == 0), "Error in batch size."
    assert (not os.path.isdir(sys.argv[4])), "Error in output folder: folder already exists, can't overwrite."

    training_file = sys.argv[1]
    development_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    output_path = sys.argv[4]
    if (output_path[-1] != "/"):
        output_path = output_path + "/"

    os.mkdir(output_path)

    return training_file,development_file, batch_size, output_path

#yield only one sequence each time, then use batchify..
#TODO: Generalize on sequences of images too
def lstm_generator(file,type,validation_start, index_arr_train,index_arr_validate):

    if(type=="training"):
        validation_offset = 0
        index_arr = index_arr_train

    elif (type == "validation"):
        validation_offset= validation_start
        index_arr = index_arr_validate

    elif (type == "development"):
        raise NotImplementedError("NOT IMPLEMENTED")

    #Generator has to loop forever for keras
    while 1:

        facetrack_index = 0
        # Only facetracks of length bigger than SEQUENCE_LENGTH will be used
        while (index_arr[facetrack_index] >=SEQUENCE_LENGTH) and (facetrack_index < len(index_arr)):

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

    training_file, development_file, BATCH_SIZE, output_path = process_arguments()

    _, training_sequence_no_samples, validation_no_samples, validation_sequence_no_samples, \
    validation_start, development_no_samples, development_sequence_no_samples, index_arr_train, index_arr_validate  = utils.set_no_samples(training_file,development_file,True,USE_VALIDATION,TRAINING_RATIO,VALIDATION_RATIO,SEQUENCE_LENGTH,STEP,FACETRACK_RATIO)

# create batch generator
    signature = ({'type': 'ndarray'}, {'type': 'ndarray'})

    training_generator = lstm_generator(training_file,"training",validation_start,index_arr_train,index_arr_validate)
    batch_training_generator = batchify(training_generator, signature, batch_size=BATCH_SIZE)

    validation_generator = lstm_generator(training_file, "validation",validation_start, index_arr_train, index_arr_validate)
    batch_validation_generator = batchify(validation_generator, signature, batch_size=BATCH_SIZE)

    development_generator = lstm_generator(development_file, "development",validation_start, index_arr_train, index_arr_validate)
    batch_development_generator = batchify(development_generator, signature, batch_size=BATCH_SIZE)

    steps_per_epoch_train, validation_steps, development_steps = \
        utils.calculate_steps_per_epoch(training_sequence_no_samples, validation_sequence_no_samples, development_sequence_no_samples,
                                        batch_size=BATCH_SIZE,
                                        image_generator=False, image_generator_factor=0);

    if (USE_VALIDATION):
        dev_val_generator = batch_validation_generator
        dev_val_steps = validation_steps
        percentage = utils.compute_majority_class(training_file, type="validation", start=validation_start,
                                                  end=validation_start + validation_no_samples)
        print("Validation set +ve label percentage: " + str(percentage))

    else:
        dev_val_generator = batch_development_generator
        dev_val_steps = development_steps
        percentage = utils.compute_majority_class(development_file, type="development", start=0,
                                                  end=development_no_samples)
        print("Development set +ve label percentage: " + str(percentage))

    model_callable = stacked_lstm.StackedLSTM()
    model = model_callable(input_shape = (SEQUENCE_LENGTH, INPUT_DIMS) )

    callbacks_list = utils.get_callabcks_list(output_path=output_path, percentage=percentage)
    utils.log_description(model=model, path=output_path, training_ratio=TRAINING_RATIO,
                          validation_ratio=VALIDATION_RATIO,
                          model_path="Stacked LSTM", dataset_path=training_file, validation_used=USE_VALIDATION,
                          batch_size=BATCH_SIZE)

    optimizer = optimizers.SSMORMS3()
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])

    model.fit_generator(batch_training_generator, verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=NB_EPOCH, validation_data = dev_val_generator, validation_steps=dev_val_steps, callbacks= callbacks_list)