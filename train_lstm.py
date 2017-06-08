from pyannote.generators.batch import batchify
from pyannote.audio.labeling.models import StackedLSTM
from pyannote.audio.optimizers import SSMORMS3
import keras
import os
import sys
import utils

NB_EPOCH = 200
BATCH_SIZE=32
FIRST_DERIVATIVE=True
SECOND_DERIVATIVE=True
INPUT_DIMS=40
SEQUENCE_LENGTH=25
STEP = 2
TRAINING_RATIO = 0.8
VALIDATION_RATIO = 1
USE_VALIDATION = True
SAVE_SEQUENCES=False
DEBUG=False

if FIRST_DERIVATIVE:
    INPUT_DIMS=INPUT_DIMS + 40

if SECOND_DERIVATIVE:
    INPUT_DIMS=INPUT_DIMS + 40

if DEBUG:
    TRAINING_RATIO = 0.01
    VALIDATION_RATIO = 0.5
    NB_EPOCH = 3

def process_arguments():

    assert len(sys.argv) == 6, "Error with number of arguments: <training set path> <development set path> <BATCH_SIZE> <Optimizer> <output_path>."
    assert (os.path.isfile(sys.argv[1])), "Error in training set: file doesn't exist."
    assert (os.path.isfile(sys.argv[2])), "Error in development set: file doesn't exist."
    assert (int(sys.argv[3]) % 2 == 0), "Error in batch size."
    assert (sys.argv[4]== val for val in ["ssmorm3","rmsprop","adadelta","adagrad","adam","adamax","sgd" ] )

    if not SAVE_SEQUENCES:
        assert (not os.path.isdir(sys.argv[5])), "Error in output folder: folder already exists, can't overwrite."

    training_file = sys.argv[1]
    development_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    optimizer_name = sys.argv[4]
    output_path = sys.argv[5]
    if (output_path[-1] != "/"):
        output_path = output_path + "/"

    if not SAVE_SEQUENCES:
        os.mkdir(output_path)

    return training_file,development_file, batch_size,optimizer_name, output_path

if __name__ == "__main__":

    training_file, development_file, BATCH_SIZE,optimizer_name, output_path = process_arguments()

    training_no_samples, training_sequence_no_samples, validation_no_samples, validation_sequence_no_samples, \
    validation_start, development_no_samples, development_sequence_no_samples, index_arr_train, index_arr_validate,index_array_dev  = utils.set_no_samples(training_file,development_file,True,USE_VALIDATION,TRAINING_RATIO,VALIDATION_RATIO,SEQUENCE_LENGTH,STEP)

    # create batch generator
    signature = ({'type': 'ndarray'}, {'type': 'ndarray'})

    training_generator = utils.lstm_generator(training_file,"training",validation_start,index_arr_train,index_arr_validate,SEQUENCE_LENGTH,STEP,FIRST_DERIVATIVE,SECOND_DERIVATIVE)
    batch_training_generator = batchify(training_generator, signature, batch_size=BATCH_SIZE)

    steps_per_epoch_train, _, _ = utils.calculate_steps_per_epoch(training_sequence_no_samples, 0, 0, batch_size=BATCH_SIZE);

    training_percentage = utils.compute_samples_majority_class(training_file, type="training", start=0,end=training_no_samples)
    print("Training set +ve label percentage: " + str(training_percentage))

    if SAVE_SEQUENCES:
        validation_generator = utils.lstm_generator(training_file, "validation", validation_start, index_arr_train,
                                                    index_arr_validate, SEQUENCE_LENGTH, STEP, FIRST_DERIVATIVE,
                                                    SECOND_DERIVATIVE,forever=False)
        batch_validation_generator = batchify(validation_generator, signature, batch_size=BATCH_SIZE)

        training_generator = utils.lstm_generator(training_file, "training", validation_start, index_arr_train,
                                                  index_arr_validate, SEQUENCE_LENGTH, STEP, FIRST_DERIVATIVE,
                                                  SECOND_DERIVATIVE,forever=False)
        batch_training_generator = batchify(training_generator, signature, batch_size=BATCH_SIZE)

        utils.save_sequences(output_path, batch_training_generator, batch_validation_generator)

    #Train
    else:

        if (USE_VALIDATION):
            validation_generator = utils.lstm_generator(training_file, "validation", validation_start, index_arr_train,
                                                  index_arr_validate,SEQUENCE_LENGTH,STEP,FIRST_DERIVATIVE,SECOND_DERIVATIVE,forever=False)
            batch_validation_generator = batchify(validation_generator, signature, batch_size=BATCH_SIZE)
            x_val, y_val = utils.consume_generator(batch_validation_generator)
            percentage = utils.compute_samples_majority_class(training_file, type="validation", start=validation_start,end=validation_start + validation_no_samples)
            print("Validation set +ve label percentage: " + str(percentage))

        else:
            development_generator = utils.lstm_generator(development_file, "development", validation_start, index_array_dev, None,SEQUENCE_LENGTH,STEP,FIRST_DERIVATIVE,SECOND_DERIVATIVE,forever=False)
            batch_development_generator = batchify(development_generator, signature, batch_size=BATCH_SIZE)
            x_val, y_val = utils.consume_generator(batch_development_generator)
            percentage = utils.compute_samples_majority_class(development_file, type="development", start=0, end=development_no_samples)
            print("Development set +ve label percentage: " + str(percentage))

        model_callable = StackedLSTM()
        model = model_callable(input_shape = (SEQUENCE_LENGTH, INPUT_DIMS) )

        if(optimizer_name=="ssmorm3"):
            optimizer = SSMORMS3()
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

        model.fit_generator(batch_training_generator, verbose=1, steps_per_epoch=steps_per_epoch_train, epochs=NB_EPOCH, validation_data = (x_val,y_val), callbacks= callbacks_list)