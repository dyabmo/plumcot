from keras.models import load_model
import numpy as np
from keras.utils.np_utils import to_categorical
import sys
import scipy.misc
import os
from glob import glob
import utils


DEFAULT_IMAGE_SIZE=224
IMAGE_SIZE_112 = 112
IMAGE_SIZE_56 = 56
INPUT_CHANNEL=3
BATCH_SIZE=32
IMAGE_GENERATOR=False
NORMALIZE=True
GREYSCALE=False

def process_arguments():

    assert len(sys.argv) >= 5, "Error with number of arguments: <Model Path> <Type> <Image Size> <Evaluated Dataset Path> (Test dataset) "
    assert (os.path.isdir(sys.argv[1])), "Error in models folder: folder doesn't exist."
    assert (sys.argv[2]=="validation" or sys.argv[2]=="development" or sys.argv[2]=="test" )
    assert (sys.argv[3] != 56 or sys.argv[3] != 112 or sys.argv[3] != 224), "Error in Image size: must be either 56:(56*112), 112:(112*112) or 224:(224*224)"
    assert (os.path.isfile(sys.argv[4])), "Error in Dataset: file doesn't exist."
    assert (os.path.isfile(sys.argv[5])), "Error in Dataset: file doesn't exist."

    model_path = sys.argv[1]
    type = sys.argv[2]
    image_size = int(sys.argv[3])
    evaluated_dataset_path = sys.argv[4]
    test_dataset_path = sys.argv[5]

    return model_path,type, image_size, evaluated_dataset_path, test_dataset_path

if __name__ == "__main__":

    model_path,type, image_size, evaluated_dataset_path, test_dataset_path =  process_arguments()

    print("Evaluated dataset path: " + evaluated_dataset_path)

    x, y = utils.load_as_numpy_array(evaluated_dataset_path,type = "development")
    print("Evaluation Dataset Size: " + str(x.shape[0]))
    positive_label_percentage_dev = (np.sum(y) / len(y)) * 100
    print("+VE label percentage: {:.2f}".format(positive_label_percentage_dev))

    x_val, y_val = utils.preprocess_cnn(x, y, image_size = image_size, normalize=NORMALIZE, greyscale=GREYSCALE)
    #x_val, y_val = utils.preprocess_lstm(x, y)

    if (test_dataset_path):
        print("Test dataset path: " + test_dataset_path)
        x2, y2 = utils.load_as_numpy_array(test_dataset_path, type="test")
        print("Test Dataset Size: " + str(x2.shape[0]))

        positive_label_percentage_test = (np.sum(y2) / len(y2)) * 100

        print("+VE label percentage: {:.2f}".format(positive_label_percentage_test))

        x_test, y_test = utils.preprocess_cnn(x2, y2, image_size=image_size, normalize=NORMALIZE, greyscale=GREYSCALE)

    models_all = glob(model_path+"/*.hdf5")
    models_all.sort()

    evaluation_logfile = open(model_path+"/"+type+"_log.csv",'w')
    evaluation_logfile.write("epoch,"+"dev"+"_acc,"+"dev"+"_loss,"+"dev"+"_mean_absolute_error")
    if(test_dataset_path):
        evaluation_logfile.write(","+"test"+"_acc,"+"test"+"_loss,"+"test"+"_mean_absolute_error")
    evaluation_logfile.write("\n")
    evaluation_logfile.flush()
    epoch =0

    for file in models_all:

        model = load_model(file)
        print("Model Path: " + file)
        score_dev = model.evaluate(x_val, y_val,batch_size=BATCH_SIZE, verbose=1)
        evaluation_logfile.write(str(int(epoch)) + "," + str(score_dev[1]) + "," + str(score_dev[0]) + "," + str(score_dev[2]))

        if (test_dataset_path):
            score_test = model.evaluate(x_test, y_test,batch_size=BATCH_SIZE, verbose=1)
            evaluation_logfile.write(","+str(score_test[1])+","+str(score_test[0])+","+str(score_test[2]))

        evaluation_logfile.write("\n")
        evaluation_logfile.flush()
        epoch = epoch + 1




