import numpy as np
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import utils


def process_arguments():

    assert (os.path.isdir(sys.argv[1])), "Error in models folder: folder doesn't exist."

    model_path = sys.argv[1]

    return model_path

if __name__ == "__main__":

    model_path =  process_arguments()

    dev_percentage = utils.compute_majority_class("/vol/work1/dyab/development_set/develop_dataset_local.h5", type="development")
    test_percentage = utils.compute_majority_class("/vol/work1/dyab/test_set/test_dataset.h5", type="test")

    evaluation_logger = np.loadtxt(model_path+"/development_log.csv",delimiter=",",skiprows=1)
    epochs = evaluation_logger[:,0]
    stop_epoch=len(epochs)
    dev_acc = list(evaluation_logger[0:stop_epoch,1])
    dev_loss = list(evaluation_logger[0:stop_epoch, 2])
    test_acc = list(evaluation_logger[0:stop_epoch, 4])
    test_loss = list(evaluation_logger[0:stop_epoch, 5])


    csv_logger = np.loadtxt(model_path+"/csv_logger.csv",delimiter=",",skiprows=1)
    train_acc = list(csv_logger[0:stop_epoch,1])
    train_loss = list(csv_logger[0:stop_epoch, 2])

    val_acc = list(csv_logger[0:stop_epoch, 4])
    val_loss = list(csv_logger[0:stop_epoch, 5])


    plt.ioff()
    plt.subplot(2, 1, 1)
    plt.title('Accuracy')
    # plt.axis([0,100,0,1])
    plt.plot(epochs, train_acc, color='r')
    plt.plot(epochs, val_acc, color='b')
    plt.plot(epochs, dev_acc, color='g')
    plt.plot(epochs, test_acc, color='y')
    plt.ylabel('accuracy')

    red_patch = mpatches.Patch(color='red', label='Train')
    blue_patch = mpatches.Patch(color='blue',label='Val: 73.29%')
    green_patch = mpatches.Patch(color='green', label='Dev: {:.2f}%'.format(dev_percentage))
    yellow_patch = mpatches.Patch(color='yellow', label='Test: {:.2f}%'.format(test_percentage))

    plt.legend(handles=[red_patch, blue_patch,green_patch,yellow_patch], loc=4)

    plt.subplot(2, 1, 2)
    plt.title('Loss')
    # plt.axis([0,100,0,5])
    plt.plot(epochs, train_loss, color='r')
    plt.plot(epochs, val_loss, color='b')
    plt.plot(epochs, dev_loss, color='g')
    plt.plot(epochs, test_loss, color='y')
    plt.ylabel('loss')

    red_patch = mpatches.Patch(color='red', label='Train')
    blue_patch = mpatches.Patch(color='blue',label='Val: 73.29%')
    green_patch = mpatches.Patch(color='green', label='Dev: {:.2f}%'.format(dev_percentage))
    yellow_patch = mpatches.Patch(color='yellow', label='Test: {:.2f}%'.format(test_percentage))

    plt.legend(handles=[red_patch, blue_patch,green_patch,yellow_patch], loc=4)

    plt.savefig(model_path+"/evaluation_graph.png")