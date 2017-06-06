from pyannote.generators.batch import batchify
from models import stacked_lstm
from models import optimizers
import keras
import os
import sys
import utils
import numpy as np
from keras.models import load_model
from glob import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import h5py
from matplotlib import colors
import random
from matplotlib.pyplot import cm

def visualize_model_weights_change(output_path):

    models_all = glob(output_path + "/*.hdf5")
    models_all.sort()

    colors_names = list(colors.cnames)

    epoch=-1
    previous_weights = list()
    mean_list_epochs = [list() for i in range(10)]

    for model_weight in models_all:
        epoch+=1

        f = h5py.File(model_weight)['model_weights']

        #print(f.attrs.items())
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
        weight_names_list = list()
        weight_values_list = list()

        #Get weight names
        for name in layer_names:
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            weight_names_list.append(weight_names)

            weight_values = [g[weight_name] for weight_name in weight_names]
            weight_values_list.append(weight_values)

        #Flatten the lists
        weight_names_list = [ val for sublist in weight_names_list for val in sublist ]
        weight_values_list =  [ val for sublist in weight_values_list for val in sublist ]

        #Get weight values
        model=load_model(model_weight, ({'SSMORMS3':optimizers.SSMORMS3}) )

        weights = model.get_weights()

        if(epoch >=1):
            mean_list = list()
            for i in range(len(weights)):
                #print(weights[i].shape)
                #update the same weight across different epochs
                mean_list_epochs[i].append(np.mean(abs(weights[i] - previous_weights[i] )))

            epochs = np.arange(epoch)

            [plt.plot(epochs, mean_list_epochs[i], color=colors_names[i]) for i in range(len(weight_names_list))  ]

            patch_list = [ mpatches.Patch(color =colors_names[i], label = weight_names_list[i] ) for i in range(len(weight_names_list))   ]
            plt.legend(handles=patch_list, loc=4)

            plt.savefig(output_path+'model_weights_test2.png')
        previous_weights = weights


if __name__ == "__main__":

    visualize_model_weights_change("/vol/work1/dyab/training_models/lstm_ssmorm3_default_step_2_dev_videos_last_derivative_graph")