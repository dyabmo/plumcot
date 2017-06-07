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
from sklearn.metrics import roc_auc_score
from models import optimizers
import os

EPOCH_INTERVAL = 5

#TODO: Implement as separate graph for each layer or each array of weights.
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

def plot_distributions(models_path,output_path,x_train,y_train,x_val,y_val):

    models_all = glob(models_path + "/*.hdf5")
    models_all.sort()


    #shape: (None,25,2)
    positive_boolean_index_train = y_train[:,:,1] == 1
    positive_boolean_index_val = y_val[:,:,1] == 1

    #print(positive_boolean_index.shape)
    #print(positive_boolean_index)

    negative_boolean_index_train = y_train[:, :, 1] == 0
    negative_boolean_index_val = y_val[:, :, 1] == 0

    #print(negative_boolean_index)

    epoch = -1
    i=1
    auc_train_list=list()
    auc_val_list=list()
    for model_weight in models_all:
        epoch+=1

        try:
            model = load_model(model_weight)
        except:
            model = load_model(model_weight, ({'SSMORMS3':optimizers.SSMORMS3}) )

        #output probabilities corresponding to postive labels.

        y_pred_train = model.predict(x_train)
        y_pred_val =  model.predict(x_val)

        y_pred_positive_train = y_pred_train[positive_boolean_index_train,1]
        y_pred_positive_val = y_pred_val[positive_boolean_index_val,1]

        #print(y_pred_positive.shape)
        #print(y_pred_positive)

        y_pred_negative_train = y_pred_train[negative_boolean_index_train,1]
        y_pred_negative_val = y_pred_val[negative_boolean_index_val,1]

        #flatten

        y_train=y_train[:,:, 1].reshape((-1, 1))
        y_pred_train = y_pred_train[:,:, 1].reshape((-1, 1))

        print(y_train.shape)
        print(y_pred_train.shape)

        y_val = y_val[:,:, 1].reshape((-1, 1))
        y_pred_val = y_pred_val[:,:, 1].reshape((-1, 1))

        auc_train = roc_auc_score(y_train, y_pred_train, average='macro', sample_weight=None)
        auc_val = roc_auc_score(y_val, y_pred_val, average='macro', sample_weight=None)

        auc_train_list.append(auc_train)
        auc_val_list.append(auc_val)

        if (epoch >= 1):
            epochs = np.arange(epoch)
            plt.plot(epochs, auc_train_list, color="green")
            plt.plot(epochs, auc_val_list, color="red")
            plt.xlabel("Epochs")
            plt.ylabel("AUC")

            green_patch = mpatches.Patch(color='green',label="Training AUC")
            red_patch = mpatches.Patch(color='red',label="Validation AUC")

            plt.legend(handles=[green_patch,red_patch],loc=4)

            plt.savefig(output_path + '/ROC_AUC_Epoch:{}.png'.format(epoch))
            plt.clf()


        ##################################################
        ##PLOT distributions
        ##################################################
        plt.subplot(2,EPOCH_INTERVAL,i)
        plt.title('Train#{}'.format(epoch))
        plt.hist(y_pred_positive_train, bins=np.linspace(0,1,20), color='green',alpha=0.5,normed=True) #add normalization
        plt.hist(y_pred_negative_train, bins=np.linspace(0,1,20), color='red',alpha=0.5,normed=True) #add normalization

        plt.subplot(2,EPOCH_INTERVAL,i+EPOCH_INTERVAL)
        plt.title('Val#{}'.format(epoch))
        plt.hist(y_pred_positive_val, bins=np.linspace(0,1,20), color='green',alpha=0.5,normed=True) #add normalization
        plt.hist(y_pred_negative_val, bins=np.linspace(0,1,20), color='red',alpha=0.5,normed=True) #add normalization

        i=i+1

        #Redraw new graph
        if(i==EPOCH_INTERVAL+1):
            plt.savefig(output_path + '/distribution_Epoch:{}-{}.png'.format(epoch-EPOCH_INTERVAL,epoch))
            plt.clf()
            i=1

if __name__ == "__main__":

    models_path = "/vol/work1/dyab/training_models/lstm_ssmorm3_default_step_2_dev_videos_last_derivative_graph"
    output_path = models_path+"/custom_evaluation/"

    if(not os.path.exists(output_path)):
        os.mkdir(output_path)
    #visualize_model_weights_change(models_path)

    x_train,y_train = utils.load_as_numpy_array("/vol/work1/dyab/training_set/lstm_trainingset_complete.hdf5")
    x_val,y_val = utils.load_as_numpy_array("/vol/work1/dyab/training_set/lstm_validationset_complete.hdf5")

    #do it once for trainng and once for validation.
    plot_distributions(models_path,output_path,x_train,y_train,x_val,y_val)