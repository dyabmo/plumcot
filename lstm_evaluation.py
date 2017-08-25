from keras.utils import to_categorical
from pyannote.generators.batch import batchify
from keras.models import load_model
from pyannote.audio.labeling.models import StackedLSTM
from sklearn.metrics import fbeta_score
import numpy as np
from glob import glob
import os.path
import time
import sys
from script_lstm import get_generator
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

BATCH_SIZE = 32
REMOVE_LCP_TOPQUESTIONS=True
USE_FACE=True
WEIGHTS_DIR = '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_audio'

DEV_NUMPY_PATH_AUDIO = "/vol/work1/dyab/development_set/numpy_arrays_cluster_old_audio"
DEV_NUMPY_PATH='/vol/work1/dyab/development_set/numpy_arrays_cluster_old_landmarks'

TEST_NUMPY_PATH_AUDIO = "/vol/work1/dyab/test_set/numpy_arrays_audio"
TEST_NUMPY_PATH='/vol/work1/dyab/test_set/numpy_arrays_landmarks'

if USE_FACE:
    DEV_NUMPY_PATH = '/vol/work1/dyab/development_set/numpy_arrays_cluster_old_landmarks_face'
    TEST_NUMPY_PATH = '/vol/work1/dyab/test_set/numpy_arrays_landmarks_face'

X_PATHS_DEV_AUDIO = sorted(glob(DEV_NUMPY_PATH_AUDIO + '/*.Xa.npy'))
X_PATHS_DEV = sorted(glob(DEV_NUMPY_PATH + '/*.XLandmarks.npy'))
Y_PATHS_DEV = sorted(glob(DEV_NUMPY_PATH + '/*.Y.npy'))

#make sure audio and video files are the same
X_PATHS_DEV = [f for f in X_PATHS_DEV if "LCP_EntreLesLignes_2011-04-05_025900" not in f]
Y_PATHS_DEV = [f for f in Y_PATHS_DEV if "LCP_EntreLesLignes_2011-04-05_025900" not in f]

X_PATHS_TEST_AUDIO = sorted(glob(TEST_NUMPY_PATH_AUDIO + '/*.Xa.npy'))
X_PATHS_TEST = sorted(glob(TEST_NUMPY_PATH + '/*.XLandmarks.npy'))
Y_PATHS_TEST = sorted(glob(TEST_NUMPY_PATH + '/*.Y.npy'))

# If you want to remove LCP videos.
if REMOVE_LCP_TOPQUESTIONS:
    X_PATHS_DEV = [f for f in X_PATHS_DEV if "LCP_TopQuestions" not in f ]
    Y_PATHS_DEV = [f for f in Y_PATHS_DEV if "LCP_TopQuestions" not in f ]
    X_PATHS_TEST = [f for f in X_PATHS_TEST if "LCP_TopQuestions" not in f]
    Y_PATHS_TEST = [f for f in Y_PATHS_TEST if "LCP_TopQuestions" not in f]

#Select best model/epoch on dev set according to AUC curve

with open(WEIGHTS_DIR+"/list_dev_prec_rec_auc", "r") as f:
    dev_models = f.readline().split(',')

#print(dev_models)
best_model = np.argmax(dev_models)
print(best_model)
model_h5 = WEIGHTS_DIR + '/{epoch:04d}.h5'.format(epoch=best_model)
model = load_model(model_h5)
f = open(WEIGHTS_DIR+"/scores_prec_rec",'w')
f.write("Model: {:04d}\n".format(best_model))
f.flush()

#Apply the upcoming methods on that dev model

#Using F-score: for beta = 1 or 0.1 or 10
#Theta = argmax ( f( p(theta), r(theta))
#Meaning that for each theta = 0.1,0.2...0.8,0.9: find precision and recall, then calculate f score, then find theta that corresponds to highest f score. Do this for each beta

score_matrix = np.zeros((3,3))
i=0

def validate_ev(x_paths, y_paths, weights_dir,x_paths_audio):

    generator = get_generator(x_paths, y_paths, forever=False,x_paths_audio=x_paths_audio)
    signature = ({'type': 'ndarray'}, {'type': 'scalar'})
    batch_generator = batchify(generator, signature, batch_size=BATCH_SIZE)

    Y_true, Y_pred = [], []
    for X, y in batch_generator:
        # Y_pred.append(model.predict(X)[:, :, 1].reshape((-1, 1)))
        # Y_true.append(y[:, :, 1].reshape((-1, 1)))
        Y_pred.append(model.predict(X).reshape((-1, 1)))
        Y_true.append(y.reshape((-1, 1)))

    y_true = np.vstack(Y_true)
    #check y_pred values
    y_pred = np.vstack(Y_pred)

    return y_true,y_pred

def compute_precision_recall():

    y_true_curve, y_pred_curve = validate_ev(X_PATHS_DEV,Y_PATHS_DEV,WEIGHTS_DIR,X_PATHS_DEV_AUDIO)


    precision, recall, thresholds = precision_recall_curve(y_true_curve, y_pred_curve )

    average_precision = average_precision_score(y_true_curve, y_pred_curve, average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall, precision, color='navy', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    print(thresholds)
    plt.savefig(WEIGHTS_DIR+"/precision_recall_curve2.png")


compute_precision_recall()

#Beta represents the ratio between precision and recall weights in F-score calculation
for beta in (1,0.1,10):

    print(beta)
    #Apply dev set
    y_true, y_pred = validate_ev(X_PATHS_DEV,Y_PATHS_DEV,WEIGHTS_DIR,X_PATHS_DEV_AUDIO)

    max_fbeta_score=0
    argmax_theta=0
    for theta_index in range(1,40,1):

        theta = theta_index/40.
        print(theta)

        #print(y_pred)

        y_pred_cutoff = y_pred >= theta
        score = fbeta_score(y_true, y_pred_cutoff, beta=beta)
        print(score)
        if(score > max_fbeta_score):
            max_fbeta_score = score
            argmax_theta = theta


    score_matrix[i][0] = beta
    score_matrix[i][1] = argmax_theta
    score_matrix[i][2] = max_fbeta_score
    i=i+1

    # Apply the 3 thetas of that selected model on test set
    # Compare 3 f-scores of test set with dev set FOR EACH EXPERIMENT
    # If the scores are close, then generalization occurred.
    y_true_test, y_pred_test = validate_ev(X_PATHS_TEST,Y_PATHS_TEST,WEIGHTS_DIR,X_PATHS_TEST_AUDIO)

    y_pred_test_cutoff = y_pred_test > argmax_theta
    test_score = fbeta_score(y_true_test, y_pred_test_cutoff, beta=beta)

    f.write("Beta: {}   Max_theta: {}   Development set F-score: {}   Test Set F-score: {}\n".format(beta,argmax_theta,max_fbeta_score,test_score))
    f.flush()