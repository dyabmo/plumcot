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


def compute_precision_recall_agg(type):

    precision_fname = "precision_"+type
    recall_fname = "recall_"+type


    precision1 = np.genfromtxt('/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_baseline_audio/'+precision_fname,delimiter=",",skip_header=0)
    recall1 =  np.genfromtxt('/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_baseline_audio/'+recall_fname,delimiter=",",skip_header=0)

    precision2 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/derivatives/'+precision_fname,
        delimiter=",", skip_header=0)
    recall2 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/derivatives/'+recall_fname,
        delimiter=",", skip_header=0)

    precision3 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_NoDerivatives/'+precision_fname,
        delimiter=",", skip_header=0)
    recall3 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_NoDerivatives/'+recall_fname,
        delimiter=",", skip_header=0)

    precision4 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_audio/'+precision_fname,
        delimiter=",", skip_header=0)
    recall4 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_audio/'+recall_fname,
        delimiter=",", skip_header=0)

    precision5 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/mouth_derivatives_audio/' + precision_fname,
        delimiter=",", skip_header=0)
    recall5 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/mouth_derivatives_audio/' + recall_fname,
        delimiter=",", skip_header=0)

    precision6 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/face_derivatives_audio/' + precision_fname,
        delimiter=",", skip_header=0)
    recall6 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/face_derivatives_audio/' + recall_fname,
        delimiter=",", skip_header=0)

    precision7 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/face_derivatives/' + precision_fname,
        delimiter=",", skip_header=0)
    recall7 = np.genfromtxt(
        '/vol/work1/dyab/training_models/bredin/face_derivatives/' + recall_fname,
        delimiter=",", skip_header=0)



    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall1[:-100], precision1[:-100], color='yellow', label='Audio only')
    #plt.plot(recall3[:-100], precision3[:-100], color='green', label='Face')
    plt.plot(recall2[:-100], precision2[:-100], color='red', label='Mouth + Derivatives')
    #plt.plot(recall7, precision7, color='cyan', label='Face + Derivatives')
    #plt.plot(recall4[:-100], precision4[:-100], color='blue', label='Face + Audio')
    #plt.plot(recall5, precision5, color='magenta', label='Mouth + Derivatives + Audio')
    #plt.plot(recall6, precision6, color='violet', label='Face + Derivatives + Audio')


    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.5, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig("/vol/work1/dyab/training_models/bredin/precision_recall_curve"+type+"_audio_mouth_derivatives.pdf")

compute_precision_recall_agg("dev")
compute_precision_recall_agg("test")
exit()

#Select best model/epoch on dev set according to AUC curve

with open(WEIGHTS_DIR+"/list_dev_prec_rec_auc", "r") as f:
    dev_models = f.readline().split(',')

#print(dev_models)
best_model = np.argmax(dev_models)
print(best_model)
model_h5 = WEIGHTS_DIR + '/{epoch:04d}.h5'.format(epoch=best_model)
model = load_model(model_h5)

#Select the best on test set
with open(WEIGHTS_DIR+"/list_test_prec_rec_auc", "r") as f:
    test_models = f.readline().split(',')

#print(dev_models)
best_model_test = np.argmax(test_models)
print(best_model_test)
model_h5_test = WEIGHTS_DIR + '/{epoch:04d}.h5'.format(epoch=best_model_test)
model_test = load_model(model_h5_test)

f = open(WEIGHTS_DIR+"/modified_scores_prec_rec",'w')
f.write("Model: {:04d}\n".format(best_model))
f.flush()

#Apply the upcoming methods on that dev model

#Using F-score: for beta = 1 or 0.1 or 10
#Theta = argmax ( f( p(theta), r(theta))
#Meaning that for each theta = 0.1,0.2...0.8,0.9: find precision and recall, then calculate f score, then find theta that corresponds to highest f score. Do this for each beta

score_matrix = np.zeros((3,3))
i=0

def validate_ev(x_paths, y_paths, input_model,x_paths_audio):

    generator = get_generator(x_paths, y_paths, forever=False,x_paths_audio=x_paths_audio)
    signature = ({'type': 'ndarray'}, {'type': 'scalar'})
    batch_generator = batchify(generator, signature, batch_size=BATCH_SIZE)

    Y_true, Y_pred = [], []
    for X, y in batch_generator:
        # Y_pred.append(model.predict(X)[:, :, 1].reshape((-1, 1)))
        # Y_true.append(y[:, :, 1].reshape((-1, 1)))
        Y_pred.append(input_model.predict(X).reshape((-1, 1)))
        Y_true.append(y.reshape((-1, 1)))

    y_true = np.vstack(Y_true)
    #check y_pred values
    y_pred = np.vstack(Y_pred)

    return y_true,y_pred

def compute_precision_recall(input_model):

    y_true_curve, y_pred_curve = validate_ev(X_PATHS_DEV,Y_PATHS_DEV,input_model,X_PATHS_DEV_AUDIO)

    y_true_curve_test, y_pred_curve_test = validate_ev(X_PATHS_TEST,Y_PATHS_TEST,input_model,X_PATHS_TEST_AUDIO)

    precision, recall, thresholds = precision_recall_curve(y_true_curve, y_pred_curve )

    precision_test, recall_test, thresholds_test = precision_recall_curve(y_true_curve_test, y_pred_curve_test)

    f = open(WEIGHTS_DIR + "/precision_dev", 'w')
    for i in range(0,precision.shape[0]):
        f.write("{},".format(precision[i]))
        f.flush()

    f2 = open(WEIGHTS_DIR + "/recall_dev", 'w')
    for i in range(0,recall.shape[0]):
        f2.write("{},".format(recall[i]))
        f2.flush()

    f3 = open(WEIGHTS_DIR + "/precision_test", 'w')
    for i in range(0, precision_test.shape[0]):
        f3.write("{},".format(precision_test[i]))
        f3.flush()

    f4 = open(WEIGHTS_DIR + "/recall_test", 'w')
    for i in range(0, recall_test.shape[0]):
        f4.write("{},".format(recall_test[i]))
        f4.flush()

    average_precision = average_precision_score(y_true_curve, y_pred_curve, average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall[:-1], precision[:-1], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision))
    plt.legend(loc="lower left")
    plt.savefig(WEIGHTS_DIR+"/precision_recall_curve_dev.png")

    average_precision_test = average_precision_score(y_true_curve_test, y_pred_curve_test, average="micro")

    # Plot Precision-Recall curve
    plt.clf()
    plt.plot(recall_test[:-100], precision_test[:-100], label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: AUC={0:0.2f}'.format(average_precision_test))
    plt.legend(loc="lower left")
    plt.savefig(WEIGHTS_DIR + "/precision_recall_curve_test.png")



def compute_baseline():

    y_true, y_pred = validate_ev(X_PATHS_DEV,Y_PATHS_DEV,WEIGHTS_DIR,X_PATHS_DEV_AUDIO)

    y_pred_cutoff = y_pred >= 0

    dev_score = fbeta_score(y_true, y_pred_cutoff, beta=1)
    print(dev_score)

    y_true_test, y_pred_test = validate_ev(X_PATHS_TEST, Y_PATHS_TEST, WEIGHTS_DIR, X_PATHS_TEST_AUDIO)

    y_pred_test_cutoff = y_pred_test >= 0
    test_score = fbeta_score(y_true_test, y_pred_test_cutoff, beta=1)
    print(test_score)

    exit()



compute_precision_recall(model)
exit()
#Beta represents the ratio between precision and recall weights in F-score calculation
beta = 1

#Apply dev set
y_true, y_pred = validate_ev(X_PATHS_DEV,Y_PATHS_DEV,model,X_PATHS_DEV_AUDIO)
theta_array_dev=list()
score_array_dev = list()

max_fbeta_score=0
argmax_theta=0
for theta_index in range(1,40,1):

    theta = theta_index/40.

    y_pred_cutoff = y_pred >= theta
    score = fbeta_score(y_true, y_pred_cutoff, beta=beta)
    if(score > max_fbeta_score):
        max_fbeta_score = score
        argmax_theta = theta

    theta_array_dev.append(theta)
    score_array_dev.append(score)
# Apply the 3 thetas of that selected model on test set
# Compare 3 f-scores of test set with dev set FOR EACH EXPERIMENT
# If the scores are close, then generalization occurred.
y_true_test, y_pred_test = validate_ev(X_PATHS_TEST,Y_PATHS_TEST,model,X_PATHS_TEST_AUDIO)

y_pred_test_cutoff = y_pred_test > argmax_theta
test_score = fbeta_score(y_true_test, y_pred_test_cutoff, beta=beta)

#change model to test
y_true_test, y_pred_test_max = validate_ev(X_PATHS_TEST,Y_PATHS_TEST,model_test,X_PATHS_TEST_AUDIO)

max_fbeta_score_test=0
argmax_theta_test=0

theta_array_test=list()
score_array_test = list()
#Do same on test as dev
for theta_index in range(1,40,1):

    theta = theta_index/40.

    y_pred_test_cutoff_max = y_pred_test_max >= theta
    score = fbeta_score(y_true_test, y_pred_test_cutoff_max, beta=beta)
    if(score > max_fbeta_score_test):
        max_fbeta_score_test = score
        argmax_theta_test = theta

    theta_array_test.append(theta)
    score_array_test.append(score)

f.write("Max_theta_dev: {}   Development set F-score: {}   Test Set F-score: {}  Max_theta_test: {}  Max Test Set F-score: {} Difference: {}\n".format(argmax_theta,max_fbeta_score,test_score,argmax_theta_test,max_fbeta_score_test, abs(test_score - max_fbeta_score_test) ))
f.flush()

plt.clf()
plt.plot(theta_array_dev, score_array_dev, color='blue', label='Development set')
plt.plot(theta_array_test, score_array_test, color='red', label='Test set')
plt.plot([0,1],[max_fbeta_score,max_fbeta_score],'b--',label="Max F1-score for Dev. set")
plt.plot([0,1],[max_fbeta_score_test,max_fbeta_score_test],'r--',label="Max F1-score for Test set")

plt.xlabel('Theta')
plt.ylabel('F1-score')
plt.ylim([0.5, 1.05])
plt.xlim([0.0, 1.0])
#plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(WEIGHTS_DIR + "/F1-score-theta.pdf")