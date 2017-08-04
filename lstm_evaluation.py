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

#TODO: Use generator from script_lstm and script_audio_baseline_lstm
#TODO: Apply on all WIGHTS_DIRs

BATCH_SIZE = 32
WEIGHTS_DIR = '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_audio'

#Select best model/epoch on dev set according to AUC curve
dev_models = np.loadtxt(WEIGHTS_DIR+"/list_dev")
best_model = np.argmax(dev_models)
model_h5 = WEIGHTS_DIR + '/{epoch:04d}.h5'.format(epoch=best_model)
model = load_model(model_h5)
f = open(WEIGHTS_DIR+"/scores",'w')
f.write("Model: {:04d}\n".format(best_model))
f.flush()


#Apply the upcoming methods on that dev model

#Using F-score: for beta = 1 or 0.1 or 10
#Theta = argmax ( f( p(theta), r(theta))
#Meaning that for each theta = 0.1,0.2...0.8,0.9: find precision and recall, then calculate f score, then find theta that corresponds to highest f score. Do this for each beta

score_matrix = np.zeros((3,3))
i=0

def validate(x_paths, y_paths, weights_dir):

    generator = get_generator(x_paths, y_paths, forever=False)
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

    return Y_true,Y_pred

#Beta represents the ratio between precision and recall weights in F-score calculation
for beta in (1,0.1,10):

    #Apply dev set
    y_true, y_pred = validate(x_paths_dev,y_paths_dev,WEIGHTS_DIR)

    max_fbeta_score=0
    argmax_theta=0
    for theta in range(0.1,0.9,0.1):

        y_pred = y_pred > theta
        score = fbeta_score(y_true, y_pred, beta=beta)

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
    Y_true, Y_pred = validate(x_paths_test,y_paths_test,WEIGHTS_DIR)

    y_pred = y_pred > argmax_theta
    test_score = fbeta_score(y_true, y_pred, beta=beta)

    f.write("Beta: {}. Max_theta: {}. Development set F-score: {}.Test Set F-score\n".format(beta,argmax_theta,max_fbeta_score,test_score))
    f.flush()