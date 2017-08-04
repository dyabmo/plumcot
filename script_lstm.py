####################################
#Originally authored by Hervé Bredin
#####################################

from keras.utils import to_categorical
from pyannote.generators.batch import batchify
from pyannote.audio.labeling.models import StackedLSTM
from keras.callbacks import ModelCheckpoint
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import roc_auc_score
import numpy as np
from glob import glob
import os.path
import time
import sys
import utils

NUMPY_PATH_AUDIO = "/vol/work1/dyab/training_set/numpy_arrays_local_audio"
NUMPY_PATH = '/vol/work1/dyab/training_set/numpy_arrays_local_landmarks'

DEV_NUMPY_PATH_AUDIO = "/vol/work1/dyab/development_set/numpy_arrays_cluster_old_audio"
DEV_NUMPY_PATH='/vol/work1/dyab/development_set/numpy_arrays_cluster_old_landmarks'

TEST_NUMPY_PATH_AUDIO = "/vol/work1/dyab/test_set/numpy_arrays_audio"
TEST_NUMPY_PATH='/vol/work1/dyab/test_set/numpy_arrays_landmarks'
BATCH_SIZE = 32

CATEGORICAL=False
REMOVE_LCP_TOPQUESTIONS=True
FIRST_DERIVATIVE=False
SECOND_DERIVATIVE=False
USE_FACE=True
USE_AUDIO = True
INPUT_DIMS = 40
WEIGHTS_DIR = '/vol/work1/dyab/training_models/bredin/one_out_of_ten_no_LCP_TopQuestions_face_audio'

if USE_FACE:
    INPUT_DIMS = 136
    NUMPY_PATH = '/vol/work1/dyab/training_set/numpy_arrays_local_landmarks_face'
    DEV_NUMPY_PATH = '/vol/work1/dyab/development_set/numpy_arrays_cluster_old_landmarks_face'
    TEST_NUMPY_PATH = '/vol/work1/dyab/test_set/numpy_arrays_landmarks_face'

if FIRST_DERIVATIVE:
    INPUT_DIMS=INPUT_DIMS + INPUT_DIMS

if SECOND_DERIVATIVE:
    INPUT_DIMS=INPUT_DIMS + INPUT_DIMS

if USE_AUDIO:
    INPUT_DIMS = INPUT_DIMS + 59

# preprocess images (resize, normalize, crop) and labels (to_categorical)
def preprocess(X, y):
    return X, to_categorical(y, num_classes=2)

# load list of paths to numpy files

X_PATHS_AUDIO = sorted(glob(NUMPY_PATH_AUDIO + '/*.Xa.npy'))
X_PATHS = sorted(glob(NUMPY_PATH + '/*.XLandmarks.npy'))
Y_PATHS = sorted(glob(NUMPY_PATH + '/*.Y.npy'))

#make sure audio and video files are the same
X_PATHS = [f for f in X_PATHS if "BFMTV_BFMStory_2012-07-16_175800" not in f]
X_PATHS = [f for f in X_PATHS if "LCP_EntreLesLignes_2012-10-16_032500" not in f]
X_PATHS = [f for f in X_PATHS if "LCP_LCPInfo13h30_2012-04-04_132700" not in f]

Y_PATHS = [f for f in Y_PATHS if "BFMTV_BFMStory_2012-07-16_175800" not in f]
Y_PATHS = [f for f in Y_PATHS if "LCP_EntreLesLignes_2012-10-16_032500" not in f]
Y_PATHS = [f for f in Y_PATHS if "LCP_LCPInfo13h30_2012-04-04_132700" not in f]

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
    X_PATHS = [f for f in X_PATHS if "LCP_TopQuestions" not in f ]
    Y_PATHS = [f for f in Y_PATHS if "LCP_TopQuestions" not in f ]
    X_PATHS_DEV = [f for f in X_PATHS_DEV if "LCP_TopQuestions" not in f ]
    Y_PATHS_DEV = [f for f in Y_PATHS_DEV if "LCP_TopQuestions" not in f ]
    X_PATHS_TEST = [f for f in X_PATHS_TEST if "LCP_TopQuestions" not in f]
    Y_PATHS_TEST = [f for f in Y_PATHS_TEST if "LCP_TopQuestions" not in f]


def check_order_x_y(X,Y):

    # make sure they are loaded in the same order (X must match y)
    for xp, yp in zip(X, Y):
        if xp[:-15] != yp[:-6]:
            print(xp, yp)
            sys.exit()

def check_order_landmarks_audio(X,X_AUDIO):

    # make sure they are loaded in the same order (X landmarks must match X audio)
    for xp, xa in zip(X, X_AUDIO):
        if xp.split("/")[-1][:-15] != xa.split("/")[-1][:-7]:
            print(xp, xa)
            sys.exit()


check_order_x_y(X_PATHS, Y_PATHS)
check_order_x_y(X_PATHS_DEV, Y_PATHS_DEV)
check_order_x_y(X_PATHS_TEST, Y_PATHS_TEST)

check_order_landmarks_audio(X_PATHS, X_PATHS_AUDIO)
check_order_landmarks_audio(X_PATHS_DEV, X_PATHS_DEV_AUDIO)
check_order_landmarks_audio(X_PATHS_TEST, X_PATHS_TEST_AUDIO)

# total number of tracks
N_TRACKS = len(X_PATHS)
LAST_TRACK = int(N_TRACKS * 0.9)
STEP = 10


# compute stats (number of samples, number of positive)
def statistics(y_paths):
    n_samples = 0
    n_positive = 0
    for y_path in tqdm(y_paths):
        y = np.load(y_path)
        n_samples += len(y)
        n_positive += np.sum(y)
    return n_samples, n_positive

# basic generator that loops forever or just once
def get_generator(x_paths, y_paths, forever=True,x_paths_audio=None):
    first_loop = True
    while forever or first_loop:
        for x_path, y_path,x_path_audio in zip(x_paths, y_paths,x_paths_audio):

            #print(x_path_audio)
            #print(x_path)
            X = np.load(x_path)
            Xa = np.load(x_path_audio)

            if CATEGORICAL:
                Y = to_categorical(np.load(y_path), num_classes=2)
            else:
                Y = np.load(y_path)
                Y=Y.reshape((len(Y),1))

            #Normalization on facetrack level
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_normalized = (X - mean) / std

            #choose the minumum number of samples of both
            n_samples = min(X_normalized.shape[0],Xa.shape[0])

            X_normalized_deriv,_ = utils.preprocess_lstm(X_normalized,Y,normalize=False,first_derivative=FIRST_DERIVATIVE,second_derivative=SECOND_DERIVATIVE)
            #print(Xa.shape)
            #print(X.shape)

            for i in range(n_samples - 25):
                xv = X_normalized_deriv[i:i+25]
                y = Y[i:i+25]
                xa = Xa[i:i+25]
                #print(xv.shape)
                #print (xa.shape)
                x = np.hstack((xv,xa))
                #print(x.shape)

                yield x, y
        first_loop = False

def train(x_paths, y_paths, weights_dir,x_paths_audio=None):
    n_samples, n_positive = statistics(y_paths)

    # estimate performance of "majority class" baseline
    baseline = 100 * n_positive / n_samples
    print('Baseline = {0:.1f}%'.format(baseline))

    # estimate number of batches per epoch
    steps_per_epoch = n_samples // BATCH_SIZE

    # create batch generator
    generator = get_generator(x_paths, y_paths,x_paths_audio=x_paths_audio)
    if CATEGORICAL:
        signature = ({'type': 'ndarray'}, {'type': 'ndarray'})
    else:
        signature = ({'type': 'ndarray'}, {'type': 'scalar'})


    batch_generator = batchify(generator, signature, batch_size=BATCH_SIZE)

    # create model
    if CATEGORICAL:
        model = StackedLSTM()((25, INPUT_DIMS))
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['acc'])
    else:

        model = StackedLSTM(final_activation="sigmoid",n_classes=1)((25, INPUT_DIMS))
        #only 4 units
        #model = StackedLSTM(lstm=[4,],mlp=[],final_activation="sigmoid",n_classes=1)((25, INPUT_DIMS))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', metrics=['acc'])

    # train model
    model_h5 = weights_dir + '/{epoch:04d}.h5'
    callbacks = [ModelCheckpoint(model_h5, period=1)]
    model.fit_generator(batch_generator, steps_per_epoch, epochs=1000,
                        verbose=1, callbacks=callbacks, workers=1)

def validate(x_paths, y_paths, weights_dir,x_paths_audio=None):

    epoch = 0
    f = open(WEIGHTS_DIR+"/list_dev",'w')
    while True:

        # sleep until next epoch is finished
        model_h5 = weights_dir + '/{epoch:04d}.h5'.format(epoch=epoch)
        if not os.path.isfile(model_h5):
            time.sleep(10)
            continue
        model = load_model(model_h5)

        generator = get_generator(x_paths, y_paths, forever=False,x_paths_audio=x_paths_audio)
        if CATEGORICAL:
            signature = ({'type': 'ndarray'}, {'type': 'ndarray'})
        else:
            signature = ({'type': 'ndarray'}, {'type': 'scalar'})
        batch_generator = batchify(generator, signature, batch_size=BATCH_SIZE)
        
        Y_true, Y_pred = [], []
        for X, y in batch_generator:
            #Y_pred.append(model.predict(X)[:, :, 1].reshape((-1, 1)))
            #Y_true.append(y[:, :, 1].reshape((-1, 1)))
            Y_pred.append(model.predict(X).reshape((-1, 1)))
            Y_true.append(y.reshape((-1, 1)))
            
        y_true = np.vstack(Y_true)
        y_pred = np.vstack(Y_pred)

        auc = roc_auc_score(y_true, y_pred, average='macro', sample_weight=None)
        print('#{epoch:04d} {auc:.4f}%'.format(epoch=epoch+1, auc=100*auc))
        f.write("{},".format(auc))
        f.flush()

        epoch += 1

#Split both X_PATHS and Y_PATHS
index_list = list()
for i in range(0, N_TRACKS-STEP ,STEP):
    index = np.arange(i,i+9,1)
    index_list.append(index)

#flatten list
index_list = [val for sublist in index_list for val in sublist]

TRAINING_X_PATHS = np.take(X_PATHS,index_list)
TRAINING_Y_PATHS = np.take(Y_PATHS,index_list)
TRAINING_X_PATHS_AUDIO = np.take(X_PATHS_AUDIO,index_list)

#take the 10th out of each 10 sequences
VALIDATION_X_PATHS = X_PATHS[9::STEP]
VALIDATION_Y_PATHS = Y_PATHS[9::STEP]
VALIDATION_X_PATHS_AUDIO = X_PATHS_AUDIO[9::STEP]

print(N_TRACKS)
if USE_AUDIO:
    #train(TRAINING_X_PATHS,
    #      TRAINING_Y_PATHS,
    #      WEIGHTS_DIR,
    #      TRAINING_X_PATHS_AUDIO)

    #validate(TRAINING_X_PATHS,
    #         TRAINING_Y_PATHS,
    #         WEIGHTS_DIR,
    #         TRAINING_X_PATHS_AUDIO)

    #validate(VALIDATION_X_PATHS,
    #         VALIDATION_Y_PATHS,
    #         WEIGHTS_DIR,
    #         VALIDATION_X_PATHS_AUDIO)

    validate(X_PATHS_DEV,
             Y_PATHS_DEV,
             WEIGHTS_DIR,
             X_PATHS_DEV_AUDIO)

    #validate(X_PATHS_TEST,
    #        Y_PATHS_TEST,
    #        WEIGHTS_DIR,
    #         X_PATHS_TEST_AUDIO)

#########################################################
#train(X_PATHS[:LAST_TRACK],
#      Y_PATHS[:LAST_TRACK],
#      WEIGHTS_DIR)

#validate(X_PATHS[:LAST_TRACK],
#         Y_PATHS[:LAST_TRACK],
#         WEIGHTS_DIR)