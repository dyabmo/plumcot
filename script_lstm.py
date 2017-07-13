from keras import optimizers
from keras.utils import to_categorical
from pyannote.generators.batch import batchify
from pyannote.audio.labeling.models import StackedLSTM
from pyannote.audio.optimizers import SSMORMS3
from keras.callbacks import ModelCheckpoint
import scipy.misc
from tqdm import tqdm
from keras.models import load_model
from sklearn.metrics import roc_auc_score
import numpy as np
from glob import glob
import os.path
import time
import sys
import utils

NUMPY_PATH = '/vol/work1/dyab/training_set/numpy_arrays_local_landmarks'
BATCH_SIZE = 32
CATEGORICAL=False
REMOVE_LCP_TOPQUESTIONS=True
FIRST_DERIVATIVE=True
SECOND_DERIVATIVE=True

# preprocess images (resize, normalize, crop) and labels (to_categorical)
def preprocess(X, y):
    return X, to_categorical(y, num_classes=2)

# load list of paths to numpy files
X_PATHS = sorted(glob(NUMPY_PATH + '/*.XLandmarks.npy'))
Y_PATHS = sorted(glob(NUMPY_PATH + '/*.Y.npy'))

# If you want to remove LCP videos.
if REMOVE_LCP_TOPQUESTIONS:
    X_PATHS = [f for f in X_PATHS if "LCP_TopQuestions" not in f ]
    Y_PATHS = [f for f in Y_PATHS if "LCP_TopQuestions" not in f ]

# make sure they are loaded in the same order (X must match y)
for xp, yp in zip(X_PATHS, Y_PATHS):
    if xp[:-15] != yp[:-6]:
        print(xp, yp)
        sys.exit()

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
def get_generator(x_paths, y_paths, forever=True):
    first_loop = True
    while forever or first_loop:
        for x_path, y_path in zip(x_paths, y_paths):

            if CATEGORICAL:
                X, Y = np.load(x_path), to_categorical(np.load(y_path), num_classes=2)
            else:
                X, Y = np.load(x_path), np.load(y_path)
                Y=Y.reshape((len(Y),1))

            #Normalization on facetrack level
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_normalized = (X - mean) / std

            n_samples = X_normalized.shape[0]

            X_normalized_deriv,_ = utils.preprocess_lstm(X_normalized,Y,normalize=False,first_derivative=FIRST_DERIVATIVE,second_derivative=SECOND_DERIVATIVE)

            print(X_normalized_deriv.shape)

            for i in range(n_samples - 25):
                x = X_normalized_deriv[i:i+25]
                y = Y[i:i+25]
                #print(x.shape)
                #print(y.shape)
                yield x, y
        first_loop = False

def train(x_paths, y_paths, weights_dir):
    n_samples, n_positive = statistics(y_paths)

    # estimate performance of "majority class" baseline
    baseline = 100 * n_positive / n_samples
    print('Baseline = {0:.1f}%'.format(baseline))

    # estimate number of batches per epoch
    steps_per_epoch = n_samples // BATCH_SIZE

    # create batch generator
    generator = get_generator(x_paths, y_paths)
    if CATEGORICAL:
        signature = ({'type': 'ndarray'}, {'type': 'ndarray'})
    else:
        signature = ({'type': 'ndarray'}, {'type': 'scalar'})


    batch_generator = batchify(generator, signature, batch_size=BATCH_SIZE)

    # create model
    if CATEGORICAL:
        model = StackedLSTM()((25, 40))
        model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['acc'])
    else:

        model = StackedLSTM(final_activation="sigmoid",n_classes=1)((25, 40))
        #only 4 units
        #model = StackedLSTM(lstm=[4,],mlp=[],final_activation="sigmoid",n_classes=1)((25, 40))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop', metrics=['acc'])

    # train model
    model_h5 = weights_dir + '/{epoch:04d}.h5'
    callbacks = [ModelCheckpoint(model_h5, period=1)]
    model.fit_generator(batch_generator, steps_per_epoch, epochs=1000,
                        verbose=1, callbacks=callbacks, workers=1)

def validate(x_paths, y_paths, weights_dir):

    epoch = 0
    f = open(WEIGHTS_DIR+"/list_validate",'w')
    while True:

        # sleep until next epoch is finished
        model_h5 = weights_dir + '/{epoch:04d}.h5'.format(epoch=epoch)
        if not os.path.isfile(model_h5):
            time.sleep(10)
            continue
        model = load_model(model_h5)

        generator = get_generator(x_paths, y_paths, forever=False)
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

if CATEGORICAL:
    WEIGHTS_DIR = "/vol/work1/dyab/training_models/bredin/derivatives"
else:
    WEIGHTS_DIR = '/vol/work1/dyab/training_models/bredin/derivatives'

#Split both X_PATHS and Y_PATHS
index_list = list()
for i in range(0, N_TRACKS-STEP ,STEP):
    index = np.arange(i,i+9,1)
    index_list.append(index)

#flatten list
index_list = [val for sublist in index_list for val in sublist]

TRAINING_X_PATHS = np.take(X_PATHS,index_list)
TRAINING_Y_PATHS = np.take(Y_PATHS,index_list)

#take the 10th out of each 10 sequences
VALIDATION_X_PATHS = X_PATHS[9::STEP]
VALIDATION_Y_PATHS = Y_PATHS[9::STEP]

print(N_TRACKS)
train(TRAINING_X_PATHS,
      TRAINING_Y_PATHS,
     WEIGHTS_DIR)

#validate(VALIDATION_X_PATHS,
#         VALIDATION_Y_PATHS,
#         WEIGHTS_DIR)

#########################################################
#train(X_PATHS[:LAST_TRACK],
#      Y_PATHS[:LAST_TRACK],
#      WEIGHTS_DIR)

#validate(X_PATHS[:LAST_TRACK],
#         Y_PATHS[:LAST_TRACK],
#         WEIGHTS_DIR)

#signature = ({'type': 'ndarray'}, {'type': 'ndarray'})
#train_generator = get_generator(X_PATHS[:LAST_TRAIN_TRACK], Y_PATHS[:LAST_TRAIN_TRACK])
#train_batch_generator = batchify(train_generator, signature, batch_size=BATCH_SIZE)
#val_generator = get_generator(X_PATHS[LAST_TRAIN_TRACK:], Y_PATHS[LAST_TRAIN_TRACK:])
#val_batch_generator = batchify(val_generator, signature, batch_size=BATCH_SIZE)
#import utils
#utils.save_sequences("/vol/work1/dyab/training_set/",train_batch_generator,val_batch_generator)