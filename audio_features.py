from pyannote.core import Segment
from pyannote.audio.features import Precomputed, utils
import pandas as pd
import numpy as np
from glob import glob

# REPERE evaluation protocol
# cf. https://github.com/pyannote/pyannote-database#pyannote-database
from pyannote.database import get_protocol
#different from the files I use
protocol = get_protocol('REPERE.SpeakerDiarization.Plumcot')
precomputed = Precomputed('/vol/work1/dyab/training_set/mfcc')

train_dir = "/vol/work1/dyab/training_set/residual_local/"
output_dir_train = "/vol/work1/dyab/training_set/numpy_arrays_local_audio/"
y_labels_dir_train = "/vol/work1/dyab/training_set/numpy_arrays_local_landmarks/"

dev_dir="/vol/work1/dyab/development_set/residual_cluster_old/"
output_dir_dev = "/vol/work1/dyab/development_set/numpy_arrays_cluster_old_audio/"
y_labels_dir_dev = "/vol/work1/dyab/development_set/numpy_arrays_cluster_old_landmarks/"

test_dir="/vol/work1/dyab/test_set/residual/"
output_dir_test =  "/vol/work1/dyab/test_set/numpy_arrays_audio/"
y_labels_dir_test = "/vol/work1/dyab/test_set/numpy_arrays_landmarks/"

def generate_audio_features(dir,output_dir,y_labels_dir):

    for current_file in protocol.test():      # iterate on all files of Phase2 training set
        print(current_file['uri'])
        landmarks_file_name = dir + current_file['uri'] + '.track.txt'

        #keys: uri, database
        try:
            features = precomputed(current_file)   # load precomputed features
            face_landmarks_numpy_array =  pd.read_csv(landmarks_file_name, sep=" ", header=None)

        except utils.PyannoteFeatureExtractionError:
            print("No features extracted for" +current_file['uri'] )
            continue
        except FileNotFoundError:
            print(landmarks_file_name+" not found")
            continue
        else:

            #get face landmarks file containing  facetrack segments
            face_landmarks_numpy_array.columns =  ["time", "id", "left", "top","right","bottom","state"]

            group_time_by_id = face_landmarks_numpy_array[['time', 'id']].groupby('id', as_index=False)

            #extract only relevant facetracks
            # Extract all Y numpy arrays ( all facetrack files) relevant to that file only
            y_numpy_arrays_names = glob(y_labels_dir + current_file['uri'] + "*.Y.npy")

            # get the list of facetracks ids
            y_numpy_arrays = list(map(lambda y: y.split("/")[-1].split(".")[1], y_numpy_arrays_names))
            #print(y_numpy_arrays)

            for index in range(len(y_numpy_arrays)):
                id = int(y_numpy_arrays[index])
                #print(id)
                min = group_time_by_id.min().ix[id]['time']
                max = group_time_by_id.max().ix[id]['time']

                #print(min,max)

                Xa = features.crop(Segment(min, max))   # obtain features as numpy array for given temporal segment
                Yv = np.load(y_labels_dir + current_file['uri'] + "." + str(id) + ".Y.npy")

                np.save(output_dir + current_file['uri'] + '.' + str(id) + '.Xa.npy', Xa)
                np.save(output_dir + current_file['uri'] + '.' + str(id) + '.Y.npy', Yv)


def create_uem(input_name,output_name):
    train_video_names = pd.read_csv(input_name, sep=" ", header=None)
    train_video_names.columns = ["name"]
    train_video_names["num1"]=0
    train_video_names["num2"]=0
    train_video_names["num3"]=0

    train_video_names.to_csv(output_name, sep=" ", header=False, index_label=False, index=False)

#create_uem('/vol/work1/dyab/training_set/train_video_list','plumcot.trn.uem')
#create_uem('/vol/work1/dyab/development_set/dev0_video_list','plumcot.dev.uem')
#create_uem('/vol/work1/dyab/test_set/test_video_list','plumcot.tst.uem')

generate_audio_features(test_dir,output_dir_test,y_labels_dir_test)