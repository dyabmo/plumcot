#!/bin/bash

chmod 755 /people/dyab/plumcot/scripts/generate_dataset_cluster.sh
#Get the specific line from file of number SGE_TASK_ID

LINE=`cat "$1" | head -n ${SGE_TASK_ID} | tail -1`

if [ "$2" == "train" ]; then
#Set directories
REPERE_DIR="/people/dyab/repere_train"
VIDEO_DIR=${REPERE_DIR}"/VIDEO/"${LINE}".MPG"
TRS_DIR=${REPERE_DIR}"/TRS/"${LINE}".trs"
XGTF_DIR=${REPERE_DIR}"/XGTF/"${LINE}".xgtf"
OUTPUT_DIR="/vol/work1/dyab/training_set"
echo "Train"
elif [ "$2" == "dev" ]; then
#Set directories
REPERE_DIR="/people/dyab/repere_dev"
VIDEO_DIR=${REPERE_DIR}"/VIDEO/dev0/"${LINE}".MPG"
TRS_DIR=${REPERE_DIR}"/TRS/dev0/"${LINE}".trs"
XGTF_DIR=${REPERE_DIR}"/XGTF/dev0/"${LINE}".xgtf"
OUTPUT_DIR="/vol/work1/dyab/development_set"
echo "Development"
elif [ "$2" == "test" ]; then
#Set directories
REPERE_DIR="/people/dyab/repere_test"
VIDEO_DIR=${REPERE_DIR}"/VIDEO/test0/"${LINE}".MPG"
TRS_DIR=${REPERE_DIR}"/TRS/test0/"${LINE}".trs"
XGTF_DIR=${REPERE_DIR}"/XGTF/test0/"${LINE}".xgtf"
OUTPUT_DIR="/vol/work1/dyab/test_set"
echo "Test"
fi

SHOT_DIR=${OUTPUT_DIR}"/residual/"${LINE}".output.json"
FACETRACK_DIR=${OUTPUT_DIR}"/residual/"${LINE}".track.txt"
LANDMARKS_DIR=${OUTPUT_DIR}"/residual/"${LINE}".landmarks.txt"
EMBEDDINGS_DIR=${OUTPUT_DIR}"/residual/"${LINE}".embedding.txt"
DLIB_DIR1="/vol/work1/dyab/dlib-models/shape_predictor_68_face_landmarks.dat"
DLIB_DIR2="/vol/work1/dyab/dlib-models/dlib_face_recognition_resnet_model_v1.dat"

(set -x; pyannote-structure.py shot "${VIDEO_DIR}" "${SHOT_DIR}")

(set -x; pyannote-face.py track --every=0.5 "${VIDEO_DIR}" "${SHOT_DIR}" "${FACETRACK_DIR}")

(set -x; pyannote-face.py extract --verbose "${VIDEO_DIR}" "${FACETRACK_DIR}" "${DLIB_DIR1}" "${DLIB_DIR2}"  "${LANDMARKS_DIR}" "${EMBEDDINGS_DIR}" )

(set -x; python /people/dyab/plumcot/xml_parser.py "${TRS_DIR}" "${XGTF_DIR}" "${FACETRACK_DIR}" "${VIDEO_DIR}" "${LANDMARKS_DIR}" "${OUTPUT_DIR}" )

echo $LINE >> ${OUTPUT_DIR}"/completed.txt"

echo ====================================================================
echo \
