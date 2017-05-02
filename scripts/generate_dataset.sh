#!/bin/bash

chmod 755 /people/dyab/plumcot/scripts/generate_dataset.sh
#Get the specific line from file of number SGE_TASK_ID

#Check if the script is running on the local machine or the cluster ?
#Host is the local machine
if [ "$HOSTNAME" == "m167" ]; then
    echo "Executing on local machine"
    LINE=`cat "$2" | head -n "$1" | tail -1`
    TYPE="$3"
    MODE="$4"
#Host is the cluster
else
    echo "Executing on cluster"
    echo $HOSTNAME
    LINE=`cat "$1" | head -n ${SGE_TASK_ID} | tail -1`
    TYPE="$2"
    MODE="$3"
fi

if [ "$TYPE" == "train" ]; then
    #Set directories
    REPERE_DIR="/people/dyab/repere_train"
    VIDEO_DIR=${REPERE_DIR}"/VIDEO/"${LINE}".MPG"
    TRS_DIR=${REPERE_DIR}"/TRS/"${LINE}".trs"
    XGTF_DIR=${REPERE_DIR}"/XGTF/"${LINE}".xgtf"
    OUTPUT_DIR="/vol/work1/dyab/training_set"
    echo "Train"

elif [ "$TYPE" == "dev" ]; then
    #Set directories
    REPERE_DIR="/people/dyab/repere_dev"
    VIDEO_DIR=${REPERE_DIR}"/VIDEO/dev2/"${LINE}".MPG"
    TRS_DIR=${REPERE_DIR}"/TRS/dev2/"${LINE}".trs"
    XGTF_DIR=${REPERE_DIR}"/XGTF/dev2/"${LINE}".xgtf"
    OUTPUT_DIR="/vol/work1/dyab/development_set"
    echo "Development"

elif [ "$TYPE" == "test" ]; then
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
FACETRACK_TALKINGFACE=${OUTPUT_DIR}"/validate_groundtruth/"${LINE}".facetrack_talkingface.txt"
XGTF_OUT_DIR=${OUTPUT_DIR}"/validate_groundtruth/"${LINE}".xgtf"
VIDEO_OUT_DIR=${OUTPUT_DIR}"/validate_groundtruth/"${LINE}".facetrack_talkingface.mp4"

DLIB_DIR1="/vol/work1/dyab/dlib-models/shape_predictor_68_face_landmarks.dat"
DLIB_DIR2="/vol/work1/dyab/dlib-models/dlib_face_recognition_resnet_model_v1.dat"

if [ "$MODE" == "normal" ]; then

    (set -x; pyannote-structure.py shot "${VIDEO_DIR}" "${SHOT_DIR}")

    (set -x; pyannote-face.py track --every=0.5 "${VIDEO_DIR}" "${SHOT_DIR}" "${FACETRACK_DIR}")

    (set -x; pyannote-face.py extract --verbose "${VIDEO_DIR}" "${FACETRACK_DIR}" "${DLIB_DIR1}" "${DLIB_DIR2}"  "${LANDMARKS_DIR}" "${EMBEDDINGS_DIR}" )

    (set -x; python /people/dyab/plumcot/scripts/xml_parser.py "${TRS_DIR}" "${XGTF_DIR}" "${FACETRACK_DIR}" "${VIDEO_DIR}" "${LANDMARKS_DIR}" "${OUTPUT_DIR}" "${FACETRACK_TALKINGFACE}")


elif [ "$MODE" == "generate_video" ]; then

    #Copy xgtf for easier comparison
    cp "${XGTF_DIR}" "${XGTF_OUT_DIR}"
    echo "Generating video"
    (set -x; python /people/dyab/plumcot/scripts/xml_parser.py "${TRS_DIR}" "${XGTF_DIR}" "${FACETRACK_DIR}" "${VIDEO_DIR}" "${LANDMARKS_DIR}" "${OUTPUT_DIR}" "${FACETRACK_TALKINGFACE}")
    (set -x; pyannote-face.py demo "${VIDEO_DIR}" "${FACETRACK_TALKINGFACE}" "${VIDEO_OUT_DIR}" --talking-face )

fi

#If this file is found, it means that the file was processed successfully
if [ -f "${FACETRACK_TALKINGFACE}" ]; then
    echo $LINE >> ${OUTPUT_DIR}"/completed_videos.txt"
else
    echo $LINE >> ${OUTPUT_DIR}"/error_log.txt"
fi

echo ====================================================================
echo \
