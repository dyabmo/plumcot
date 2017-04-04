#!/bin/bash

LINE=$(cat /vol/work1/dyab/train_video_list | head -n "$1" | tail -1 )
echo $LINE
#Set directories
REPERE_DIR="/people/dyab/repere_train"
OUTPUT_DIR="/vol/work1/dyab/training_set"
DLIB_DIR1="/vol/work1/dyab/dlib-models/shape_predictor_68_face_landmarks.dat"
DLIB_DIR2="/vol/work1/dyab/dlib-models/dlib_face_recognition_resnet_model_v1.dat"
VIDEO_DIR=${REPERE_DIR}"/VIDEO/"${LINE}".MPG"
TRS_DIR=${REPERE_DIR}"/TRS/"${LINE}".trs"
XGTF_DIR=${REPERE_DIR}"/XGTF/"${LINE}".xgtf"
SHOT_DIR=${OUTPUT_DIR}"/residual/"${LINE}".output.json"
FACETRACK_DIR=${OUTPUT_DIR}"/residual/"${LINE}".track.txt"
LANDMARKS_DIR=${OUTPUT_DIR}"/residual/"${LINE}".landmarks.txt"
EMBEDDINGS_DIR=${OUTPUT_DIR}"/residual/"${LINE}".embedding.txt"

(set -x; pyannote-structure.py shot "${VIDEO_DIR}" "${SHOT_DIR}")

(set -x; pyannote-face.py track --every=0.5 "${VIDEO_DIR}" "${SHOT_DIR}" "${FACETRACK_DIR}")

(set -x; pyannote-face.py extract --verbose "${VIDEO_DIR}" "${FACETRACK_DIR}" "${DLIB_DIR1}" "${DLIB_DIR2}"  "${LANDMARKS_DIR}" "${EMBEDDINGS_DIR}" )

(set -x; python /people/dyab/plumcot/xml_parser.py "${TRS_DIR}" "${XGTF_DIR}" "${FACETRACK_DIR}" "${VIDEO_DIR}" "${LANDMARKS_DIR}" "${OUTPUT_DIR}" )

echo ====================================================================
echo \
