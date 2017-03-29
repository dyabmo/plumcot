#!/bin/bash

#Activate conda environment in case it's not activated
source activate deeplearning

chmod 755 /people/dyab/plumcot/validate_groundtruth.sh
#Set directories
REPERE_DIR="/people/dyab/repere_train"
OUTPUT_DIR="/vol/work1/dyab/groundtruth_validation"
DLIB_DIR="/vol/work1/dyab/dlib.face.landmarks.dat"

XML_PARSER="/people/dyab/plumcot/xml_parser.py"

PYTHON_DIR="/people/dyab/pyannote-video/scripts"
PYANNOTE_STRUCTURE=${PYTHON_DIR}"/pyannote-structure.py"
PYANNOTE_FACE=${PYTHON_DIR}"/pyannote-face.py"

#Read each video ID from input file and process it to generate face track and shot.json
while IFS='' read -r LINE || [[ -n "$LINE" ]]; do

    VIDEO_DIR=${REPERE_DIR}"/VIDEO/"${LINE}".MPG"
    TRS_DIR=${REPERE_DIR}"/TRS/"${LINE}".trs"
    XGTF_DIR=${REPERE_DIR}"/XGTF/"$LINE".xgtf"

    SHOT_DIR=${OUTPUT_DIR}"/residual/"${LINE}".output.json"
    FACETRACK_DIR=${OUTPUT_DIR}"/residual/"${LINE}".track.txt"
    LANDMARKS_DIR=${OUTPUT_DIR}"/residual/"${LINE}".landmarks.txt"

    FACETRACK_TALKINGFACE=${OUTPUT_DIR}"/residual/"${LINE}".facetrack_talkingface.txt"
    VIDEO_OUT_DIR=${OUTPUT_DIR}"/"${LINE}".talkingface.mp4"
    XGTF_OUT_DIR=${OUTPUT_DIR}"/"${LINE}".xgtf"

    #Copy xgtf for easier comparison
    cp "${XGTF_DIR}" "${XGTF_OUT_DIR}"
    
    echo Starting to split videos into shots\
    #Split video into shots, generate shot.json for each video file.
    echo python "${PYANNOTE_STRUCTURE}" shot "${VIDEO_DIR}" "${SHOT_DIR}"
    python "${PYANNOTE_STRUCTURE}" shot "${VIDEO_DIR}" "${SHOT_DIR}"

    echo Starting to track faces according to splitted shots\
    #To track faces according to splitted shots
    #Note: setting the interval "--every" is important for quick processing!
    python "${PYANNOTE_FACE}" track --every=0.5 "${VIDEO_DIR}" "${SHOT_DIR}" "${FACETRACK_DIR}"

    echo Generate landmarks\
    #Generate landmarks

    #Change to master to execute landmarks command
    cd /people/dyab/pyannote-video
    git checkout master

    echo python "${PYANNOTE_FACE}" landmarks "${VIDEO_DIR}" "${DLIB_DIR}" "${FACETRACK_DIR}" "${LANDMARKS_DIR}"
    python "${PYANNOTE_FACE}" landmarks "${VIDEO_DIR}" "${DLIB_DIR}" "${FACETRACK_DIR}" "${LANDMARKS_DIR}"

    echo Generate facetrack_talkingface\
    #Run xml_parser to generate the talking_face labels
    echo python "${XML_PARSER}" "${TRS_DIR}" "${XGTF_DIR}" "${FACETRACK_DIR}" "${VIDEO_DIR}" "${LANDMARKS_DIR}" "${FACETRACK_TALKINGFACE}"
    python "${XML_PARSER}" "${TRS_DIR}" "${XGTF_DIR}" "${FACETRACK_DIR}" "${VIDEO_DIR}" "${LANDMARKS_DIR}" "${FACETRACK_TALKINGFACE}"

    #Change to develop to execute demo command
    cd /people/dyab/pyannote-video
    git checkout develop
    #echo Generate demo video\
    #Generate video with talking face annotation
    #echo python "${PYANNOTE_FACE}" demo "${VIDEO_DIR}" "${FACETRACK_TALKINGFACE}" "${VIDEO_OUT_DIR}" --talking-face=True
    python "${PYANNOTE_FACE}" demo "${VIDEO_DIR}" "${FACETRACK_TALKINGFACE}" "${VIDEO_OUT_DIR}" --talking-face=True

    echo ====================================================================
    echo \

done < "$1"
