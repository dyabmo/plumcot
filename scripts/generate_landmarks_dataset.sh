#!/bin/bash

chmod 755 /people/dyab/plumcot/scripts/generate_landmarks_dataset.sh


#Check if the script is running on the local machine or the cluster ?
#Host is the local machine
if [ "$HOSTNAME" == "m167" ]; then
    echo "Executing on local machine"
    LINE=`cat "$2" | head -n "$1" | tail -1`
    Y_DIR="$3"
    OUTPUT="$4"
#Host is the cluster
else
    echo "Executing on cluster"
    echo $HOSTNAME
    LINE=`cat "$1" | head -n ${SGE_TASK_ID} | tail -1`
    Y_DIR="$2"
    OUTPUT="$3"
fi

COMPLETE_LINE="/vol/work1/dyab/training_set/residual_local/"${LINE}".landmarks.txt"

(set -x; python /people/dyab/plumcot/scripts/preprocess_coordinates.py "${COMPLETE_LINE}" "${Y_DIR}" "${OUTPUT}" )
