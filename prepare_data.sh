#!/bin/bash

#TODO: 
#Make paths in the form of variables
#Process AVI or MPG ?

printf "Starting to split videos into shots"
#To split video into shots:
/people/dyab/pyannote-video/scripts/pyannote-structure.py shot /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.MPG /vol/work1/dyab/shot_output.json

printf "Finished splitting video into shots"

printf "Starting to track faces according to splitted shots"
#To track faces according to splitted shots
#Note: setting the interval is very important for quick processing!
/people/dyab/pyannote-video/scripts/pyannote-face.py track --every=0.5 /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.MPG /vol/work1/dyab/shot_output.json /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.track.txt
printf "Finished tracking faces according to splitted shots"

printf "Starting to generate face tracks demo video"
#To view the result of face tracking on video
/people/dyab/pyannote-video/scripts/pyannote-face.py demo /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.MPG /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.track.txt /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.OUTPUT.mp4
printf "Finished generating face tracks demo video"
