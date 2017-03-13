#############################################################################
#
# LIMSI - Orsay
# Equipe TLP
#
# Authors:
# Mohamed Dyab
#
#############################################################################

#Argument 1 : <XGTF file path>
#Argument 2 : <TRS file path>
#Argument 3 : <Face track file path>

#Usage of xml_parser.py :
# xml_parser.py <TRS file path> <XGTF file path> <Face track file path>
# xml_parser.py /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.trs
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.xgtf
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.track.txt
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.MPG

#TODO
#use template like this to write file #Face track template:
#FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
#                 '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f} '
 #                '{status:s}\n')
#FACE_TEMPLATE.format

# Add exceptions for non existing files
#Check non implemented errors\
#Add non-talking faces
#Generate face frames from track id

import sys
from lxml import etree
from lxml import objectify
from io import StringIO, BytesIO
from xml.dom import minidom
import pandas as pd
import cv2
import numpy as np
from pyannote.video import Video
import scipy.misc


#Assert argument
assert len(sys.argv) == 5, "Error with number of argument : python xml_parser.py <TRS file path> <XGTF file path> <Face track file path> <video file path>"

#Function to create bounding box from polygon points
def create_bounding_box(polygon_points):

    #left , right , top , bottom
    x_min,x_max,y_min,y_max = 999.0, 0.0 ,999.0, 0.0


    for point in polygon_points:

        #process for x
        if (point[0] < x_min ):
            x_min = point[0]
        if (point[0] > x_max):
            x_max = point[0]

        # process for y
        if (point[1] < y_min):
            y_min = point[1]
        if (point[1] > y_max):
            y_max = point[1]

    #Important: return as: left, top, right, bottom so as to be the same as face track file
    return x_min,y_min,x_max,y_max
    #return [[x_min,y_min ],[x_min,y_max],[x_max,y_min],[x_max,y_max]]

###################################################################################
# Read the input files
###################################################################################

#Read TRS file
trs_file  = etree.parse(sys.argv[1])

#Read XGTF file ( A different way than the former!)
xgtf_file = minidom.parse(sys.argv[2])

#Read face track file
face_track_file = pd.read_csv(sys.argv[3], sep=" ", header = None)
face_track_file.columns = ["time", "id", "left", "top","right","bottom","state"]

#Read MPG video file to generate images from it
video = Video(sys.argv[4])

###################################################################################
# Extract number of frames, frame rate, horizontal frame size, vertical frame size.
###################################################################################
#Set FrameRate constant to 25, then it's , then it's multipied by frame_rate value extracted from XGTF file
frame_rate_constant = 25.0

num_frames,frame_rate_ratio,h_frame_size,v_frame_size = 0,0.0,0,0

#Go to <sourcefile>
#       <file>
#        <attribute>
node_source_file= xgtf_file.getElementsByTagName('sourcefile')
node_file_id = node_source_file[0].getElementsByTagName('file')
node_attributes = node_file_id[0].getElementsByTagName('attribute')

for item in node_attributes:
    if(item.attributes["name"].value == "NUMFRAMES"):
        node_numframes = item.getElementsByTagName('data:dvalue')
        num_frames = int(node_numframes[0].attributes['value'].value)

    if (item.attributes["name"].value == "FRAMERATE"):
        node_numframes = item.getElementsByTagName('data:fvalue')
        frame_rate_ratio = float(node_numframes[0].attributes['value'].value)

    if (item.attributes["name"].value == "H-FRAME-SIZE"):
        node_numframes = item.getElementsByTagName('data:dvalue')
        h_frame_size = int(node_numframes[0].attributes['value'].value)

    if (item.attributes["name"].value == "V-FRAME-SIZE"):
        node_numframes = item.getElementsByTagName('data:dvalue')
        v_frame_size = int(node_numframes[0].attributes['value'].value)

#print(num_frames,frame_rate_ratio,h_frame_size,v_frame_size )

#Frame rate is 25 * the ratio extracted from XGTF file
frame_rate = frame_rate_ratio * frame_rate_constant
#print(frame_rate)

######################################################################
# Process TRS file
######################################################################

# Create new dictionary to add speaker ids as key and speaker names as value
speaker_id_name_dict = {}
id_temp,name_temp = "",""

#Parse speaker IDs and names
for speaker in trs_file.xpath('//Speakers/Speaker'):
    for key,value in speaker.attrib.items():

        if(key=="id"):
            id_temp = value

        if(key=="name"):
            name_temp=value

        speaker_id_name_dict[id_temp] = name_temp

#Testing
#print(speaker_id_name_dict)

#Create a list of speech turns
speech_turn_list = list()
speaker_id_temp= ""
startTime_temp,endTime_temp =0.0,0.0
update_list_boolean=False

#Search for <Turn> tags
for turn in trs_file.xpath('//Section/Turn'):
    for key,value in turn.attrib.items():

        if (key == "speaker"):
            speaker_id_temp = value
            #Only update the list if a "speaker" tag exists, otherwise discard the update
            update_list_boolean = True

        if (key == "startTime"):
            startTime_temp = float(value)

        if (key == "endTime"):
            endTime_temp = float(value)

    # Only update the list if a "speaker" tag exists, otherwise discard the update
    if (update_list_boolean):
        update_list_boolean = False

        #Discard speech turns with more than one speaker at the same time
        #If more than one speaker ID exists, they will be separated by space
        #It means that String.split will give a list of size more than 1
        if(len(speaker_id_temp.split()) == 1):

            #Add speaker's real name instead of his/her ID
            real_name = speaker_id_name_dict[speaker_id_temp]
            speech_turn_list.append([real_name,startTime_temp,endTime_temp])
            speaker_id_temp = ""
            startTime_temp, endTime_temp = 0.0, 0.0

#Test
print("################# Speech turns from TRS file #####################")
for item in speech_turn_list:
    print(item)
print(len(speech_turn_list))

######################################################################
# Process XGTF file
######################################################################
face_data_list = list()
real_name_temp = ""
framespan_temp,start_frame_temp,end_frame_temp = 0.0,0.0,0.0
datapoint_list_temp=list()

#Boolean used to process <object> node under some conditions
name_known_boolean = False

#Search for <object> attribute
itemlist = xgtf_file.getElementsByTagName('object')
for item in itemlist:
    # Only process objects with name="Personne"
    if( item.attributes['name'].value == "PERSONNE"):

        #Get the framespan, split on ":" and see if they are the same or not
        framespan1,framespan2 = item.attributes["framespan"].value.split(":")

        #If they are the same, store as float, else raise an exception
        if(framespan1 == framespan2):
            framespan_temp = float(framespan1)

            # change frame number to seconds
            framespan_temp = framespan_temp / frame_rate

        else:
            raise NotImplementedError

        #Get attributes of each Object
        attribs = item.getElementsByTagName('attribute')
        for attrib in attribs:

            #Search for attribute NOM to get the name of the face coordinates
            if (attrib.attributes['name'].value == "NOM"):
                data_item = attrib.getElementsByTagName("data:svalue")

                #Search for first data item since only one data tag exists in this <attribute name="NOM"> tag
                #Exclude "Inconnu
                if("Inconnu" not in data_item[0].attributes["value"].value):
                    real_name_temp = data_item[0].attributes["value"].value
                    name_known_boolean = True


            if (name_known_boolean):

                #Get start frame
                if (attrib.attributes['name'].value == "STARTFRAME"):
                    data_item = attrib.getElementsByTagName("data:dvalue")
                    start_frame_temp = float(data_item[0].attributes["value"].value)

                    #change frame number to seconds
                    start_frame_temp = start_frame_temp/frame_rate


                # Get end frame
                if (attrib.attributes['name'].value == "ENDFRAME"):
                    data_item = attrib.getElementsByTagName("data:dvalue")
                    end_frame_temp = float(data_item[0].attributes["value"].value)

                    # change frame number to seconds
                    end_frame_temp = end_frame_temp / frame_rate

                # Search for attribute TETE to extract polygon coordinates
                if (attrib.attributes['name'].value == "TETE"):

                    #Search for <data:polygon>
                    data_item = attrib.getElementsByTagName("data:polygon")

                    #Search for data:point
                    data_points = data_item[0].getElementsByTagName("data:point")

                    #Print data points
                    for data_point in data_points:
                        datapoint_list_temp.append([ float(data_point.attributes["x"].value),float(data_point.attributes["y"].value)])
                        #print(data_point.attributes["x"].value,data_point.attributes["y"].value)

                    #Create bounding box from polygon points
                    x_min, y_min, x_max, y_max = create_bounding_box(datapoint_list_temp)
                    #print(bounding_box)

                    #Update the list then
                    face_data_list.append([real_name_temp,framespan_temp,start_frame_temp,end_frame_temp,x_min, y_min, x_max, y_max])

                    #Reset the temporary values
                    datapoint_list_temp=list()
                    real_name_temp, framespan_temp = "", ""
                    start_frame_temp, end_frame_temp = 0.0,0.0
                    # Reset boolean after usage
                    name_known_boolean = False

print("###############################################################")
print("################# Face data from XGTF file ####################")
for item in face_data_list:
    print(item)
print(len(face_data_list))


#################################################################################
#Now add face coordinates to speech turns
#################################################################################

face_speech_list = list()
for speech_turn in speech_turn_list:

    speaker_name = speech_turn[0]
    begin_time = speech_turn[1]
    end_time = speech_turn[2]
    has_face_boolean = False
    #talking_face_boolean=False

    for item in face_data_list:

        face_name = item[0]
        time = item[1]
        x_min, y_min, x_max, y_max = item[4],item[5],item[6],item[7]

        #The speaker face capture time must have been during his speech, so that we get a talking face trainin point
        if (speaker_name == face_name and time > begin_time and time < end_time ):

            face_speech_list.append([speaker_name,time,begin_time,end_time,x_min, y_min, x_max, y_max])

            has_face_boolean = True

print("##################### Face-speech list ######################################")
print(face_speech_list)
######################################################################################
# Process face track file
######################################################################################

#Change face coordinates in frame from relative to absolute
face_track_file['left'] = face_track_file['left'] * h_frame_size
face_track_file['right'] = face_track_file['right'] * h_frame_size
face_track_file['top'] = face_track_file['top'] * v_frame_size
face_track_file['bottom'] = face_track_file['bottom'] * v_frame_size

#Match coordinates from xgtf and Face track.
#Match using time coordinate. Take care: there might be two faces at the same time, that's when I will need the face frame coordinates.

#group time by ID, get its min and its max, search for time frame of xgtf file within.
group_time_by_id=face_track_file[['time','id']].groupby('id',as_index=False)

#Search for face tracks corresponding to each entry in "face_speech_list"
#The purpose is to find one face track ID matching current entry of ""face_speech_list"

############### TO BE SET TO ZER0
##################################################

dataset_generation_params = list()

for face_speech_list_index in range(1,len(face_speech_list)):

    #just for better readability
    time_index = 1
    facetracks_list=list()

    #Search for each face track id in face track file
    #If more than one ID is found, I need to match frame coordinates
    for index in range (0,len(group_time_by_id)):

        #frame time has to be between the min and max time of a face track
        if(face_speech_list[face_speech_list_index][time_index] > group_time_by_id.min().ix[index]['time'] and
           face_speech_list[face_speech_list_index][time_index] < group_time_by_id.max().ix[index]['time']):
            print(group_time_by_id.min().ix[index]['id'])
            facetracks_list.append(group_time_by_id.min().ix[index]['id'])

    #Finished search for facetracks
    #If they are more than 1, find the closest frame coordinate match frame coordinates
    if(len(facetracks_list)>1):
        raise NotImplementedError

    face_track_id=int(facetracks_list[0])

    #Now add Name, facetrack ID, boolean is speaking or not, to "dataset_generation_params"
    name = face_speech_list[0]
    is_talking = True
    dataset_generation_params.append([ name,face_track_id,is_talking])

#################################################################################
#Generate frames from video capture


print(face_track_file[face_track_file['id']==62])

images_test = face_track_file[face_track_file['id']==62]
for index in range(0,len(images_test)):

    item = images_test.iloc[[index]]
    frame = video(float(item['time']))

    #indexing= ymin,ymax.xmin,x max
    img = frame[int(item['top']):int(item['bottom']) , int(item['left']):int(item['right'])]
    print(frame.shape)

    scipy.misc.imsave('/vol/work1/dyab/frames_test/outfile'+str(index)+'.jpg', img)

