#############################################################################
#
# LIMSI - Orsay
# Equipe TLP
#
# Authors:
# Mohamed Dyab
#
#############################################################################

###################################################################################
#Usage:
###################################################################################
#Argument 1 : <TRS file path>
#Argument 2 : <XGTF file path>
#Argument 3 : <Face track file path>
#Argument 4 : <Video file path>
#Argument 5 : <Face landmark file path>
#Argument 6 : <Output folder path>

# xml_parser.py <TRS file path> <XGTF file path> <Face track file path> <Video file path> <Face landmark file path> <Output folder path>
# xml_parser.py /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.trs
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.xgtf
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.track.txt
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.MPG
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.landmarks.txt
#               /vol/work1/dyab/BFMTV_CultureEtVous/


###################################################################################
# Imports
###################################################################################
import os
import sys
from lxml import etree
from xml.dom import minidom
import pandas as pd
import numpy as np
from pyannote.video import Video
import scipy.misc

###################################################################################
# Important parameters
###################################################################################

#Set FrameRate constant to 25, then it's , then it's multipied by frame_rate value extracted from XGTF file
FRAME_RATE_CONSTANT = 25.

#Resizing generated face frames, width and height
OUTPUT_IMAGE_HEIGHT = 224
OUTPUT_IMAGE_WIDTH  = 224

#Number of channels
CHANNELS_NUMBER = 3

#Frame error tolerance: when there are several facetracks for the same person
FRAME_ERROR_TOLERANCE = 400.

#Tolerance for time comparison
TIME_TOLERANCE=0.5

#Just for readability
LAST_ELEMENT=-1

DEBUG=False
DEBUG_DETAIL=False

GENERATE_IMAGES=False
GENERATE_FACETRACK_FILE=False

###################################################################################
# Function to create bounding box from polygon points
# Can be used for either XGTF polygon points or facial landmarks points
###################################################################################
def create_bounding_box(polygon_points):

    #left , right , top , bottom
    x_min,x_max,y_min,y_max = 999., 0. ,999., 0.

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

###################################################################################
# Print a list
###################################################################################
def print_list_items(input_list):
    for item in input_list:
        print(item)
    print("Size: "+str(len(input_list)))


###################################################################################
# Read the input files
###################################################################################
def readInputFiles(arguments):

    #Assert argument

    assert len(arguments) == 7, "Error with number of argument : python xml_parser.py <TRS file path> <XGTF file path> <Face track file path> <video file path> <Face landmark file path> <output folder path>"

    try:
        #Read TRS file
        trs_file  = etree.parse(arguments[1])
    except:
        print("Error reading TRS file ")
        raise

    try:
        #Read XGTF file ( A different way than the former!)
        xgtf_file = minidom.parse(arguments[2])

    except:
        print("Error reading XGTF file")
        raise

    try:
        #Read face track file
        face_track_file = pd.read_csv(arguments[3], sep=" ", header = None)
        face_track_file.columns = ["time", "id", "left", "top","right","bottom","state"]
        face_track_file["state"]=0

    except:
        print("Error reading Face track file")
        raise

    try:
        #Read MPG video file to generate images from it
        video = Video(filename = arguments[4])

        # Extract video id from video file path argument
        video_id = arguments[4].split('/')[LAST_ELEMENT].split('.')[0]

    except:
        print("Error reading Video file ")
        raise


    try:
        #Read face track file
        face_landmarks_file = np.loadtxt(arguments[5])

    except:
        print("Error reading Face landmarks file")
        raise

    if not os.path.isdir(arguments[6]):
        os.makedirs(arguments[6])
    output_dir=arguments[6]


    return trs_file, xgtf_file, face_track_file, video, video_id, face_landmarks_file,output_dir

###################################################################################
# Extract number of frames, frame rate, horizontal frame size, vertical frame size.
###################################################################################
def extract_frame_info(xgtf_file):

    num_frames,frame_rate_ratio,h_frame_size,v_frame_size = 0 , 0. , 0 , 0

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
            if(len(node_numframes)==0):
                frame_rate_ratio=1
            else:
                frame_rate_ratio = float(node_numframes[0].attributes['value'].value)

        if (item.attributes["name"].value == "H-FRAME-SIZE"):
            node_numframes = item.getElementsByTagName('data:dvalue')
            h_frame_size = int(node_numframes[0].attributes['value'].value)

        if (item.attributes["name"].value == "V-FRAME-SIZE"):
            node_numframes = item.getElementsByTagName('data:dvalue')
            v_frame_size = int(node_numframes[0].attributes['value'].value)

    #print(num_frames,frame_rate_ratio,h_frame_size,v_frame_size )

    #Frame rate is 25 * the ratio extracted from XGTF file
    frame_rate = frame_rate_ratio * FRAME_RATE_CONSTANT

    return frame_rate, h_frame_size,v_frame_size

######################################################################
# Process TRS file
######################################################################
def process_trs_file(trs_file):
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

    #Create a list of speech turns
    speech_turn_list = list()
    speaker_id_temp= ""
    startTime_temp,endTime_temp =0. , 0.
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
                startTime_temp, endTime_temp = 0. , 0.

    return speech_turn_list

######################################################################
# Process XGTF file
######################################################################
def process_xgtf_file(xgtf_file,frame_rate):

    xgtf_data_list = list()
    real_name_temp = ""
    framespan_temp,start_frame_temp,end_frame_temp = 0. , 0. , 0.
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
                raise NotImplementedError("XGTF unusual file format")

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

                        #Create bounding box from polygon points
                        x_min, y_min, x_max, y_max = create_bounding_box(datapoint_list_temp)

                        #Update the list then
                        xgtf_data_list.append([real_name_temp, framespan_temp, start_frame_temp, end_frame_temp, x_min, y_min, x_max, y_max])

                        #Reset the temporary values
                        datapoint_list_temp=list()
                        real_name_temp, framespan_temp = "", ""
                        start_frame_temp, end_frame_temp = 0. , 0.
                        # Reset boolean after usage
                        name_known_boolean = False

    return xgtf_data_list

#################################################################################
#Now add face coordinates to speech turns
#################################################################################
def combine_face_speech(speech_turn_list,xgtf_data_list):

    face_speech_list = list()
    for speech_turn in speech_turn_list:

        speaker_name = speech_turn[0]
        begin_time = speech_turn[1]
        end_time = speech_turn[2]

        for item in xgtf_data_list:

            face_name = item[0]
            time = item[1]
            appearance_start_time,appearance_end_time = item[2],item[3]
            x_min, y_min, x_max, y_max = item[4],item[5],item[6],item[7]

            if (speaker_name == face_name):

                # The speaker face capture time must have been during his speech, so that we get a talking face trainin point
                #Another case is that the speaker's duration of talking is included in his appearance duration in XGTF file)
                if( (time > begin_time and time < end_time) or
                    (begin_time > appearance_start_time and end_time < appearance_end_time) ):

                    #Handle special case if empty list
                    if(len(face_speech_list) == 0):
                        face_speech_list.append([speaker_name, time,appearance_start_time,appearance_end_time, x_min, y_min, x_max, y_max,list()])
                        face_speech_list[0][LAST_ELEMENT].append([begin_time, end_time])

                    #If name already exists, append the talking time, else: create new entry
                    else:
                        name_found=False
                        for i in range(0,len(face_speech_list)):
                            if ( speaker_name == face_speech_list[i][0] ):
                                name_found = True
                                #ADD time intervarls to the last appended element in the list
                                face_speech_list[i][LAST_ELEMENT].append([begin_time, end_time])

                        if(not name_found):
                            face_speech_list.append([speaker_name, time, appearance_start_time, appearance_end_time, x_min, y_min, x_max, y_max,list()])
                            face_speech_list[LAST_ELEMENT][LAST_ELEMENT].append([begin_time, end_time])
                            name_found=False

    return face_speech_list

######################################################################################
# Process face track file
######################################################################################
def process_facetrack_file(face_track_file,face_speech_list,h_frame_size,v_frame_size):

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

    for face_speech_list_index in range(0,len(face_speech_list)):

        #Several face track IDs might be added for a single entry ..
        face_speech_list[face_speech_list_index].append(list())

        #just for better readability
        time_index, start_time,end_time, x_min_index, y_min_index, x_max_index, y_max_index  = 1, 2, 3, 4, 5, 6, 7

        #List containing face track IDs that are present at the same time frame of face_speech_list
        facetracks_list=list()

        #Search for each face track id in face track file
        #If more than one ID is found, I need to match frame coordinates
        for index in range (0,len(group_time_by_id)):

            #print(str(face_speech_list[face_speech_list_index][time_index]) + " > " + str(group_time_by_id.min().ix[index]['time'])+ " and " + \
            #str(face_speech_list[face_speech_list_index][time_index]) + " < " + str(group_time_by_id.max().ix[index]['time']))

            #Strict assumption: frame capture time has to be between the min and max time of a face track
            if(face_speech_list[face_speech_list_index][time_index] > group_time_by_id.min().ix[index]['time'] and
               face_speech_list[face_speech_list_index][time_index] < group_time_by_id.max().ix[index]['time']):
                facetracks_list.append(group_time_by_id.min().ix[index]['id'])

                if(DEBUG_DETAIL): print(group_time_by_id.min().ix[index]['id'])

            #The previous assumption might not work for every facetrack because one person might have several
            #facetracks in sequence, so we try to match XGTF file time interval with facetrack time interval
            #If facetrack time interval is withing XGTF time interval along with some tolerance, add the face track
            elif((face_speech_list[face_speech_list_index][start_time] ) <= group_time_by_id.min().ix[index]['time'] and
                     (face_speech_list[face_speech_list_index][end_time]) >= group_time_by_id.max().ix[index]['time']):
                facetracks_list.append(group_time_by_id.min().ix[index]['id'])

                if (DEBUG_DETAIL): print("after relaxing assumption:" + str(group_time_by_id.min().ix[index]['id']))

        #=====================================
        #Finished searching for facetracks
        #=======================================

        #print_list_items(facetracks_list)
        #print(facetracks_list)
        if(len(facetracks_list)==1):
            face_speech_list[face_speech_list_index][LAST_ELEMENT].append(int(facetracks_list[0]))

        #If there is more than 1, i.e more than one tracked face in that particular time, find the
        # closest frame coordinate match frame coordinates
        #Note: It might be more than 1 face track for a given face, so take all frames above a threshold, not just
        # the minimum one
        elif(len(facetracks_list)>1):

            for item in facetracks_list:

                #Get the mean of all left,right,top,bottom for each facetrack id
                group_by_id = face_track_file[['left','top','right','bottom', 'id']].groupby('id', as_index=False).mean()

                temp_difference = abs(face_speech_list[face_speech_list_index][x_min_index] - float(group_by_id[group_by_id['id'] == item]['left']) ) + \
                                  abs(face_speech_list[face_speech_list_index][y_min_index] - float(group_by_id[group_by_id['id'] == item]['top'] ))  + \
                                  abs(face_speech_list[face_speech_list_index][x_max_index] - float(group_by_id[group_by_id['id'] == item]['right'])) + \
                                  abs(face_speech_list[face_speech_list_index][y_max_index] - float(group_by_id[group_by_id['id'] == item]['bottom']))

                if (DEBUG_DETAIL): print("Difference: ", str(temp_difference))

                #Add face track ID whenever it's within the frame tolerance
                if(temp_difference < FRAME_ERROR_TOLERANCE):

                    face_track_id = int(item)
                    face_speech_list[face_speech_list_index][LAST_ELEMENT].append(face_track_id)

    return face_speech_list

#################################################################################
# Get bounding box from face landmarks
#################################################################################
def get_landmarks_bounding_box(face_landmarks_file,h_frame_size,v_frame_size):

    face_landmarks_dataframe = pd.DataFrame(columns = ["time", "id","left", "top","right","bottom"])

    for item in face_landmarks_file:

        # Create bounding box from polygon points
        #Put the points in a form of a list to be processed by create_bounding_box()
        #Also scale
        points_list = list()
        coordinates = item[2:]
        for i in range(0, len(coordinates) , 2):

            real_x = coordinates[i] * h_frame_size
            real_y = coordinates[i+1] * v_frame_size
            points_list.append([real_x,real_y])

        x_min, y_min, x_max, y_max = create_bounding_box(points_list)

        item_to_append = pd.Series([item[0], item[1],x_min, y_min, x_max, y_max],index=["time", "id","left", "top","right","bottom"])
        face_landmarks_dataframe = face_landmarks_dataframe.append(item_to_append,ignore_index=True)

    return face_landmarks_dataframe

#################################################################################
#Generate training set as Xv.npy and Y.npy
#################################################################################
def generate_training_set(face_speech_list,face_track_file,face_landmarks_dataframe,output_dir):
    #For each entry in our list:

    print_list_items(face_speech_list)
    for face_speech_item in face_speech_list:

        face_track_id_list = face_speech_item[LAST_ELEMENT]

        # For each face track ID in our entry
        for face_track_id_item in face_track_id_list:

            # Extract only the part of face_track_file belonging to a certain id that we identified as talking-face
            facetrack_id_frames = face_track_file[face_track_file['id'] == face_track_id_item]

            # Extract only the part of face_landmarks_dataframe belonging to a certain id that we identified as talking-face
            facelandmarks_id_frames = face_landmarks_dataframe[face_landmarks_dataframe['id'] == face_track_id_item]

            #Number of frames in this facetrack ID
            image_frames_size = len(facetrack_id_frames)

            #Create a 4-dimensional numpy array containing:
            # (n_frames x height x width x n_channels) containing the sequence of face images
            Xv = np.zeros((image_frames_size, OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT, CHANNELS_NUMBER), dtype=np.uint8)

            #Y contains a 1 - dimensionaL numpy array(n_samples) containing the groundtruth label
            # (0 for not -talking, 1 for talking)
            #default is not talking
            Y = np.zeros(image_frames_size,dtype=np.uint8)

            #Loop on each frame in that extracted part
            for index in range(0, image_frames_size):

                item = facetrack_id_frames.iloc[[index]]

                #Accessing it with time id is more accurate
                #Landmakrs file might be shorter than face track file, so some time entries might not exist!
                face_landmark_item = facelandmarks_id_frames[facelandmarks_id_frames['time'] == float(item['time'])]

                #Get frame using time
                frame_time = float(item['time'])
                frame = video(frame_time)

                union_image = frame[int(item['top']):int(item['bottom']) , int(item['left']):int(item['right'])]

                if ( len(face_landmark_item) > 0):
                    # Just to remove unneeded duplicates
                    face_landmark_item = face_landmark_item.iloc[[0]]

                    # Image that contains the union of facetrack and face landmark!
                    img_top = min(int(face_landmark_item['top']), int(item['top']))
                    img_bottom = max(int(face_landmark_item['bottom']), int(item['bottom']))
                    img_left = min(int(face_landmark_item['left']), int(item['left']))
                    img_right = max(int(face_landmark_item['right']), int(item['right']))

                    union_image = frame[img_top:img_bottom, img_left:img_right]

                union_image = scipy.misc.imresize(union_image, (OUTPUT_IMAGE_WIDTH, OUTPUT_IMAGE_HEIGHT))

                if(GENERATE_IMAGES):
                    if not os.path.isdir('/vol/work1/dyab/frames_test/'+video_id):
                        os.makedirs('/vol/work1/dyab/frames_test/'+video_id)
                    scipy.misc.imsave('/vol/work1/dyab/frames_test/'+video_id+'/' + str(face_track_id_item) + '.' + str(index) +'.jpg', union_image)

                #Add the image to numpy array
                Xv[index,:,:,:] = union_image

                #Add TRUE boolean only for frames withing talking-face
                #Indexs 8 contains the list of talking time intervals
                for item_time in face_speech_item[8]:
                    if(frame_time <= item_time[1] and frame_time >= item_time[0]   ):
                        Y[index] = 1
                        face_track_file.loc[ (face_track_file.id == face_track_id_item) & (face_track_file.time == float(item.time)) , 'state' ] = 1

            #Save Xv after the loop ends and all frames are added
            np.save(output_dir + "/" + video_id + '.' + str(face_track_id_item) + '.Xv.npy', Xv)
            np.save(output_dir + "/" + video_id + '.' + str(face_track_id_item) + '.Y.npy', Y)

            if (DEBUG):
                print(Xv)
                print(Xv.shape)
                print(Y.shape)
                print(Y)

    if(GENERATE_FACETRACK_FILE):
        face_track_file['time'] = face_track_file['time'].map('{:.3f}'.format)
        face_track_file['id'] = face_track_file['id'].map('{:d}'.format)
        face_track_file['left'] = (face_track_file['left'] / h_frame_size).map('{:.3f}'.format)
        face_track_file['right'] = (face_track_file['right'] / h_frame_size).map('{:.3f}'.format)
        face_track_file['top'] = (face_track_file['top'] / v_frame_size).map('{:.3f}'.format)
        face_track_file['bottom'] = (face_track_file['bottom'] / v_frame_size).map('{:.3f}'.format)
        face_track_file.to_csv(output_dir, sep=" ", header=False,index_label = False, index=False)

if __name__ == "__main__":

    # Read input files
    trs_file, xgtf_file, face_track_file, video, video_id, face_landmarks_file, output_dir = readInputFiles(sys.argv)

    #Extract frame rate, horizontal frame size, vertical frame size.
    frame_rate, h_frame_size, v_frame_size = extract_frame_info(xgtf_file)

    #Process TRS file
    speech_turn_list = process_trs_file(trs_file)

    #Process XGTF file
    xgtf_data_list = process_xgtf_file(xgtf_file,frame_rate)

    #Combine face coordinates with speech turns
    face_speech_list_init = combine_face_speech(speech_turn_list, xgtf_data_list)

    #Augment face speech list with face track IDs
    face_speech_list = process_facetrack_file(face_track_file, face_speech_list_init, h_frame_size, v_frame_size)

    #Get landmarks bounding box to combine with face track bounding box
    face_landmarks_dataframe = get_landmarks_bounding_box(face_landmarks_file, h_frame_size, v_frame_size)

    generate_training_set(face_speech_list, face_track_file, face_landmarks_dataframe, output_dir)

    print("################# Speech turns from TRS file #####################")
    print_list_items(speech_turn_list)

    print("################# Face data from XGTF file ####################")
    print_list_items(xgtf_data_list)

    print("##################### Face-speech list ######################################")
    print_list_items(face_speech_list)
