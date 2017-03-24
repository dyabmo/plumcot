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
#TODO
###################################################################################
#Add face landmarks, make sure it's a square, what is the format of landmark file!
#Create bounding box from face landmark points
#fix error messages
#Annotate the video with a cool speaking sprite ..

#Put them in some functions

#put outputs in working directory

#use template like this to write file #Face track template:
#FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
#                 '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f} '
 #                '{status:s}\n')
#FACE_TEMPLATE.format

#Use pickle for persistance later in read_files.py

###################################################################################
#Usage:
###################################################################################
#Argument 1 : <TRS file path>
#Argument 2 : <XGTF file path>
#Argument 3 : <Face track file path>
#Argument 4 : <Video file path>
#Argument 5 : <Face landmark file path>

# xml_parser.py <TRS file path> <XGTF file path> <Face track file path> <Video file path>
# xml_parser.py /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.trs
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.xgtf
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.track.txt
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.MPG
#               /vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.landmarks.txt


###################################################################################
# Imports
###################################################################################
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
frame_rate_constant = 25.

#Resizing generated face frames, width and height
output_image_size_height = 128
output_image_size_width  = 128

#Number of channels
channels_number = 3

#Frame error tolerance: when there are several facetracks for the same person
frame_error_tolerance = 400.  # By experimentation

#Tolerance for time comparison
tolerance_seconds=0.5

#Not important.. just for readability
last_element=-1

debug=False
###################################################################################
# Assert argument
###################################################################################

assert len(sys.argv) == 6, "Error with number of argument : python xml_parser.py <TRS file path> <XGTF file path> <Face track file path> <video file path> <Face landmark file path>"

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
# Read the input files
###################################################################################

try:
    #Read TRS file
    trs_file  = etree.parse(sys.argv[1])
except:
    print("Error reading TRS file ")

try:
    #Read XGTF file ( A different way than the former!)
    xgtf_file = minidom.parse(sys.argv[2])

except:
    print("Error reading XGTF file")

try:
    #Read face track file
    face_track_file = pd.read_csv(sys.argv[3], sep=" ", header = None)
    face_track_file.columns = ["time", "id", "left", "top","right","bottom","state"]
    face_track_file["state"]=0

except:
    print("Error reading Face track file")

try:
    #Read MPG video file to generate images from it
    video = Video(filename = sys.argv[4])

except:
    print("Error reading Video file ")

#Extract video id from video file path argument
video_id = sys.argv[4].split('/')[last_element].split('.')[0]

try:
    #Read face track file
    face_landmarks_file = np.loadtxt(sys.argv[5])

except:
    print("Error reading Face landmarks file")

###################################################################################
# Extract number of frames, frame rate, horizontal frame size, vertical frame size.
###################################################################################

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

#Test
print("################# Speech turns from TRS file #####################")
for item in speech_turn_list:
    print(item)
print("Size: "+str(len(speech_turn_list)))

######################################################################
# Process XGTF file
######################################################################
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
                        #print(data_point.attributes["x"].value,data_point.attributes["y"].value)

                    #Create bounding box from polygon points
                    x_min, y_min, x_max, y_max = create_bounding_box(datapoint_list_temp)
                    #print(bounding_box)

                    #Update the list then
                    xgtf_data_list.append([real_name_temp, framespan_temp, start_frame_temp, end_frame_temp, x_min, y_min, x_max, y_max])

                    #Reset the temporary values
                    datapoint_list_temp=list()
                    real_name_temp, framespan_temp = "", ""
                    start_frame_temp, end_frame_temp = 0. , 0.
                    # Reset boolean after usage
                    name_known_boolean = False

print("#"*100)
print("################# Face data from XGTF file ####################")
for item in xgtf_data_list:
    print(item)
print("Size: " + str(len(xgtf_data_list)))


#################################################################################
#Now add face coordinates to speech turns
#################################################################################

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
                    face_speech_list[0][last_element].append([begin_time, end_time])

                #If name already exists, append the talking time, else: create new entry
                else:
                    name_found=False
                    for i in range(0,len(face_speech_list)):
                        if ( speaker_name == face_speech_list[i][0] ):
                            name_found = True
                            #ADD time intervarls to the last appended element in the list
                            face_speech_list[i][last_element].append([begin_time,end_time])

                    if(not name_found):
                        face_speech_list.append([speaker_name, time, appearance_start_time, appearance_end_time, x_min, y_min, x_max, y_max,list()])
                        face_speech_list[last_element][last_element].append([begin_time, end_time])
                        name_found=False

print("#"*100)
print("##################### Face-speech list ######################################")
for item in face_speech_list:
    print(item)
#print("Size: "+str(len(face_speech_list)))
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

        #Strict assumption: frame capture time has to be between the min and max time of a face track
        if(face_speech_list[face_speech_list_index][time_index] > group_time_by_id.min().ix[index]['time'] and
           face_speech_list[face_speech_list_index][time_index] < group_time_by_id.max().ix[index]['time']):
            facetracks_list.append(group_time_by_id.min().ix[index]['id'])

            if(debug): print(group_time_by_id.min().ix[index]['id'])

        #The previous assumption might not work for every facetrack because one person might have several
        #facetracks in sequence, so we try to match XGTF file time interval with facetrack time interval
        #If facetrack time interval is withing XGTF time interval along with some tolerance, add the face track
        elif((face_speech_list[face_speech_list_index][start_time] ) <= group_time_by_id.min().ix[index]['time'] and
                 (face_speech_list[face_speech_list_index][end_time]) >= group_time_by_id.max().ix[index]['time']):
            facetracks_list.append(group_time_by_id.min().ix[index]['id'])

            if (debug): print("after relaxing assumption:"+str(group_time_by_id.min().ix[index]['id']))

    #=====================================
    #Finished searching for facetracks
    #=======================================

    print(facetracks_list)
    if(len(facetracks_list)==1):
        face_speech_list[face_speech_list_index][last_element].append(int(facetracks_list[0]))

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

            #if (debug):
            print("Difference: ",str(temp_difference))

            #Add face track ID whenever it's within the frame tolerance
            if(temp_difference < frame_error_tolerance):

                face_track_id = int(item)
                face_speech_list[face_speech_list_index][last_element].append(face_track_id)

print("#"*100)
print("##################### Face-speech list with face-track IDs ##################################")
for item in face_speech_list:
    print(item)

#################################################################################
# Get bounding box from face landmarks
#################################################################################

face_landmarks_dataframe = pd.DataFrame(columns = ["time", "id","left", "top","right","bottom"])

for item in face_landmarks_file:

    # Create bounding box from polygon points
    #Put the points in a form of a list to be processed by create_bounding_box()
    #Also scale
    points_list = list()
    coordinates = item[2:]
    for i in range(0, len(coordinates) , 2):

        real_x = coordinates[i] * v_frame_size
        real_y = coordinates[i+1] * h_frame_size
        points_list.append([real_x,real_y])

    x_min, y_min, x_max, y_max = create_bounding_box(points_list)

    item_to_append = pd.Series([item[0], item[1],x_min, y_min, x_max, y_max],index=["time", "id","left", "top","right","bottom"])
    face_landmarks_dataframe = face_landmarks_dataframe.append(item_to_append,ignore_index=True)

if debug:
    print(face_landmarks_dataframe)
    print (len(face_landmarks_dataframe))

#################################################################################
#Generate training set as Xv.npy and Y.npy
#################################################################################

#For each entry in our list:
for face_speech_item in face_speech_list:

    face_track_id_list = face_speech_item[last_element]

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
        Xv = np.zeros((image_frames_size, output_image_size_width, output_image_size_height, channels_number))

        #Y contains a 1 - dimensionaL numpy array(n_samples) containing the groundtruth label
        # (0 for not -talking, 1 for talking)
        #default is not talking
        Y = np.zeros(image_frames_size)

        #Loop on each frame in that extracted part
        for index in range(0, image_frames_size):

            item = facetrack_id_frames.iloc[[index]]

            #Accessing it with time id is more accurate
            #Landmakrs file might be shorter than face track file, so some time entries might not exist!
            face_landmark_item = facelandmarks_id_frames[facelandmarks_id_frames['time'] == float(item['time'])]

            #Get frame using time
            frame_time = float(item['time'])
            frame = video(frame_time)

            if ( len(face_landmark_item) > 0):
                # Just to remove unneeded duplicates
                face_landmark_item = face_landmark_item.iloc[[0]]
                imglandmark = frame[ int(face_landmark_item['left']):int(face_landmark_item['right']),int(face_landmark_item['top']):int(face_landmark_item['bottom'])]
                #scipy.misc.imsave('/vol/work1/dyab/frame_test/' + 'outfile' + str(face_track_id_item) +'.' +str(index) +'.' + 'landmark'+'.jpg', imglandmark)


            #Crop the face coordinates from the frame
            #indexing= ymin,ymax.xmin,x max
            img = frame[int(item['top']):int(item['bottom']) , int(item['left']):int(item['right'])]

            #Resize the cropped image
            img = scipy.misc.imresize(img, (output_image_size_width,output_image_size_height))

            #Add the image to numpy array
            Xv[index,:,:,:] = img

            #Add TRUE boolean only for frames withing talking-face
            #Indexs 8 contains the list of talking time intervals
            for item_time in face_speech_item[8]:
                if(frame_time <= item_time[1] and frame_time >= item_time[0]   ):
                    Y[index] = 1
                    face_track_file.loc[ (face_track_file.id == face_track_id_item) & (face_track_file.time == float(item.time)) , 'state' ] = 1

            #Save the cropped image
            #scipy.misc.imsave('/vol/work1/dyab/frame_test/' + 'outfile' + str(face_track_id_item) +'.' +str(index) + '.jpg', img)

        #Save Xv after the loop ends and all frames are added
        #np.save("/vol/work1/dyab/training_set/"+ video_id +'.'+ str(face_track_id_item) + '.Xv.npy', Xv)
        #np.save("/vol/work1/dyab/training_set/"+ video_id + '.' + str(face_track_id_item) + '.Y.npy', Y)
        if (debug):
            print(Xv.shape)
            print(Xv)
            print(Y.shape)
        #print(Y)

face_track_file['left'] = (face_track_file['left'] / h_frame_size).map('{:.3f}'.format)
face_track_file['right'] = (face_track_file['right'] / h_frame_size).map('{:.3f}'.format)
face_track_file['top'] = (face_track_file['top'] / v_frame_size).map('{:.3f}'.format)
face_track_file['bottom'] = (face_track_file['bottom'] / v_frame_size).map('{:.3f}'.format)


face_track_file.to_csv('face_track_file.csv', sep=" ", header=False,index_label = False, index=False)
