#############################################################################
#
# LIMSI - Orsay
# Equipe TLP
#
# Authors:
# Mohamed Dyab
#
#############################################################################

#Argument 1 :
#Argument 2 :

import sys
from lxml import etree
from lxml import objectify
from io import StringIO, BytesIO
from xml.dom import minidom

#Usage of xml_parser.py :


#TODO
#Enter filenames as arguments
#assert arguments assert len(sys.argv) == 3, "Error with number of argument : python extract-keyframe.py <video_path> <nframes>"
#Intersect TRS and XGTF
#Use herve's tool for intersection

#Set FrameRate for this file = 25, I have to verify this for other files
#I may need to read idx file for each video to know frame rate
frame_rate = 25.0

#Function to create bounding box from polygon points
def create_bounding_box(polygon_points):

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

    return [[x_min,y_min ],[x_min,y_max],[x_max,y_min],[x_max,y_max]]

######################################################################
# Process TRS file
######################################################################

#Read TRS file
trs_tree  = etree.parse("/vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.trs")

# Create new dictionary to add speaker ids as key and speaker names as value
speaker_id_name_dict = {}
id_temp,name_temp = "",""

#Parse speaker IDs and names
for speaker in trs_tree.xpath('//Speakers/Speaker'):
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
for turn in trs_tree.xpath('//Section/Turn'):
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

#Read XGTF file
xmldoc = minidom.parse("/vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.xgtf")

face_data_list = list()
real_name_temp = ""
framespan_temp,start_frame_temp,end_frame_temp = 0.0,0.0,0.0
datapoint_list_temp=list()

#Process under some conditions
name_known_boolean = False

#Search for <object> attribute
itemlist = xmldoc.getElementsByTagName('object')
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
                    bounding_box = create_bounding_box(datapoint_list_temp)
                    #print(bounding_box)

                    #Update the list then
                    face_data_list.append([real_name_temp,framespan_temp,start_frame_temp,end_frame_temp,bounding_box])

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
        face_boundary_box = item[4]

        #The speaker face capture time must have been during his speech, so that we get a talking face trainin point
        if (speaker_name == face_name and time > begin_time and time < end_time ):

            face_speech_list.append([speaker_name,time,begin_time,end_time,face_boundary_box])

            has_face_boolean = True

print("##################### Face-speech list ######################################")
print(face_speech_list)
print(len(face_speech_list))
