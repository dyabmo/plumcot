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
#Change numeric strings to floats
#Convert from frames to seconds on XGTF

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
speaker_id_temp,startTime_temp,endTime_temp = "","",""
update_list_boolean=False

#Search for <Turn> tags
for turn in trs_tree.xpath('//Section/Turn'):
    for key,value in turn.attrib.items():

        if (key == "speaker"):
            speaker_id_temp = value
            #Only update the list if a "speaker" tag exists, otherwise discard the update
            update_list_boolean = True

        if (key == "startTime"):
            startTime_temp = value

        if (key == "endTime"):
            endTime_temp = value

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
            speaker_id_temp, startTime_temp, endTime_temp = "", "", ""

#Test
#print(speech_turn_list)
#print(len(speech_turn_list))

######################################################################
# Process XGTF file
######################################################################

#Read XGTF file
xmldoc = minidom.parse("/vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.xgtf")

face_data_list = list()
real_name_temp,framespan_temp,start_frame_temp,end_frame_temp = "","","",""
datapoint_list_temp=list()

#Process under some conditions
name_known_boolean = False

#Search for <object> attribute
itemlist = xmldoc.getElementsByTagName('object')
for item in itemlist:
    # Only process objects with name="Personne"
    if( item.attributes['name'].value == "PERSONNE"):

        #Get the framespan
        framespan_temp = item.attributes["framespan"].value

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
                    start_frame_temp = data_item[0].attributes["value"].value

                # Get end frame
                if (attrib.attributes['name'].value == "ENDFRAME"):
                    data_item = attrib.getElementsByTagName("data:dvalue")
                    end_frame_temp = data_item[0].attributes["value"].value

                # Search for attribute TETE to extract polygon coordinates
                if (attrib.attributes['name'].value == "TETE"):

                    #Search for <data:polygon>
                    data_item = attrib.getElementsByTagName("data:polygon")

                    #Search for data:point
                    data_points = data_item[0].getElementsByTagName("data:point")

                    #Print data points
                    for data_point in data_points:
                        datapoint_list_temp.append([data_point.attributes["x"].value,data_point.attributes["y"].value])
                        print(data_point.attributes["x"].value,data_point.attributes["y"].value)


        face_data_list.append([real_name_temp,framespan_temp,start_frame_temp,end_frame_temp,datapoint_list_temp])
        datapoint_list_temp=list()

print(face_data_list)