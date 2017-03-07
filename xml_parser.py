from lxml import etree
from lxml import objectify
from io import StringIO, BytesIO

#TODO
#Parameterize the filenames
#READ XGTF File

tree = etree.parse("/vol/work1/dyab/BFMTV_CultureEtVous_2012-04-16_065040.trs")
#print(etree.tostring(tree.getroot()))

# Create new dictionary to add speaker ids as key and speaker names as value
speaker_id_name_dict = {}
id_temp,name_temp = "",""

#Parse speaker IDs and names
for speaker in tree.xpath('//Speakers/Speaker'):
    for key,value in speaker.attrib.items():

        if(key=="id"):
            id_temp = value

        if(key=="name"):
            name_temp=value

        speaker_id_name_dict[id_temp] = name_temp

#Testing
print(speaker_id_name_dict["spk1"])
print(speaker_id_name_dict)

#Create a list of speech turns
speech_turn_list = list()
speaker_id_temp,startTime_temp,endTime_temp = "","",""
update_list_boolean=False

#Search for <Turn> tags
for turn in tree.xpath('//Section/Turn'):
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
print(speech_turn_list)
print(len(speech_turn_list))