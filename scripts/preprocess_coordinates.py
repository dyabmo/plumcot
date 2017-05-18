import numpy as np
from glob import glob
import sys

LAST_INDEX = -1
FACETRACK_INDEX=1
LANDMARKS_START_INDEX = 2
MOUTH_START_INDEX = 48
LEFT_EYE_INDEX=36
RIGHT_EYE_INDEX=45
NUMBER_OF_MOUTH_POINTS = 20
X=0
Y=1

def preprocess_arguments():

    face_landmarks = sys.argv[1]
    y_labels = sys.argv[2]
    output_directory = sys.argv[3]

    return face_landmarks, y_labels, output_directory

def compute_coordinates(raw_coordinates):

    coordinates_list = list()


    #raw_coordinates -> (c0,c1), (c2,c3), ...#
    #Transform this way for more readably computations
    points = np.asarray(list( zip(raw_coordinates[0::2],raw_coordinates[1::2])) )
    #print(len(points))
    #print(points)

    # horizontal Distance between the two eye points
    d = abs(points[RIGHT_EYE_INDEX,0] - points[LEFT_EYE_INDEX,0])

    # The average of all mouth points
    x = np.mean(points[MOUTH_START_INDEX:, X])
    y = np.mean(points[MOUTH_START_INDEX:, Y])

    #Normalize each point using the average point and the distance rom the two eyes
    for point in points[MOUTH_START_INDEX:]:
        x= (point[X] - x) / d
        coordinates_list.append(x)

        y = (point[Y] - y) / d
        coordinates_list.append(y)

    coordinates  = np.asarray(coordinates_list)
    return coordinates

def preprocess(face_landmarks_dir,y_labels_dir,output_directory):

    face_landmarks_names = glob(face_landmarks_dir+"/*.landmarks.txt")

    #loop on each video using its name
    for face_landmarks_name in face_landmarks_names:

        print(face_landmarks_name)  #load landmark file
        face_landmarks_numpy_array = np.loadtxt(face_landmarks_name,dtype=np.float32)

        video_name = face_landmarks_name.split("/")[LAST_INDEX].split(".")[0]

        #Extract all Y numpy arrays ( all facetrack files) relevant to that file only
        y_numpy_arrays_names = glob(y_labels_dir+video_name+ "*.Y.npy")

        #get the list of facetracks ids
        y_numpy_arrays = list(map(lambda y: y.split("/")[LAST_INDEX].split(".")[1], y_numpy_arrays_names))

        #Loop on each facetrack of the current video
        for y_numpy_array in y_numpy_arrays:

            #print(y_numpy_array)

            #Select entries that only correspond the current facetack id from landmarks file
            boolean  = face_landmarks_numpy_array[:,FACETRACK_INDEX] == int(y_numpy_array)
            facetrack_data = face_landmarks_numpy_array[boolean]
            #print(facetrack_data)

            facetrack_length = len(facetrack_data)
            #print(len(face_landmarks_numpy_array))
            #print(facetrack_length)

            Xv = np.zeros((facetrack_length, NUMBER_OF_MOUTH_POINTS * 2), dtype=np.float32)

            for index in range(facetrack_length):
            #for item in facetrack_data:
                #get the coordinates, ignore the first two values used for time and facetrack number
                raw_coordinates = facetrack_data[index,LANDMARKS_START_INDEX:]

                coordinates = compute_coordinates(raw_coordinates)
                #print(coordinates)
                Xv[index,:] = coordinates

            np.save(output_directory + "/" + video_name + '.' + str(y_numpy_array) + '.XLandmarks.npy', Xv)

if __name__ == "__main__":

    face_landmarks_dir, y_labels_dir, output_directory = preprocess_arguments()

    preprocess( face_landmarks_dir, y_labels_dir,output_directory)