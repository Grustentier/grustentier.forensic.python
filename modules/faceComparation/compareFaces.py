'''
Created on 18.01.2021

@author: grustentier
'''
print("""
  ___                        ___                                            _     _              
 | __|  __ _   __   ___     / __|  ___   _ __    _ __   __ _   _ _   __ _  | |_  (_)  ___   _ _  
 | _|  / _` | / _| / -_)   | (__  / _ \ | '  \  | '_ \ / _` | | '_| / _` | |  _| | | / _ \ | ' \ 
 |_|   \__,_| \__| \___|    \___| \___/ |_|_|_| | .__/ \__,_| |_|   \__,_|  \__| |_| \___/ |_||_|
                                                |_|                                              
  _                ___                     _                  _     _                            
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|               
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                 
         |__/                                                                                    

""")

__version__ = '0.1'

import os
import cv2
import sys
import numpy   
import argparse
import face_recognition 

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

parser = argparse.ArgumentParser(description='Comparation of faces ', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--database', default=None, type=str, help='The image database')
parser.add_argument('--inputdir', default=None, type=str, help='The input directoy with face images to compare with database')
parser.add_argument('--exportdir', default=None, type=str, help='The image to compare with database')
arguments = parser.parse_args() 

FILE_TYPES = ["bmp", "jpg", "jpeg", "png"]


def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]


def collectImageFilePaths(input_dir):
    filePaths = []
    for root, _, files in os.walk(input_dir): 
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths


def boolean_string(s):
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'


def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 755)


def compare(known_face_encodings, known_face_names): 
    for path in collectImageFilePaths(arguments.database):
        if path.endswith("gitignore"):continue
        fileName = path.split(os.sep)[-1].split(".")[0]       
        frame = cv2.imread(path)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:,:,::-1]        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) 
    
        if len(face_locations) == 0:continue        
       
        for face_encoding in face_encodings:
            # matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = numpy.argmin(face_distances)
            probability = str(round(1.0 - face_distances[best_match_index], 2))            
            exportMarkedFace(face_locations, known_face_names[best_match_index], probability, frame.copy(), fileName)        


def exportMarkedFace(faceLocations, faceName, probability, frame, fileName): 
    flIndex = 0 
    for faceLocation in faceLocations:
        (top, right, bottom, left) = faceLocation    
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, faceName + " " + str(probability), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        print("Found", "\"" + faceName + "\"", "with", probability, "in image", "\"" + str(fileName).split(os.sep)[-1] + "\"")
        cv2.imwrite(arguments.exportdir + str(probability) + "-" + fileName + "-" + faceName + "_" + str(flIndex) + ".png", frame) 
        flIndex += 1


def printProgress(steps, maximum, name="progress", bar_length=20, width=20): 
    if maximum == 0:
        percent = 1.0
    else:
        percent = float(steps) / maximum
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write("\r{0: <{1}}[{2}]{3}%".format("", 0, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()    
    
    if steps >= maximum: 
        sys.stdout.write('\n\n')


def compareFaces():
    assert arguments.database and len(arguments.database) > 0 and os.path.exists(arguments.database) and os.path.isdir(arguments.database), "Please check your database directory ..."
    assert arguments.inputdir and len(arguments.inputdir) > 0 and os.path.exists(arguments.inputdir) and os.path.isdir(arguments.inputdir), "Please check your input directory ..."
    assert arguments.exportdir and len(arguments.exportdir) > 0 , "Please check your export directory ..."
    if arguments.exportdir.endswith(os.sep) is False:arguments.exportdir += os.sep     
    
    createExportDir(arguments.exportdir) 
    
    known_face_encodings = []
    known_face_names = []
    
    imageFilePaths = collectImageFilePaths(arguments.inputdir)
    
    printProgress(0, len(imageFilePaths))
    
    for i in range(0, len(imageFilePaths)):
        inputPath = imageFilePaths[i]
        fileName = inputPath.split(os.sep)[-1].split(".")[0]  
        known_identity_image = face_recognition.load_image_file(inputPath)
        known_identity_face_encodings = face_recognition.face_encodings(known_identity_image)
        
        if len(known_identity_face_encodings) == 1: 
            known_face_encodings.append(known_identity_face_encodings[0])
            known_face_names.append(fileName)   
            
        printProgress(i, len(imageFilePaths)) 
    
    print("")
    compare(known_face_encodings, known_face_names)  


if __name__ == "__main__":
    compareFaces()
    
