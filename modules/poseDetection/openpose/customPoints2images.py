'''
Created on 11.11.2019

@author: grustentier
'''

print("""


   ___                              ___                      ___              _           ___                                  
  / _ \   _ __   ___   _ _    ___  | _ \  ___   ___  ___    | _ \  ___   ___ | |_   ___  | _ \  _ _   ___   __   ___   ___  ___
 | (_) | | '_ \ / -_) | ' \  |___| |  _/ / _ \ (_-< / -_)   |  _/ / _ \ (_-< |  _| |___| |  _/ | '_| / _ \ / _| / -_) (_-< (_-<
  \___/  | .__/ \___| |_||_|       |_|   \___/ /__/ \___|   |_|   \___/ /__/  \__|       |_|   |_|   \___/ \__| \___| /__/ /__/
         |_|                                                                                                                   
                   _                                  _          _            ___     _                                        
  __   _  _   ___ | |_   ___   _ __      _ __   ___  (_)  _ _   | |_   ___   |_  )   (_)  _ __    __ _   __ _   ___   ___      
 / _| | || | (_-< |  _| / _ \ | '  \    | '_ \ / _ \ | | | ' \  |  _| (_-<    / /    | | | '  \  / _` | / _` | / -_) (_-<      
 \__|  \_,_| /__/  \__| \___/ |_|_|_|   | .__/ \___/ |_| |_||_|  \__| /__/   /___|   |_| |_|_|_| \__,_| \__, | \___| /__/      
                                        |_|                                                             |___/                  
  _                ___                     _                  _     _                                                          
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                                              
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                                             
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                                               
         |__/                                                                                                                  


""")

__version__ = '0.1'

import os
import sys
import cv2  
import json 
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..","..",".."))
from modules.poseDetection.openpose.classes.RigPoints import RigPoints

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--keypoint_dir', type=str, help='The path to parent dir with keypoint json files', default='') 
parser.add_argument('--image_dir', type=str, help='The path to dir with images to label', default='')
parser.add_argument('--export_dir', type=str, help='The export directory path', default='')
parser.add_argument('--circle_radius', type=int, help='The default ground circle radius for marked keypoints', default=3)  
parser.add_argument('--display_ratios', type=str, help='True or 1 for showing keypoint prediction ratios besides keypoints', default=True)
arguments = parser.parse_args()

IMAGE_FILE_TYPES = ["bmp","jpg","jpeg","png"]
KEYPOINT_FILE_TYPES = ["json"]

def boolean_string(s):
    print(str(s).lower())
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'

def collectPoints(keypointFilePaths):  
    processableRigPoints = [] 
    for filePath in keypointFilePaths:
        processableRigPoints.append({"filePath":filePath,"pointData":[]})
        rigPoints4Person = []         
        keyPointFile = open(filePath,"r")          
        jsonObject = json.loads(keyPointFile.read())              
        if len(jsonObject["people"]) > 0:            
            personRigPoints = []
            for peopleIndex in range(0,len(jsonObject["people"])):
                points = jsonObject["people"][peopleIndex]["pose_keypoints_2d"]
                personRigPoints.append(points)
            rigPoints4Person.append(personRigPoints)  
        
        for rigPoints in rigPoints4Person:  
            for rps in rigPoints:  
                processableRigPoints[-1]["pointData"].append(RigPoints.getProcessablePoints(rps)) 
 
    return processableRigPoints    

def points2Image(processableRigPoints4People,frame):   
    for processableRigPoints4Person in processableRigPoints4People: 
        for pointData in processableRigPoints4Person:   
            plotPoints(pointData,frame,4)         
    return frame
    
def plotPoints(pointData,image,thickness=3):     
    coords = pointData[1] 
    
    assert len(coords) >=3,"Missing coord entries. Maybe ratio value is missing or something else"
    
    predictionRatio = round(coords[2],2)

    color = (0,0,0) 
   
    if round(coords[2],1) >= 0.6 and round(coords[2],1) < 0.7:
        color = (4,125,200)
    elif round(coords[2],1) >= 0.7 and round(coords[2],1) < 0.8:
        color = (4,200,200)
    elif round(coords[2],1) >= 0.8 and round(coords[2],1) < 0.9:
        color = (4,200,105)
    elif round(coords[2],1) >= 0.9:
        color = (0,255,0)
    else: color = (0,0,255)        
    
    radius = 3
    
    if arguments.circle_radius is not None:
        radius = arguments.circle_radius * (1 + (1.0-predictionRatio))
        cv2.circle(image, (int(coords[0]),int(coords[1])), int(radius) , color, thickness)
        radius = arguments.circle_radius

    if boolean_string(arguments.display_ratios) is True:
        font = cv2.FONT_HERSHEY_SIMPLEX
        #bottomLeftCornerOfText = (int(coords[0]) + (2*int(radius)),int(coords[1]) + (1*int(radius)))
        fontScale              = 0.4
        fontThickness          = 1 
        fontColor              = color
        #lineType               = 1
        text                   = str(round(coords[2],2))
        
        text_size, _ = cv2.getTextSize(text, font, fontScale, fontThickness)
        text_w, text_h = text_size
        cv2.rectangle(image, (int(coords[0] + 3*int(radius)),int(coords[1])), (int(coords[0]) + text_w + 3*int(radius), int(coords[1]) + text_h + 1), (255,255,255), -1)
        cv2.putText(image, text, (int(coords[0] + 3*int(radius)), int(coords[1]) + text_h), font, fontScale, fontColor, fontThickness)
        
def findImageFilePath(imageFilePaths,keypointFileName):
    for imageFilePath in imageFilePaths: 
        imageFileName = str(imageFilePath).split(os.sep)[-1]
        if "_" in imageFileName:
            imageFileName = str(imageFileName)[0:str(imageFileName).rindex("_")] 
        if "." in imageFileName:
            imageFileName = str(imageFileName).split(".")[0]
         
        if imageFileName == keypointFileName:
            return imageFilePath
    return None

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 755)
        
def printProgress(steps,maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0,int((steps/maximum)*maxSteps2Console)):
        output +="."
    print("["+output+"]", str(int(round((steps/maximum)*100,0)))+"%") 
    
def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.image_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in IMAGE_FILE_TYPES])
    return filePaths

def collectKeypointFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.keypoint_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in KEYPOINT_FILE_TYPES])
    return filePaths

if __name__ == "__main__": 
    assert arguments.image_dir and len(arguments.image_dir) > 0 and os.path.exists(arguments.image_dir) and os.path.isdir(arguments.image_dir), "Please check your input directory (--image_dir)..."
    if arguments.image_dir.endswith(os.sep) is False:arguments.image_dir+=os.sep    
    assert arguments.keypoint_dir and len(arguments.keypoint_dir) > 0 and os.path.exists(arguments.keypoint_dir) and os.path.isdir(arguments.keypoint_dir), "Please check your input directory (--keypoint_dir)..."
    if arguments.keypoint_dir.endswith(os.sep) is False:arguments.keypoint_dir+=os.sep
    assert arguments.export_dir and len(arguments.export_dir) > 0 , "Please check your export directory ..."
    if arguments.export_dir.endswith(os.sep) is False:arguments.export_dir+=os.sep 
        
    createExportDir(arguments.export_dir) 
    
    imageFilePaths = collectImageFilePaths()
    keypointFilePaths = collectKeypointFilePaths()
     
    rigPoints2File = collectPoints(keypointFilePaths)     
    
    step = 0
    printProgress(step,len(rigPoints2File))
    
    for data in rigPoints2File:
        step+=1
        keyointFilePath = data["filePath"]
        keyointFileName = str(keyointFilePath).split(os.sep)[-1]
        if "_" in keyointFileName:
            keyointFileName = str(keyointFileName)[0:str(keyointFileName).rindex("_")] 
        if "." in keyointFileName:
            keyointFileName = str(keyointFileName).split(".")[0]
        
        imageFilePath = findImageFilePath(imageFilePaths,keyointFileName)
        if imageFilePath is None:
            printProgress(step,len(rigPoints2File))
            continue
        
        image2Label = cv2.imread(imageFilePath)
        labeledImage = points2Image(data["pointData"],image2Label)
        
        imageName = str(imageFilePath).split(os.sep)[-1]
        
        cv2.imwrite(arguments.export_dir+imageName,labeledImage)
        
        printProgress(step,len(rigPoints2File))
           
    print("FINISHED...")
        
    
    
    
    