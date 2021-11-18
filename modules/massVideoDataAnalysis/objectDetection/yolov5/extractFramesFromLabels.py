'''
Created on 14.07.2021

@author: grustentier
''' 

print("""


 __   __   ___    _       ___          ___     ___                                ___         _                       _               
 \ \ / /  / _ \  | |     / _ \  __ __ | __|   | __|  _ _   __ _   _ __    ___    | __| __ __ | |_   _ _   __ _   __  | |_   ___   _ _ 
  \ V /  | (_) | | |__  | (_) | \ V / |__ \   | _|  | '_| / _` | | '  \  / -_)   | _|  \ \ / |  _| | '_| / _` | / _| |  _| / _ \ | '_|
   |_|    \___/  |____|  \___/   \_/  |___/   |_|   |_|   \__,_| |_|_|_| \___|   |___| /_\_\  \__| |_|   \__,_| \__|  \__| \___/ |_|  
                                                                                                                                      
  _                ___                     _                  _     _                                                                 
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                                                     
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                                                    
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                                                      
         |__/                                                                                                                         


""")

import re
import os
import sys
import cv2
import time
import shutil
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
from modules.massVideoDataAnalysis.objectDetection.yolov5.classes.CocoClasses import CocoClasses
 
parser = argparse.ArgumentParser(description='Extracting videoframes to pre processed labels from extractLabels.py object detection process.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--label_database', default=None, type=str, help='The root directory with included label dirs (--label_export_dir from extractLabels.py).') 
parser.add_argument('--video_database', default=None, type=str, help='The root directory with included videos to find')  
parser.add_argument('--classes', default='person,car', type=str, help='Comma separated class names like person,bicycle,car,... or class indices like 0,1,2,... , or mixed.')
parser.add_argument('--probability', default=0.5, type=float, help='Min prediction probability')  
parser.add_argument('--draw_boundingboxes', default=True, type=str, help='True or 1 for drawing bounding boxes around predicted objects')
parser.add_argument('--display', default=False, type=str, help='True or 1 for showing predicted video frames')
parser.add_argument('--export_frames', default=True, type=str, help='True or 1 for exporting image frames')
parser.add_argument('--export_boundingboxes', default=False, type=str, help='True or 1 for exporting only predicted boundingbox areas')
arguments = parser.parse_args()  

LABEL_FILE_TYPES = ["txt"]
VIDEO_FILE_TYPES = ["avi", "mpg", "mpeg", "mp4"]


def boolean_string(s):
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'


def cleanString(string, replacement='_'):
    a = re.sub('[^a-zA-Z0-9.?]', replacement, string) 
    return re.sub(replacement + '+', replacement, a)


def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        os.makedirs(exportDirPath, 0o0777)


def getLabelFile2FrameIndex(frameIndex, labelFilePaths):
    for i in range(0, len(labelFilePaths)):
        labelPath = labelFilePaths[i]
        filename = str(labelPath).split(os.sep)[-1]
        filename = str(filename).split(".")[-2]
        labelIndex = str(filename).split("_")[-1]
        
        if int(labelIndex) == int(frameIndex):
            return labelPath, i
    return None, None


def isValidClass(yoloClass, classes):
    for userClass in classes:
        if str(userClass).isnumeric() is False:
            userClass = CocoClasses.getIndex(userClass)
        if int(yoloClass) == int(userClass):
            return True
    return False


def extractFrame(parameters):    
    if parameters["labelPath"] is None:return    
    playTime = time.strftime('%Hh%Mmin%Ss', time.gmtime(parameters["frameIndex"] / parameters["FPS"]))    
    frame_height, frame_width = parameters["frame"].shape[:2]      
    file = open(parameters["labelPath"], 'r')
    lines = file.readlines()
    acceptEntry = False 
    processedClasses = []
    
    for line in lines:
        line = str(line).replace("\n", "")
        split = str(line).split(" ")            
        yoloClass = split[0] 
        xcenter = float(split[1])
        ycenter = float(split[2])
        w = float(split[3])
        h = float(split[4])   
        probability = float(split[5])
        
        if isValidClass(yoloClass, parameters["classes"]) is False:continue 
        # if probability is None or str(probability).isnumeric() is False or probability < probability:continue
        if probability is None or probability < parameters["probability"]:continue
        
        acceptEntry = True
                
        if CocoClasses.getClass(yoloClass) not in processedClasses:processedClasses.append(CocoClasses.getClass(yoloClass))   
                 
        xcenter = xcenter * frame_width
        ycenter = ycenter * frame_height
        w = w * frame_width
        h = h * frame_height            
        x1 = xcenter - (w / 2) 
        y1 = ycenter - (h / 2)
        
        if boolean_string(parameters["exportBoundingboxes"]) is True: 
            cv2.imwrite(parameters["exportDir"] + "images" + os.sep + "boundingBoxes" + os.sep + str(playTime) + "_" + str(parameters["frameIndex"] + 1) + "_boundingbox.jpg", parameters["frame"][int(y1): int(y1 + h), int(x1): int(x1 + w)]) 

        if boolean_string(parameters["drawBoundingboxes"]) is True:          
            cv2.rectangle(parameters["frame"], (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (0, 255, 0), 2)
            
    file.close()
    
    processedClasses = sorted(processedClasses)
    
    if acceptEntry is True:
        if boolean_string(parameters["exportFrames"]) is True:
            cv2.imwrite(parameters["exportDir"] + "images" + os.sep + str(playTime) + "_" + str(parameters["frameIndex"] + 1) + ".jpg", parameters["frame"])
        parameters["protocoll"].append({"text":"Objekt der Klasse: " + str(processedClasses) + ", gefunden  in frame: " + str(parameters["frameIndex"] + 1) + ", Spielzeit: " + str(playTime), "frame":parameters["frameIndex"] + 1})
        return parameters["frame"]
    
    return None

        
def exportProtocols(protocolData, exportDir):
    protocol = open(exportDir + "protocol.txt", "a")
    tex_protocol = open(exportDir + "TeX-protocol.txt", "a")    
    tex_protocol.write("\\begin{enumerate}" + "\n")
    for entry in protocolData:
        protocol.write(entry["text"] + "\n") 
        tex_protocol.write("\\item " + entry["text"] + "\n")
    tex_protocol.write("\\end{enumerate}" + "\n")
    protocol.close()
    tex_protocol.close()    

    
def getSubDirectories(dirname):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(getSubDirectories(dirname))
    return sorted(subfolders)

 
def findVideo(parentLabelDir, video_database):    
    videoName2Find = str(parentLabelDir).split(os.sep)[-1]     
    videoName2Find = cleanString(videoName2Find)
    parentDirVideoName2Find = str(parentLabelDir).split(os.sep)[-2]  
    parentDirVideoName2Find = cleanString(parentDirVideoName2Find)
    
    for videoPath in video_database:
        videoName = str(videoPath).split(os.sep)[-1]
        videoName = cleanString(videoName)
        parentDirVideoName = str(videoPath).split(os.sep)[-2]
        parentDirVideoName = cleanString(parentDirVideoName)
        #print(parentDirVideoName2Find, parentDirVideoName, videoName2Find, videoName)
        if str(parentDirVideoName2Find).lower() == str(parentDirVideoName).lower() and str(videoName2Find).lower() == str(videoName).lower():
            return videoPath;
    return None


def removeExistingExportDir(exportDirPath):
    if os.path.exists(exportDirPath):
        try:
            print("deleting existing export dir...", exportDirPath)
            shutil.rmtree(exportDirPath)
            print("finished...")
        except OSError as e:
            print("Error: %s : %s" % (exportDirPath, e.strerror))

            
def collectVideoFilePaths(video_database):
    filePaths = []
    for root, _, files in os.walk(video_database):  
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in VIDEO_FILE_TYPES])
    return filePaths  

            
def collectLabelFilePaths(labelDir):
    filePaths = []
    for root, _, files in os.walk(labelDir):  
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in LABEL_FILE_TYPES])
    return filePaths 


def printProgress(steps, maximum):
    output = ""
    maxSteps2Console = 20
    for _ in range(0, int((steps / maximum) * maxSteps2Console)):
        output += "."
    print("[" + output + "]", str(int(round((steps / maximum) * 100, 0))) + "%") 


def extractFramesFromLabels(label_database=None, video_database=None, classes=None, probability=0.5, draw_boundingboxes=True, display=False, export_frames=True, export_boundingboxes=False):    
    '''label_database'''
    assert label_database is not None and len(label_database) > 0 and os.path.exists(label_database) and os.path.isdir(label_database), "Please check your input directory (--label_database)..."
    if label_database.endswith(os.sep) is False:label_database += os.sep    
    LABEL_DIRS = [sub for sub in getSubDirectories(label_database) if str(sub).split(os.sep)[-1] == "labels"]       
    
    '''video_database'''
    assert video_database and len(video_database) > 0 and os.path.exists(video_database) and os.path.isdir(video_database), "Please check your input directory (--video_database)..."
    VIDEO_DATABASE = collectVideoFilePaths(video_database)
     
    '''classes'''
    assert classes and len(classes) > 0, "Please check your input directory (--classes)..."
    CLASSES_FROM_PARAMETER = []
    if str(classes).find(",") >= 0:
        CLASSES_FROM_PARAMETER = str(classes).strip().split(",")
    elif str(classes).find(";") >= 0:
        CLASSES_FROM_PARAMETER = str(classes).strip().split(";")
    else:
        CLASSES_FROM_PARAMETER.append(str(classes))  
        
    step = 0
    printProgress(step, len(LABEL_DIRS))          
    
    for labelDir in LABEL_DIRS: 
        step += 1    
        
        parentDirPath = os.path.abspath(os.path.join(labelDir, os.pardir)) 
        # parentDirName = str(parentDirPath).split(os.sep)[-1]
        
        currentVideoPath = findVideo(parentDirPath, VIDEO_DATABASE)
        if currentVideoPath is None:
            print("No video to to label path (", labelDir, ") has been found!!!")
            printProgress(step, len(LABEL_DIRS))            
            continue               

        LABEL_FILE_PATHS = collectLabelFilePaths(labelDir)
        if len(LABEL_FILE_PATHS) == 0:            
            print("No labels files in path (", labelDir, ") has been found!!!")            
            printProgress(step, len(LABEL_DIRS))
            continue        
        
        EXPORT_DIR = parentDirPath + "_by-FoSIL" + os.sep
        removeExistingExportDir(EXPORT_DIR)
        createExportDir(EXPORT_DIR + "images" + os.sep) 
        if boolean_string(export_boundingboxes) is True: 
            createExportDir(EXPORT_DIR + "images" + os.sep + "boundingBoxes" + os.sep) 
        
        video = cv2.VideoCapture(currentVideoPath)
        FPS = video.get(cv2.CAP_PROP_FPS)
        # fourcc = video.get(cv2.CAP_PROP_FOURCC) 
        # FRAME_COUNT = int(video.get(cv2.CAP_PROP_FRAME_COUNT))   
        
        PROTOCOL = [] 

        '''
            Dirty Fix. Long time processing of currentvideo. Reason by unable setting, which frame number to process
        '''
        if str(str(currentVideoPath).split(".")[-1]).lower() == "mp4":
            
            frameIndex = 0
            while video.isOpened():
                check, frame = video.read()  
                labelPath, labelFileIndex = getLabelFile2FrameIndex(frameIndex + 1, LABEL_FILE_PATHS)   
                if check is True:                    
                    frame = extractFrame({"frame":frame, "classes":CLASSES_FROM_PARAMETER, "labelPath":labelPath, "exportDir":EXPORT_DIR, "frameIndex":frameIndex, "FPS":FPS, "drawBoundingboxes":draw_boundingboxes, "exportFrames":export_frames, "exportBoundingboxes":export_boundingboxes, "probability":probability, "protocoll":PROTOCOL})
                else:break
                
                if labelFileIndex is not None:
                    LABEL_FILE_PATHS.pop(int(labelFileIndex))
                
                ''' Each label file has been processed than canceling frame reading of current video.'''
                if len(LABEL_FILE_PATHS) == 0:
                    break
                    
                if boolean_string(display) is True and frame is not None:
                    cv2.imshow("Frame", frame)
                    ch = 0xFF & cv2.waitKey(1)  # Wait for a second
                    if ch == 27:break 
                    
                frameIndex = frameIndex + 1 
        else: 
            
            '''
            Fast Solution. Only processing relevant video frames from corresponding label index.
            '''
            for labelPath in LABEL_FILE_PATHS:
                filename = str(labelPath).split(os.sep)[-1]
                filename = str(filename)[0:str(filename).rindex(".")] 
                frameIndex = int(str(filename).split("_")[-1]) - 1
                video.set(cv2.CAP_PROP_POS_FRAMES, frameIndex) 
                check, frame = video.read()    
                if check is True:
                    frame = extractFrame({"frame":frame, "classes":CLASSES_FROM_PARAMETER, "labelPath":labelPath, "exportDir":EXPORT_DIR, "frameIndex":frameIndex, "FPS":FPS, "drawBoundingboxes":draw_boundingboxes, "exportFrames":export_frames, "exportBoundingboxes":export_boundingboxes, "probability":probability, "protocoll":PROTOCOL})
        
                if boolean_string(display) is True and frame is not None:
                    cv2.imshow("Frame", frame)
                    ch = 0xFF & cv2.waitKey(1)  # Wait for a second
                    if ch == 27:break  
    
        video.release()    
        
        if boolean_string(display) is True:cv2.destroyAllWindows() 
        
        exportProtocols(sorted(PROTOCOL, key=lambda x: x["frame"], reverse=False), EXPORT_DIR)   
        
        os.system('chmod 777 -R ' + EXPORT_DIR)    
        
        printProgress(step, len(LABEL_DIRS))
        
    print("FINISHED...")

        
if __name__ == "__main__":
    extractFramesFromLabels(arguments.label_database, arguments.video_database, arguments.classes, arguments.probability, arguments.draw_boundingboxes, arguments.display, arguments.export_frames, arguments.export_boundingboxes)
    
