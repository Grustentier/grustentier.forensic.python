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

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..","..","..",".."))
import cv2
import time
import shutil
import argparse
from general.classes.utils.PathUtils import PathUtils 
from general.classes.utils.StringUtils import StringUtils
from modules.massVideoDataAnalysis.objectDetection.yolov5.classes.CocoClasses import CocoClasses
 
parser = argparse.ArgumentParser(description='Extracting videoframes to pre processed labels from extractLabels.py object detection process.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--label_database', default='', type=str, help='The root directory with included label dirs (--label_export_dir from extractLabels.py).') 
parser.add_argument('--video_database', default='', type=str, help='The root directory with included videos to find')  
parser.add_argument('--classes', default='person,car', type=str, help='Comma separated class names like person,bicycle,car,... or class indices like 0,1,2,... , or mixed.')
parser.add_argument('--probability', default=0.5, type=float, help='Min prediction probability')  
parser.add_argument('--draw_boundingboxes', default=True, type=bool, help='True for drawing bounding boxes around predicted objects')
parser.add_argument('--display', default=False, type=bool, help='Shows predicted video frames')
parser.add_argument('--export_frames', default=True, type=bool, help='True for exporting image frames')
parser.add_argument('--export_boundingboxes', default=False, type=bool, help='True for exporting only predicted boundingbox areas')

arguments = parser.parse_args()  

def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        os.makedirs(exportDirPath, 0o0777)

def getLabelFile2FrameIndex(frameIndex):
    for i in range(0,len(LABEL_FILE_PATHS)):
        labelPath = LABEL_FILE_PATHS[i]
        filename = str(labelPath).split("/")[-1]
        filename = str(filename).split(".")[-2]
        labelIndex = str(filename).split("_")[-1]
        
        if int(labelIndex) == int(frameIndex):
            return labelPath, i
    return None,None

def isValidClass(yoloClass):
    for userClass in CLASSES_FROM_PARAMETER:
        if str(userClass).isnumeric() is False:
            userClass = CocoClasses.getIndex(userClass)
        if int(yoloClass) == int(userClass):
            return True
    return False

def extractFrame():
    if labelPath is None:return    
    playTime = time.strftime('%Hh%Mmin%Ss', time.gmtime(frameIndex/FPS))    
    frame_height, frame_width = frame.shape[:2]      
    file = open(labelPath, 'r')
    lines = file.readlines()
    acceptEntry = False 
    processedClasses = []
    
    for line in lines:
        line = str(line).replace("\n","")
        split = str(line).split(" ")            
        yoloClass = split[0] 
        xcenter = float(split[1])
        ycenter = float(split[2])
        w = float(split[3])
        h = float(split[4])   
        probability = float(split[5])
        
        if isValidClass(yoloClass) is False:continue 
        #if probability is None or str(probability).isnumeric() is False or probability < arguments.probability:continue
        if probability is None or probability < arguments.probability:continue
        
        acceptEntry = True
                
        if CocoClasses.getClass(yoloClass) not in processedClasses:processedClasses.append(CocoClasses.getClass(yoloClass))   
                 
        xcenter = xcenter *  frame_width
        ycenter = ycenter * frame_height
        w = w * frame_width
        h = h * frame_height            
        x1 = xcenter - (w / 2) 
        y1 = ycenter - (h / 2)
         
        if arguments.draw_boundingboxes is True:          
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x1+w),int(y1+h)),(0,255,0),2)
            
        if arguments.export_boundingboxes is True: 
            cv2.imwrite(EXPORT_DIR + "images/" + str(playTime)+"_"+str(frameIndex+1)+"_boundingbox.jpg",frame[int(y1): int(y1+h),int(x1): int(x1+w)] )
            
    file.close()
    
    processedClasses = sorted(processedClasses)
    
    if acceptEntry is True:
        if arguments.export_frames is True:
            cv2.imwrite(EXPORT_DIR + "images/" + str(playTime)+"_"+str(frameIndex+1)+".jpg",frame)
        PROTOCOL.append({"text":"Objekt der Klasse: " + str(processedClasses)+", gefunden  in frame: " + str(frameIndex+1) + ", Spielzeit: " +str(playTime),"frame":frameIndex+1})
        return frame
    
    return None
        
def exportProtocols(protocolData):
    protocol = open(EXPORT_DIR+"protocol.txt", "a")
    tex_protocol = open(EXPORT_DIR+"TeX-protocol.txt", "a")    
    tex_protocol.write("\\begin{enumerate}"+"\n")
    for entry in protocolData:
        protocol.write(entry["text"]+"\n") 
        tex_protocol.write("\\item "+entry["text"]+"\n")
    tex_protocol.write("\\end{enumerate}"+"\n")
    protocol.close()
    tex_protocol.close()    
    
def getSubDirectories(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(getSubDirectories(dirname))
    return sorted(subfolders)
 
def findVideo(parentLabelDir):   
    
    videoName2Find = str(parentLabelDir).split("/")[-1]     
    videoName2Find = StringUtils.cleanString(videoName2Find)
    parentDirVideoName2Find = str(parentLabelDir).split("/")[-2]  
    parentDirVideoName2Find = StringUtils.cleanString(parentDirVideoName2Find)
    
    for videoPath in VIDEO_DATABASE:
        videoName = str(videoPath).split("/")[-1]
        videoName = StringUtils.cleanString(videoName)
        parentDirVideoName = str(videoPath).split("/")[-2]
        parentDirVideoName = StringUtils.cleanString(parentDirVideoName)
        
        if str(parentDirVideoName2Find).lower() == str(parentDirVideoName).lower() and str(videoName2Find).lower() == str(videoName).lower():
            return videoPath;
    return None

def printProgress(numberOfComparations):
    if numberOfComparations % 100 == 0:
        sys.stdout.write("."+str(numberOfComparations)+".\n")
        sys.stdout.flush()
    else:
        sys.stdout.write(".")
        sys.stdout.flush()  

if __name__ == "__main__":  
    
    '''label_database'''
    assert arguments.label_database and len(arguments.label_database) > 0 and os.path.exists(arguments.label_database) and os.path.isdir(arguments.label_database), "Please check your input directory (--label_database)..."
    if arguments.label_database.endswith("/") is False:arguments.label_database+="/"    
    labelDirs = [sub for sub in getSubDirectories(arguments.label_database) if str(sub).split("/")[-1] == "labels"]    
    numberOfProcessedlabelDirs = len(labelDirs)
    
    '''video_database'''
    assert arguments.video_database and len(arguments.video_database) > 0 and os.path.exists(arguments.video_database) and os.path.isdir(arguments.video_database), "Please check your input directory (--video_database)..."
    VIDEO_DATABASE = PathUtils.getFilePaths(arguments.video_database)
     
    '''classes'''
    assert arguments.classes and len(arguments.classes) > 0, "Please check your input directory (--classes)..."
    CLASSES_FROM_PARAMETER = []
    if str(arguments.classes).find(",") >= 0:
        CLASSES_FROM_PARAMETER = str(arguments.classes).strip().split(",")
    elif str(arguments.classes).find(";") >= 0:
        CLASSES_FROM_PARAMETER = str(arguments.classes).strip().split(";")
    else:
        CLASSES_FROM_PARAMETER.append(str(arguments.classes))  
        
     
    for labelDir in labelDirs: 
            
        parentDirPath = os.path.abspath(os.path.join(labelDir, os.pardir)) 
        parentDirName = str(parentDirPath).split("/")[-1]     
        
        print("process label dir of parent",parentDirPath)    
        
        currentVideoPath = findVideo(parentDirPath)
        if currentVideoPath is None:
            print("No video to to label path (",labelDir,") has been found!!!")
            numberOfProcessedlabelDirs = numberOfProcessedlabelDirs - 1
            print("number of label dirs still to be processed:",numberOfProcessedlabelDirs)
            continue               

        LABEL_FILE_PATHS = PathUtils.getFilePaths(labelDir)        
        
        EXPORT_DIR =  parentDirPath + "_by-FoSIL/"  

        if os.path.exists(EXPORT_DIR):
            try:
                print("deleting existing export dir...",EXPORT_DIR)
                shutil.rmtree(EXPORT_DIR)
                print("finished...")
            except OSError as e:
                print("Error: %s : %s" % (EXPORT_DIR, e.strerror))
                
        if len(LABEL_FILE_PATHS) == 0:            
            print("No labels files in path (",labelDir,") has been found!!!")
            numberOfProcessedlabelDirs = numberOfProcessedlabelDirs - 1
            print("number of label dirs still to be processed:",numberOfProcessedlabelDirs)
            continue    

        createExportDir(EXPORT_DIR + "images/") 
        
        video = cv2.VideoCapture(currentVideoPath)
        FPS = video.get(cv2.CAP_PROP_FPS)
        #fourcc = video.get(cv2.CAP_PROP_FOURCC) 
        FRAME_COUNT = int(video.get(cv2.CAP_PROP_FRAME_COUNT))   
        
        PROTOCOL = [] 

        '''
            Dirty Fix. Long time processing of currentvideo. Reason by unable setting, which frame number to process
        '''
        if str(str(currentVideoPath).split(".")[-1]).lower() == "mp4":
        #if str(str(currentVideoPath).split(".")[-1]).lower() is not None:
            
            frameIndex = 0
            while video.isOpened():
                check, frame =  video.read()  
                labelPath,labelFileIndex = getLabelFile2FrameIndex(frameIndex+1)   
                if check is True:
                    frame = extractFrame()
                else:break
                
                if labelFileIndex is not None:
                    LABEL_FILE_PATHS.pop(int(labelFileIndex))
                
                ''' Each label file has been processed than canceling frame reading of current video.'''
                if len(LABEL_FILE_PATHS) == 0:
                    break
                    
                if arguments.display is True and frame is not None:
                    cv2.imshow("Frame", frame)
                    ch = 0xFF & cv2.waitKey(1) # Wait for a second
                    if ch == 27:break 
                    
                frameIndex = frameIndex + 1
                
                FRAME_COUNT = FRAME_COUNT - 1
                printProgress(FRAME_COUNT)
        else: 
            
            '''
            Fast Solution. Only processing relevant video frames from corresponding label index.
            '''
            for labelPath in LABEL_FILE_PATHS:
                filename = str(labelPath).split("/")[-1]
                filename = str(filename)[0:str(filename).rindex(".")] 
                frameIndex = int(str(filename).split("_")[-1]) - 1
                video.set(cv2.CAP_PROP_POS_FRAMES,frameIndex) 
                check, frame =  video.read()    
                if check is True:
                    frame = extractFrame()
        
                if arguments.display is True and frame is not None:
                    cv2.imshow("Frame", frame)
                    ch = 0xFF & cv2.waitKey(1) # Wait for a second
                    if ch == 27:break  
    
        video.release()    
        
        if arguments.display is True:
            cv2.destroyAllWindows() 
        
        exportProtocols(sorted(PROTOCOL, key=lambda x: x["frame"],reverse=False))   

        os.chmod(EXPORT_DIR, 0o777) 
        os.chmod(EXPORT_DIR + "images/", 0o777) 
        os.chmod(EXPORT_DIR + "protocol.txt", 0o777)
        os.chmod(EXPORT_DIR + "TeX-protocol.txt", 0o777)  
                    
        numberOfProcessedlabelDirs = numberOfProcessedlabelDirs - 1
        print("number of label dirs still to be processed:",numberOfProcessedlabelDirs)
    