'''
Created on 06.08.2021

@author: grustentier
'''
print("""


 __   __   ___    _       ___          ___     _             _            _     ___         _                       _               
 \ \ / /  / _ \  | |     / _ \  __ __ | __|   | |     __ _  | |__   ___  | |   | __| __ __ | |_   _ _   __ _   __  | |_   ___   _ _ 
  \ V /  | (_) | | |__  | (_) | \ V / |__ \   | |__  / _` | | '_ \ / -_) | |   | _|  \ \ / |  _| | '_| / _` | / _| |  _| / _ \ | '_|
   |_|    \___/  |____|  \___/   \_/  |___/   |____| \__,_| |_.__/ \___| |_|   |___| /_\_\  \__| |_|   \__,_| \__|  \__| \___/ |_|  
                                                                                                                                    
  _                ___                     _                  _     _                                                               
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                                                   
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                                                  
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                                                    
         |__/                                                                                                                       


""")

import os
import sys
import shutil
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..","..","..",".."))
from modules.massVideoDataAnalysis.objectDetection.yolov5.classes.CocoClasses import CocoClasses

parser = argparse.ArgumentParser(description='Object detetion within all videos of a give root directory and label information extraction for post processing.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_dir', default='', type=str, help='The root directory including sub dirs and/or video files. In ideal case, videos of type *.avi')
parser.add_argument('--label_export_dir', default=None, type=str, help='The export directory including same folder structure of --video_dir (self generated). Processed video files will be represented as directory by given video file name. In this directory predition labels as *.txt files are included within a directory called labels.')
parser.add_argument('--classes', default='person,car', type=str, help='Comma separated class names like person,bicycle,car,... or class indices like 0,1,2,... , or mixed.')
arguments = parser.parse_args() 

def ignore_files(directory, files):return [f for f in files if os.path.isfile(os.path.join(directory, f))]

def getClasses2PredictFromRequestParameter():
    assert arguments.classes and len(arguments.classes) > 0, "Please check your input directory (--classes)..."
    CLASSES_FROM_PARAMETER = []
    if str(arguments.classes).find(",") >= 0:
        CLASSES_FROM_PARAMETER = str(arguments.classes).strip().split(",")
    elif str(arguments.classes).find(";") >= 0:
        CLASSES_FROM_PARAMETER = str(arguments.classes).strip().split(";")
    else:
        CLASSES_FROM_PARAMETER.append(str(arguments.classes)) 
        
    CLASSES = ""
    for clazz in CLASSES_FROM_PARAMETER:
        if str(clazz).isnumeric():
            CLASSES += " " + str(clazz)
        else:
            CLASSES += " " + str(CocoClasses.getIndex(clazz))
    return CLASSES[1:] 

if __name__ == "__main__":
    assert arguments.video_dir and len(arguments.video_dir) > 0 and os.path.isdir(arguments.video_dir),"Please check your video directory (--video_dir)" 
    if arguments.video_dir.endswith(os.sep) is False:arguments.video_dir+=os.sep  
    assert arguments.label_export_dir and arguments.label_export_dir is not None,"Please check your export directory (--label_export_dir)" 
    if arguments.label_export_dir.endswith(os.sep) is False:arguments.label_export_dir+=os.sep
    
    CLASSES = getClasses2PredictFromRequestParameter() 
    
    if os.path.exists(arguments.label_export_dir):
        try:
            shutil.rmtree(arguments.label_export_dir)
        except OSError as e:
            print("Error: %s : %s" % (arguments.label_export_dir, e.strerror))   
 
    shutil.copytree(arguments.video_dir,arguments.label_export_dir,ignore=ignore_files) 
    
    for root, _, files in os.walk(arguments.video_dir): 
        for file in files:           
            if root.endswith(os.sep) is False:root+=os.sep 
            filePath = root + file
            exportDirPath = arguments.label_export_dir + str(root).replace(arguments.video_dir,"")  
            if exportDirPath.endswith(os.sep) is False:exportDirPath+=os.sep
            #yolo_detect --source filePath --save-txt --save-conf --classes 0 2 --project $2 --nosave --name file --weights yolov5x.pt
            #os.system("yolo_detect --source "+filePath+" --save-txt --save-conf --classes "+CLASSES+" --project "+exportDirPath+" --nosave --name "+file+" --weights yolov5x.pt")
            os.system("yolov5 detect --source "+filePath+" --save-txt --save-conf --classes "+CLASSES+" --project "+exportDirPath+" --nosave --name "+file+" --weights yolov5x.pt")
     
    os.chmod(arguments.label_export_dir, 0o777) 