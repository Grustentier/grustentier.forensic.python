'''
Created on Nov 22, 2021

@author: grustentier
'''

print('''
         _               _   _   _                             _                                         _   
  _ __  (_) __ __  ___  | | (_) | |__    __ __ __  ___   _ _  | |__  __ _   _ _   ___   _  _   _ _    __| |  
 | '_ \ | | \ \ / / -_) | | | | | '_ \   \ V  V / / _ \ | '_| | / / / _` | | '_| / _ \ | || | | ' \  / _` |  
 | .__/ |_| /_\_\ \___| |_| |_| |_.__/    \_/\_/  \___/ |_|   |_\_\ \__,_| |_|   \___/  \_,_| |_||_| \__,_|  
 |_|                                                                                                         
  _                ___                     _                  _     _                                        
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                            
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                           
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                             
         |__/                                                                                                

''')

import os
import sys
import cv2
import pafy
import time
import imutils
import argparse
import pixellib
from pixellib.instance import instance_segmentation
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

parser = argparse.ArgumentParser(description='Code for analysis of variable sequence motif positions  for different topologies.')
parser.add_argument('--url', default='https://www.youtube.com/watch?v=AdUw5RdyZxI', help='Path to video from stream, local file or default from webcam')
parser.add_argument('--model_path', default=os.path.dirname(__file__) + os.sep + 'models' + os.sep + 'mask_rcnn_coco.h5', type=str, help='Path to mask_rcnn_coco.h5 model file or other one,')
parser.add_argument('--classes', default='person,car', type=str, help='Comma separated class names like person, bicycle, car... etc.')
arguments = parser.parse_args() 


def getVideoCapture(url):
    if os.path.isfile(url) is True and os.path.exists(url): 
        return cv2.VideoCapture(url)
        
    count = 3
    best = None
    while best is None and count > 0:
        try:
            video = pafy.new(url)
            best = video.getbest(preftype="mp4")            
        except:
            count = count - 1
            print("next try for getting best stream in 1 second")
            time.sleep(1)
    
    if best is None:
        return cv2.VideoCapture(0)            
    
    return cv2.VideoCapture(best.url)


def fromVideo(url, model_path, classes):
    ''' check --model path parameter '''
    assert model_path and len(model_path) > 0 and os.path.exists(model_path) and os.path.isfile(model_path), "Please check your model file path (--model_path)..."
    
    ''' check --classes parameter '''
    assert classes and len(classes) > 0, "Please check your input directory (--classes)..."
    classesFromParameter = []
    if str(classes).find(",") >= 0:
        classesFromParameter = str(classes).strip().split(",")
    elif str(classes).find(";") >= 0:
        classesFromParameter = str(classes).strip().split(";")
    else:
        classesFromParameter.append(str(classes)) 
        
    ''' load model '''
    segment_model = instance_segmentation()
    segment_model.load_model(model_path) 
    target_classes = segment_model.select_target_classes()
    
    ''' set valid classes to predict '''
    for currentClass in classesFromParameter:
        assert currentClass in target_classes.keys(), "Select one of the following classes (case sensitive): " + str(target_classes.keys())
        target_classes[currentClass] = "valid"
    
    ''' get video capture '''
    cap = getVideoCapture(url)  
        
    while(True):
        _, frame = cap.read()
        if frame is None: break       
        res = segment_model.segmentFrame(frame, segment_target_classes=target_classes, show_bboxes=True)
        frame = res[1]
        cv2.imshow('frame', imutils.resize(frame, width=1024))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": 
    fromVideo(arguments.url, arguments.model_path, arguments.classes)
    
