'''
Created on 17.11.2021

@author: grustentier
'''

print('''
               _               ___                          _                                         _   
  _  _   ___  | |  ___  __ __ | __|   __ __ __  ___   _ _  | |__  __ _   _ _   ___   _  _   _ _    __| |  
 | || | / _ \ | | / _ \ \ V / |__ \   \ V  V / / _ \ | '_| | / / / _` | | '_| / _ \ | || | | ' \  / _` |  
  \_, | \___/ |_| \___/  \_/  |___/    \_/\_/  \___/ |_|   |_\_\ \__,_| |_|   \___/  \_,_| |_||_| \__,_|  
  |__/                                                                                                    
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
import yolov5
import imutils
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from modules.objectDetection.yolov5.classes.CocoClasses import CocoClasses

parser = argparse.ArgumentParser(description='Code for analysis of variable sequence motif positions  for different topologies.')
parser.add_argument('--url', default='https://www.youtube.com/watch?v=AdUw5RdyZxI', help='Path to video from stream, local file or default from webcam')
parser.add_argument('--model_path', default=os.path.dirname(__file__) + os.sep + 'models' + os.sep + 'yolov5s.pt', type=str, help='Path to yolov5*.pt model file or other one,')
parser.add_argument('--classes', default='person,car', type=str, help='Comma separated class names like person,bicycle,car,... or class indices like 0,1,2,... , or mixed.')
arguments = parser.parse_args()


def isValidClass(yoloClass, classes):
    for userClass in classes:
        if str(userClass).isnumeric() is False:
            userClass = CocoClasses.getIndex(userClass)
        if int(yoloClass) == int(userClass):
            return True
    return False


def predictByYolo(img, model):
    # perform inference
    # results = model(img)
    
    # inference with larger input size
    results = model(img, size=1280)
    
    # inference with test time augmentation
    # results = model(img, augment=True)
    
    # parse results
    predictions = results.pred[0]
    boxes = predictions[:,:4]  # x1, x2, y1, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    
    return boxes, scores, categories

    
def drawBoundingBox(img, boxes, categories, classesFromParameter):
    for i in range (0, len(categories)):
        cat = categories[i]
        if isValidClass(int(cat), classesFromParameter) is True:
            box = boxes[i]
            if len(box) == 4:
                xB = int(box[2])
                xA = int(box[0])
                yB = int(box[3])
                yA = int(box[1])                
                cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)     


def getVideoCapture(url):
    if os.path.isfile(arguments.url) is True and os.path.exists(arguments.url): 
        return cv2.VideoCapture(arguments.url)
        
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
    assert model_path and len(model_path) > 0 , "Please check your model file path (--model_path)..."
    
    ''' check --classes parameter '''
    assert classes and len(classes) > 0, "Please check your input directory (--classes)..."
    classesFromParameter = []
    if str(classes).find(",") >= 0:
        classesFromParameter = str(classes).strip().split(",")
    elif str(classes).find(";") >= 0:
        classesFromParameter = str(classes).strip().split(";")
    else:
        classesFromParameter.append(str(classes)) 
        
    ''' load yolov5 model '''
    model = yolov5.load(model_path)
    
    ''' get video capture '''
    cap = getVideoCapture(url)    
    
    while(True):
        _, frame = cap.read()
        if frame is None: break       
        boxes, _, categories = predictByYolo(frame, model)
        drawBoundingBox(frame, boxes, categories, classesFromParameter)
        
        cv2.imshow('frame', imutils.resize(frame, width=1024))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    fromVideo(arguments.url, arguments.model_path, arguments.classes)
    
