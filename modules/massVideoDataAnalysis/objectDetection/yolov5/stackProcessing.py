'''
Created on 18.11.2021

@author: grustentier
'''

import os
import sys
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

parser = argparse.ArgumentParser(description='Object detetion within all videos of a give root directory and label information extraction for post processing.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_database', default=None, type=str, help='The root directory including sub dirs and/or video files. In ideal case, videos of type *.avi')
parser.add_argument('--label_database', default=None, type=str, help='The export directory including same folder structure of --video_database (self generated). Processed video files will be represented as directory by given video file name. In this directory predition labels as *.txt files are included within a directory called labels.')
parser.add_argument('--classes', default='person,car', type=str, help='Comma separated class names like person,bicycle,car,... or class indices like 0,1,2,... , or mixed.')
parser.add_argument('--probability', default=0.5, type=float, help='Min prediction probability') 
parser.add_argument('--draw_boundingboxes', default=True, type=str, help='True or 1 for drawing bounding boxes around predicted objects')
parser.add_argument('--display', default=False, type=str, help='True or 1 for showing predicted video frames')
parser.add_argument('--export_frames', default=True, type=str, help='True or 1 for exporting image frames')
parser.add_argument('--export_boundingboxes', default=False, type=str, help='True or 1 for exporting only predicted boundingbox areas')
arguments = parser.parse_args()  

if __name__ == "__main__":
    
    start_time = time.time()
    from modules.massVideoDataAnalysis.objectDetection.yolov5.extractLabels import extractLabels
    extractLabels(arguments.video_database, arguments.label_database, arguments.classes)
    from modules.massVideoDataAnalysis.objectDetection.yolov5.extractFramesFromLabels import extractFramesFromLabels
    extractFramesFromLabels(arguments.label_database, arguments.video_database, arguments.classes, arguments.probability, arguments.draw_boundingboxes, arguments.display, arguments.export_frames, arguments.export_boundingboxes)
     
    print("Processing Time:", time.strftime('%Hh%Mmin%Ss', time.gmtime(time.time() - start_time)))
