'''
Created on 02.09.2021

@author: Grustentier
'''

print("""
  ___                       ___                                 __  __                      _                
 | __|  __ _   __   ___    |_ _|  _ __    __ _   __ _   ___    |  \/  |  ___   _ _   _ __  | |_    ___   _ _ 
 | _|  / _` | / _| / -_)    | |  | '  \  / _` | / _` | / -_)   | |\/| | / _ \ | '_| | '_ \ | ' \  / -_) | '_|
 |_|   \__,_| \__| \___|   |___| |_|_|_| \__,_| \__, | \___|   |_|  |_| \___/ |_|   | .__/ |_||_| \___| |_|  
                                                |___/                               |_|                      
              _                    __                                                _                       
  _  _   ___ (_)  _ _    __ _     / _|  __ _   __   ___   _ __    ___   _ _   _ __  | |_    ___   _ _        
 | || | (_-< | | | ' \  / _` |   |  _| / _` | / _| / -_) | '  \  / _ \ | '_| | '_ \ | ' \  / -_) | '_|       
  \_,_| /__/ |_| |_||_| \__, |   |_|   \__,_| \__| \___| |_|_|_| \___/ |_|   | .__/ |_||_| \___| |_|         
                        |___/                                                |_|                             
  _                ___                     _                  _     _                                        
 | |__   _  _     / __|  _ _   _  _   ___ | |_   ___   _ _   | |_  (_)  ___   _ _                            
 | '_ \ | || |   | (_ | | '_| | || | (_-< |  _| / -_) | ' \  |  _| | | / -_) | '_|                           
 |_.__/  \_, |    \___| |_|    \_,_| /__/  \__| \___| |_||_|  \__| |_| \___| |_|                             
         |__/                                                                                                

source: https://pypi.org/project/facemorpher/
""")

__version__ = '0.1'

import os
import sys 
import argparse
import facemorpher

parser = argparse.ArgumentParser(description='Code for feature detection.')
parser.add_argument('--input_dir', default=None, help='Path to input dir with images to morph.')
parser.add_argument('--export_dir', default=None, type=str, help='The export/output directory')
parser.add_argument('--display', default=False, type=str, help='True or 1 for displaying morphing results in separate popups.')   
arguments = parser.parse_args()

FILE_TYPES = ["bmp", "jpg", "jpeg", "png"]


def boolean_string(s):
    print(str(s).lower())
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'


def collectImageFilePaths(input_dir):
    filePaths = []
    for root, _, files in os.walk(input_dir): 
        if root.endswith(os.sep) is False:root += os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths


def createExportDir(exportDirPath):
    if os.path.exists(exportDirPath) == False:
        print("Create Directory: " + str(exportDirPath))    
        os.makedirs(exportDirPath, 755)

        
def printProgress(steps, maximum, name="progress", bar_length=20, width=20): 
    if maximum == 0:
        percent = 1.0
    else:
        percent = float(steps) / maximum
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    # sys.stdout.write("\r{0: <{1}} : [{2}]{3}%".format(name, width, arrow + spaces, int(round(percent*100))))
    sys.stdout.write("\r{0: <{1}}[{2}]{3}%".format("", 0, arrow + spaces, int(round(percent * 100))))
    sys.stdout.flush()    
    
    if steps >= maximum: 
        sys.stdout.write('\n\n')         


def morphFaces(input_dir, export_dir, display):
    assert input_dir and input_dir is not None and len(input_dir) > 0 and os.path.exists(input_dir) and os.path.isdir(input_dir), "Please check your input directory (--input_dir)..."
    if input_dir.endswith(os.sep) is False:input_dir += os.sep
    assert export_dir and export_dir is not None and len(export_dir) > 0 , "Please check your export directory (--export_dir)"
    if export_dir.endswith(os.sep) is False:export_dir += os.sep     
    
    createExportDir(export_dir)  
    
    printProgress(0, 2)    
     
    # Get a list of image paths in a folder
    imgpaths = facemorpher.list_imgpaths(input_dir)
    # To morph, supply an array of face images:
    facemorpher.morpher(imgpaths, out_video=export_dir + "morph.avi", plot=boolean_string(display)) 
    
    printProgress(1, 2)
     
    # To average, supply an array of face images:
    facemorpher.averager(collectImageFilePaths(input_dir), out_filename=export_dir + "morph.png", plot=boolean_string(display)) 
     
    printProgress(2, 2)
    
    os.system('chmod 777 -R ' + export_dir)
    print("FINISHED...")

    
if __name__ == "__main__":
    morphFaces(arguments.input_dir, arguments.export_dir, arguments.display)    
