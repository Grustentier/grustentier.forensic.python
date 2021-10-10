'''
Created on 02.09.2021

@author: grustentier
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
import argparse
import facemorpher

parser = argparse.ArgumentParser(description='Code for feature detection.')
parser.add_argument('--input_dir', default=None, help='Path to input dir with images to morph.')
parser.add_argument('--export_dir', default=None, type=str, help='The export/output directory')
parser.add_argument('--display', default=False, type=str, help='True or 1 for displaying morphing results in separate popups.')   
arguments = parser.parse_args()

FILE_TYPES = ["bmp","jpg","jpeg","png"]

def boolean_string(s):
    print(str(s).lower())
    if str(s).lower() not in ['false', 'true', '1', '0']:
        raise ValueError('Not a valid boolean string')
    return str(s).lower() == 'true' or str(s).lower() == '1'

def collectImageFilePaths():
    filePaths = []
    for root, _, files in os.walk(arguments.input_dir):  
        if root.endswith(os.sep) is False:root+=os.sep
        filePaths.extend([root + file for file in files if str(file).split(".")[-1].lower() in FILE_TYPES])
    return filePaths

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


if __name__ == "__main__":
    assert arguments.input_dir and arguments.input_dir is not None and len(arguments.input_dir) > 0 and os.path.exists(arguments.input_dir) and os.path.isdir(arguments.input_dir), "Please check your input directory (--input_dir)..."
    if arguments.input_dir.endswith(os.sep) is False:arguments.input_dir+=os.sep
    assert arguments.export_dir and arguments.export_dir is not None and len(arguments.export_dir) > 0 , "Please check your export directory (--export_dir)"
    if arguments.export_dir.endswith(os.sep) is False:arguments.export_dir+=os.sep     
    
    createExportDir(arguments.export_dir)  
    
    printProgress(0,2)
    
     
    # Get a list of image paths in a folder
    imgpaths = facemorpher.list_imgpaths(arguments.input_dir)
    # To morph, supply an array of face images:
    facemorpher.morpher(imgpaths,out_video=arguments.export_dir+"morph.avi", plot=boolean_string(arguments.display)) 
    
    printProgress(1,2)
     
    # To average, supply an array of face images:
    facemorpher.averager(collectImageFilePaths(), out_filename=arguments.export_dir+"morph.png",plot=boolean_string(arguments.display)) 
     
    printProgress(2,2)
    
    os.system('chmod 777 -R ' + arguments.export_dir)
    print("FINISHED...")
    
    