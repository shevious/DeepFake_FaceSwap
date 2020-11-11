#!/usr/bin/env python

# # DeepFakes 101: Facial Swap (1) - Data Preprocessing
# 
# This script aims to generate a clean training data for deepfake generation by extracting facial data of training subjects via dlib's face_reocgnition. Based on the implementation by Ovalery16, in turn based on the original implementation by ShoanLu.
# 
# Main improvements were focused on compatibility with Google Colaboratory environment as a stand-alone module.
# 
# 
# This notebook utilizes a Google Drive download to access auxilliary supporting image processing scripts. However, this could be replaced with simple git clone command.
# 

# 1. Load dependencies
# ccccoding: uuutf-8

# In[1]:


#Download auxilliary components first


import os
#os.system('gdown https://drive.google.com/uc?id=1O0jrWmtAoSN-W8AwmO0GrqhcYPzkZB2Y')
#os.system('gdown https://drive.google.com/uc?id=1sW6isWgkgiurXtQnYB_iM7OfVlpVlpgV')
#os.system('gdown https://drive.google.com/uc?id=15ilnghI30jH5IxMj9DGRxlI3yb0JxKQM  ')
#os.system('gdown https://drive.google.com/uc?id=1dLC8I9rElNpexVtIxJ5qETTKR2rAlPfk')
#os.system('gdown https://drive.google.com/uc?id=111RTFhYZsDEMWDsegpVZKc3KEqsSQ7mM  ')
#os.system('unzip deepfake_aux.zip')
#os.system('unzip nicolas.zip')
#os.system('unzip tom.zip')
#os.system('unzip chris.zip')


# In[2]:


#Install dependencies if missing

#get_ipython().system('pip install face_recognition')

#get_ipython().system('pip install scandir')
#get_ipython().system('pip install h5py')
#get_ipython().system('pip install opencv-python')
#get_ipython().system('pip install scikit-image')
#get_ipython().system('pip install dlib')
#get_ipython().system('pip install tqdm')


# In[ ]:


import cv2
from pathlib import Path
import face_recognition
from lib_1.PluginLoader import PluginLoader
from lib_1.faces_detect import detect_faces
from lib_1.FaceFilter import FaceFilter
import os
os.system('mkdir -p extracted')
from os import path


# 2. Define directories

# In[ ]:


#input_directory="../content/chris/"  #TODO Change argument here of the input data, should be either chris, tom, or nicolas
#input_directory="chris/"  #TODO Change argument here of the input data, should be either chris, tom, or nicolas
input_directory="obama-jpg"  #TODO Change argument here of the input data, should be either chris, tom, or nicolas



output_directory="../content/obama-cut"


# Define extraction functions

# In[ ]:


def load_filter():
    #filter_file = '../content/filter/chrisfilter.jpg' # TODO Change argument here depending on what youre trying to extract
    filter_file = '../content/filter/obamafilter.jpg' # TODO Change argument here depending on what youre trying to extract
    if os.path.exists(filter_file):
        print('Loading reference image for filtering')
        return FaceFilter(filter_file)
    else:
        print("Filter not detected")

def get_faces(image):
    faces_count = 0
    filterDeepFake = load_filter()
    
    for face in detect_faces(image):
        
        if filterDeepFake is not None and not filterDeepFake.check(face):
            print('Skipping not recognized face!')
            continue
        

        yield faces_count, face


# In[6]:


os.listdir(input_directory)


# We list the image in the input directory and we extract the faces in each of them

# In[7]:


files = [i for i in os.listdir(input_directory)]
         
         
from matplotlib import pyplot as plt
#from google.colab.patches import cv2_imshow

extractor_name = "Align" # TODO Pass as argument
extractor = PluginLoader.get_extractor(extractor_name)()

"""

#Single Example test

example  ="../content/data/CR_2012.jpg"


image = cv2.imread(example)

for idx, face in get_faces(image):
           resized_image = extractor.extract(image, face, 256)
           output_file = output_directory+"/"+str(Path(example).stem)
           cv2.imwrite(str(output_file) + str(idx) + Path(example).suffix, resized_image)
"""
#Simply iterating over the folder is insufficient, imread needs paths, so create them into a list.

def find_all_files(directory):
    for root, dirs, files in os.walk(directory):

        for file in files:
            p=os.path.join(root, file)
            p=p.split("/")[len(p.split("/"))-2]
            name, ext = os.path.splitext(p)

            yield os.path.join(root, file)
folder_img = find_all_files(input_directory)

try:
    for filename in folder_img:
        #print(file)
        #filename = Path(input_directory+file)
        
        #for ii in filename:
            #print(ii)
        #print(filename)
        
        # unicode error
        if 'Chris_Hemsworth_Weight_Loss_' in filename:
            continue
        
        image = cv2.imread(filename)
        
        print(filename)
        for idx, face in get_faces(image):
            print('extractor')
            resized_image = extractor.extract(image, face, 256)
            output_file = output_directory+"/"+str(Path(filename).stem)
            print('imwrite')
            cv2.imwrite(str(output_file) + str(idx) + Path(filename).suffix, resized_image)

except Exception as e:
    print('Failed to extract from image: {}. Reason: {}'.format(filename, e))
    
   
   


# In[8]:


# Zip up results for use later, rename as you see fit. Remember to save in your own drive

#os.system('zip -r extracted_chris.zip extracted')

