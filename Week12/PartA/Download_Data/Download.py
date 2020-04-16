import os
import zipfile
import requests
import shutil
import progressbar
import math
from tqdm.notebook import tqdm
from io import StringIO,BytesIO
def download_image(url):
  if(os.path.isdir(os.getcwd()+"/tiny-imagenet-200.zip")):
    print("Data already downloaded")
    return
  r = requests.get(url, stream=True)
  print('Downloading'+url)
  zip_ref = zipfile.ZipFile(BytesIO(r.content))
  zip_ref.extractall('./')
  zip_ref.close()

def segregate():
  count1 = 0
  for root, dirs, files in tqdm(os.walk(os.getcwd()+'/tiny-imagenet-200/train/')):  # replace the . with your starting directory
     count = 0
     for file in files:  
        if not os.path.exists(os.getcwd()+'/MergeData/Train/'+file.split('_')[0]+'/'):
             os.makedirs(os.getcwd()+'/MergeData/Train/'+file.split('_')[0]+'/')
        if not os.path.exists(os.getcwd()+'/MergeData/Val/'+file.split('_')[0]+'/'):
             os.makedirs(os.getcwd()+'/MergeData/Val/'+file.split('_')[0]+'/')
        if('txt' not in file):
            if(count<385):
                count=count+1
                shutil.copy(root+'/'+file, os.getcwd()+'/MergeData/Train/'+file.split('_')[0]+'/')
            else:
                shutil.copy(root+'/'+file, os.getcwd()+'/MergeData/Val/'+file.split('_')[0]+'/')
  
  text_file = open(os.getcwd()+"/tiny-imagenet-200/val/val_annotations.txt", "r")
  Lines = text_file.readlines()
  count1 = 0
  for line in tqdm(Lines):
    shutil.copy(os.getcwd()+'/tiny-imagenet-200/val/images'+'/'+line.split('\t')[0], 'MergeData/Val/'+line.split('\t')[1]+'/')
  text_file.close()
