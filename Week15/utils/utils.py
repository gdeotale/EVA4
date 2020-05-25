import glob
import cv2
import numpy as np
import os
from google.colab.patches import cv2_imshow
from tqdm.notebook import tqdm
from pathlib import Path

def show(tensors1, tensors2, tensors3, tensors4, tensors5, bools, strs=None):
  grid_tensor1 = torchvision.utils.make_grid(tensors1.detach().cpu())
  grid_image1 = grid_tensor1.permute(1,2,0)
  grid_tensor2 = torchvision.utils.make_grid(tensors2.detach().cpu())
  grid_image2 = grid_tensor2.permute(1,2,0)
  grid_tensor3 = torchvision.utils.make_grid(tensors3.detach().cpu())
  grid_image3 = grid_tensor3.permute(1,2,0)
  grid_tensor4 = torchvision.utils.make_grid(tensors4.detach().cpu())
  grid_image4 = grid_tensor4.permute(1,2,0)
  grid_tensor5 = torchvision.utils.make_grid(tensors5.detach().cpu())
  grid_image5 = grid_tensor5.permute(1,2,0)
  plt.figure(figsize=(100,100))
  plt.subplot(511)
  plt.xlabel('Bg+Fg')
  plt.imshow(grid_image1)
  plt.subplot(512)
  plt.xlabel('Mask')
  plt.imshow(grid_image2)
  plt.subplot(513)
  plt.xlabel('Mask_Pred')
  plt.imshow(grid_image3)
  plt.subplot(514)
  plt.xlabel('Depth')
  plt.imshow(grid_image4)
  plt.subplot(515)
  plt.xlabel('Depth_Pred')
  plt.imshow(grid_image5)
  if(bools==True):
    plt.savefig("Output_test/"+strs+".jpg")
  plt.cla()
  plt.clf()
  plt.close()
  
  #plt.show()
def show1(tensors1):
  grid_tensor1 = torchvision.utils.make_grid(tensors1.detach().cpu())
  grid_image1 = grid_tensor1.permute(1,2,0)
  plt.figure(figsize=(100,100))
  plt.imshow(grid_image1)
  plt.show()

def getmaps():
  maps = {}
  root_dir = '/content/ZData/'
  f1 = Path(root_dir+'Background/')
  f1_files = list(sorted(f1.glob('*.jpg')))
  for i in f1_files:
    num = str(i).split('.')[0].split('_')[1]
    maps[num] = i
  return maps 

def getmaps1(pathg):
  maps = {}
  root_dir = '/content/ZData/'
  f1 = Path(root_dir+pathg)
  f1_files = list(sorted(f1.glob('*/*.jpg')))
  count = 0
  for i in f1_files:
    maps[count] = i
    count = count+1
  return maps 

def getmeanstd(path):
  count = 0
  sum_mean = [0, 0, 0]; sum_std = [0, 0, 0]
  for i in tqdm(glob.glob(path)):
    img = cv2.imread(i)
    r = img[:,:,2]/255
    g = img[:,:,1]/255
    b = img[:,:,0]/255
    sum_mean[0] += np.mean(r)
    sum_std[0] += np.std(r)
    sum_mean[1] += np.mean(g)
    sum_std[1] += np.std(g)
    sum_mean[2] += np.mean(b)
    sum_std[2] += np.std(b)
    count=count+1
    if(count>100000):
      break
    
  sum_mean[0] = sum_mean[0]/(count)
  sum_std[0] = sum_std[0]/count
  sum_mean[1] = sum_mean[1]/count
  sum_std[1] = sum_std[1]/count
  sum_mean[2] = sum_mean[2]/count
  sum_std[2] = sum_std[2]/count
  print("Mean: -",sum_mean)
  print("stdDev: -", sum_std)
  return sum_mean, sum_std

def calculate_iou(target, prediction, thresh):
        intersection = np.logical_and(np.greater(target,thresh), np.greater(prediction,thresh))
        union = np.logical_or(np.greater(target,thresh), np.greater(prediction,thresh))
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )