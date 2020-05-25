from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image
import cv2

def transform_(mean_, std_):
    transform = transforms.Compose([
                  transforms.Resize((128,128)),
                  transforms.ToTensor(), 
                  transforms.Normalize(mean_, std_),
                  #transforms.ColorJitter(brightness=0.8,saturation=0.0,contrast=0.0,hue=0.0),
                  ])
    return transform

class LoadDataset(Dataset):
  def __init__(self, root_dir, maps, f2, f3, f4, len_G):
    self.meanbg = [0.449,0.367, 0.308]
    self.stdbg = [0.226,0.217,0.208]
    self.meanfgbg = [0.45,0.376,0.322]
    self.stdfgbg = [0.238,0.23,0.224]
    self.meanmask = [0.079]
    self.stdmask = [0.254]
    self.meandepth = [0.35]
    self.stddepth = [0.208]
    self.leng = len_G
    self.f2_files = f2
    self.f3_files = f3
    self.f4_files = f4
    self.transformbg = transform_(self.meanbg, self.stdbg)
    self.transformbgfg = transform_(self.meanfgbg, self.stdfgbg)
    self.transformmask = transform_(self.meanmask, self.stdmask)
    self.transformdepth = transform_(self.meandepth, self.stddepth)
    self.maps= maps
  
  def __len__(self):
    return self.leng
  
  def __getitem__(self, idx):
    num = str(self.f2_files[idx]).split('/')[-1].split('_')[1][1:]
    f1_image = Image.open(self.maps[num])
    f2_image = Image.open(self.f2_files[idx])
    f3_image = Image.open(self.f3_files[idx]).convert("L")
    #f3_image = f3_image.convert(mode='RGB')
    f4_image = Image.open(self.f4_files[idx]).convert("L")
    #f4_image = f4_image.convert(mode='RGB')
    if self.transformbg:
      f1_image = self.transformbg(f1_image)
      f2_image = self.transformbgfg(f2_image)
      f3_image = self.transformmask(f3_image)
      f4_image = self.transformdepth(f4_image)
    return {'f1_image': f1_image,'f2_image': f2_image, 'f3_image': f3_image, 'f4_image': f4_image}