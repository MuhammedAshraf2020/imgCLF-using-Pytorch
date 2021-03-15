
import os
import torch 
import pandas as pd
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize


class CustomDataSet(Dataset):
  def __init__(self , csv_file , root_dir , transform = None):
    self.annotation = pd.read_csv(csv_file)
    self.root_dir   = root_dir
    self.transform  = transform
  
  def __len__(self):
    return len(self.annotation)
  
  def __getitem__(self , index):
    path    = self.annotation.iloc[index , 1]
    img     = io.imread(path)
    img     = resize(img, (224 , 224))
    y_label = torch.tensor(int(self.annotation.iloc[index , 2]))
    if self.transform:
      img = self.transform(img)
      return img , y_label
