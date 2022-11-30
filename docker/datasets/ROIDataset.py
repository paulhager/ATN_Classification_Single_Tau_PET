import torch
import csv
from torch.utils.data import Dataset

class roi_dataset(Dataset):
  def __init__(self, path):
    with open(path,'r') as f:
      reader = csv.reader(f)
      self.data = []
      for r in reader:
          r2 = [float(r1) for r1 in r]
          self.data.append(r2)

  def __getitem__(self, idx):
    return torch.tensor(self.data[idx], dtype=torch.float)

  def __len__(self):
    return len(self.data)
