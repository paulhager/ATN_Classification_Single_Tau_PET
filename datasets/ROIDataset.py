import os
import torch
import csv
from torch.utils.data import Dataset
import torchio as tio

class roi_dataset(Dataset):
  def __init__(self, path, target_type, perturb=0):
    with open(path,'r') as f:
      reader = csv.reader(f)
      self.data = []
      for r in reader:
          r2 = [float(r1) for r1 in r]
          self.data.append(r2)
    self.target_type = target_type
    self.perturb_indx = perturb

  def perturb_item(self, item):
    if self.perturb_indx:
      if self.perturb_indx == 2:
        for i in range(2,8):
          item[0][i] = 0
      else:
        item[0][self.perturb_indx] = 0
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_amyloid(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    item = [torch.tensor(self.data[indx][:209], dtype=torch.float), torch.tensor(self.data[indx][209:], dtype=torch.float)]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_amyloid_dkt(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    item = [torch.tensor(self.data[indx][:114], dtype=torch.float), torch.tensor(self.data[indx][114:], dtype=torch.float)]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_amyloid_dkt_multitask(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    _input = torch.tensor(self.data[indx][:114], dtype=torch.float)
    target = {'regional':torch.tensor(self.data[indx][114:219], dtype=torch.float),
              'positivity':torch.tensor(self.data[indx][219], dtype=torch.long)}
    item = [_input, target]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_tau(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    item = [torch.tensor(self.data[indx][:9]+self.data[indx][209:], dtype=torch.float), torch.tensor(self.data[indx][9:209], dtype=torch.float)]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_gmvolume(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    target = torch.tensor(self.data[indx][209:], dtype=torch.float)
    item = [torch.tensor(self.data[indx][:209], dtype=torch.float), target]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_gmvolume_multitask(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    _input = torch.tensor(self.data[indx][:209], dtype=torch.float)
    target = {'regional':torch.tensor(self.data[indx][209:307], dtype=torch.float),
              'positivity':torch.tensor(self.data[indx][307], dtype=torch.long)}
    item = [_input, target]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_gmvolume_dkt(roi_dataset):
  def __init__(self, path, target_type, perturb=0):
    super().__init__(path, target_type, perturb)

  def __getitem__(self, indx):
    item = [torch.tensor(self.data[indx][:114], dtype=torch.float), torch.tensor(self.data[indx][114:], dtype=torch.float)]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class roi_dataset_gmvolume_noMultiModal(roi_dataset):
  def __init__(self, path, perturb=0):
    super().__init__(path=path)
    self.perturb_indx = perturb

  def __getitem__(self, indx):
    item = [torch.tensor(self.data[indx][9:209], dtype=torch.float), torch.tensor(self.data[indx][209:], dtype=torch.float)]
    item = self.perturb_item(item)
    return item

  def __len__(self):
    return len(self.data)

class image_dataset(Dataset):
  def __init__(self, path, target, target_type, split, augment):
    base = os.path.join('/home/paulhager/Projects/ROIPredictor/data',path,target)
    ims_p = os.path.join(base,'ims_{}.pt'.format(split))
    t = 'targets_{}.pt'.format(split) if target_type == 'roi' else 'targets_binary_{}.pt'.format(split)
    targets_p = os.path.join(base,t)

    self.ims = torch.load(ims_p)
    self.targets = torch.load(targets_p)
    self.augment = augment

  def __getitem__(self, indx):
    im_t = self.ims[indx]
    target_t = self.targets[indx]
    
    if self.augment:
      im_t = self.augment_ims(im_t)

    item = [im_t, target_t]

    return item

  def __len__(self):
    return len(self.targets)

  def augment_ims(self, im):
    transform = tio.Compose([
      tio.RandomBlur(p=0.2, std=3),
      tio.RandomFlip('LR', p=0.2),
      tio.OneOf({
        tio.RandomAffine(scales=0.1, degrees=10): 0.5,
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=15): 0.5
      }, p=0.2)
    ])
    return transform(im)