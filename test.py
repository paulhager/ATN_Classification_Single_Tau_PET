import os
import csv
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.multiprocessing
import pytorch_lightning as pl

from models.ROIPredictor import RoiPredictor
from models.ROIPredictor_Multitask import RoiPredictor_Multitask
from datasets.ROIDataset import roi_dataset_amyloid, roi_dataset_tau, roi_dataset_gmvolume, roi_dataset_gmvolume_noMultiModal, image_dataset, roi_dataset_amyloid_dkt, roi_dataset_gmvolume_dkt, roi_dataset_amyloid_dkt_multitask, roi_dataset_gmvolume_multitask

torch.multiprocessing.set_sharing_strategy('file_system')
pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

parser = ArgumentParser()

parser.add_argument('--test_set', type=int, default=5)
parser.add_argument('--cross_val', type=bool, default=False)
parser.add_argument('--model_name', type=str)
parser.add_argument('--target', type=str, default='amyloid')
parser.add_argument('--perturb_index', type=int, default=0)
parser.add_argument('--splits', type=str)
parser.add_argument('--target_type', type=str, default='roi')
parser.add_argument('--ckpt_target', type=str, default='loss')
parser = RoiPredictor.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)

hparams = parser.parse_args()

model = RoiPredictor_Multitask if hparams.target_type=='multitask' else RoiPredictor

if hparams.target == 'amyloid':
  if 'dkt' in hparams.splits:
    if hparams.target_type=='multitask':
      dataset = roi_dataset_amyloid_dkt_multitask
    else:
      dataset = roi_dataset_amyloid_dkt
  else:
    dataset = roi_dataset_amyloid
elif hparams.target == 'tau':
  dataset = roi_dataset_tau
elif hparams.target == 'gmvolume':
  if 'dkt' in hparams.splits:
    dataset = roi_dataset_gmvolume_dkt
  else:
    if hparams.target_type=='multitask':
      dataset = roi_dataset_gmvolume_multitask
    else:
      dataset = roi_dataset_gmvolume
elif hparams.target == 'gmvolume_nomm':
  dataset = roi_dataset_gmvolume_noMultiModal
else:
  raise ValueError('Invalid target. Must be tau, amyloid, or gmvolume')



trainer = pl.Trainer.from_argparse_args(hparams, logger=False, gpus=1, check_val_every_n_epoch=20)

cross_val_mean_loss = 0
for i in range(5):
  if i==4 and hparams.test_set==5 and not hparams.cross_val:
    i=5
  if not hparams.cross_val:
    if i != hparams.test_set:
      continue
  model_folder = os.path.join(os.getcwd(),'tb_logs',hparams.model_name,'split_{}'.format(i))
  model_path = os.path.join(model_folder,'best_val_{}.ckpt'.format(hparams.ckpt_target))
  predictor = model.load_from_checkpoint(model_path)
  predictor.loss = torch.nn.functional.mse_loss
  
  if 'image' in hparams.model_name:
    dataset_test = image_dataset(hparams.splits,hparams.target,hparams.target_type,i)
  else:
    dataset_test = dataset('/home/paulhager/Projects/ROIPredictor/data/{}/split_{}.csv'.format(hparams.splits,i), hparams.target_type, hparams.perturb_index)
  test_loader = DataLoader(dataset_test, num_workers=64)

  preds = trainer.predict(predictor, test_loader)
  if hparams.target_type == 'multitask':
    regional = []
    positivity = []
    for p in preds:
      regional.append(p[0])
      positivity.append(p[1])
    regional_df = pd.DataFrame(np.squeeze(torch.stack(regional).numpy()))
    positivity_df = pd.DataFrame(np.squeeze(torch.stack(positivity).numpy()),columns=['pred_0','pred_1'])
    ap_df = pd.concat([positivity_df,regional_df],axis=1)
  else:
    all_preds = torch.stack(preds) 
    ap_df = pd.DataFrame(np.squeeze(all_preds.numpy()))
  mean_loss = predictor.loss_sum/len(dataset_test)
  cross_val_mean_loss += mean_loss
  print('Mean Loss for split {} is {}\n'.format(i,mean_loss))

  out_file = os.path.join(model_folder,'preds.csv')
  ap_df.to_csv(out_file, index=False, header=False)
if hparams.cross_val:
  cross_val_mean_loss = cross_val_mean_loss/5
  print('Mean Loss over all splits is {}\n'.format(cross_val_mean_loss))