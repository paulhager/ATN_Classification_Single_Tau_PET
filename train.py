import os
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging, TQDMProgressBar, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from models.ROIPredictor import RoiPredictor
from models.ROIPredictor_Multitask import RoiPredictor_Multitask
from datasets.ROIDataset import roi_dataset_amyloid, roi_dataset_tau, roi_dataset_gmvolume, roi_dataset_gmvolume_noMultiModal, image_dataset, roi_dataset_amyloid_dkt, roi_dataset_gmvolume_dkt, roi_dataset_amyloid_dkt_multitask, roi_dataset_gmvolume_multitask
from Utils import get_next_version


from ray import tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.suggest.hyperopt import HyperOptSearch


torch.multiprocessing.set_sharing_strategy('file_system')
pl.seed_everything(42)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

parser = ArgumentParser()

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--cross_val', type=bool, default=False)
parser.add_argument('--tune', type=bool, default=False)
parser.add_argument('--target', type=str, default='amyloid')
parser.add_argument('--target_type', type=str, default='roi')
parser.add_argument('--splits', type=str, default='splits')
parser.add_argument('--final', type=bool, default=False)
parser.add_argument('--val_check_n', type=int, default=50)
parser.add_argument('--input', type=str, default='csv')
parser.add_argument('--model', type=str, default='efficientnet-b0')
parser.add_argument('--augment', type=bool, default=False)
parser.add_argument('--multitask', type=bool, default=False)
parser = RoiPredictor.add_model_specific_args(parser) # CAUTION: STATIC METHOD. IF ADDING MODEL SPECIFIC ARGS, MUST BE DONE IN THIS CLASS
parser = pl.Trainer.add_argparse_args(parser)

hparams = parser.parse_args()

model = RoiPredictor_Multitask if hparams.multitask else RoiPredictor

model_name = "roi_predictor"
target = 'multitask' if hparams.multitask else hparams.target_type
log_dir = os.path.join("tb_logs",model_name,hparams.input,target)

if hparams.target == 'amyloid':
  if 'dkt' in hparams.splits:
    if hparams.multitask:
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
    if hparams.multitask:
      dataset = roi_dataset_gmvolume_multitask
    else:
      dataset = roi_dataset_gmvolume
elif hparams.target == 'gmvolume_nomm':
  dataset = roi_dataset_gmvolume_noMultiModal
else:
  raise ValueError('Invalid target. Must be tau, amyloid, or gmvolume')



if hparams.input == 'csv':
  dataset_1 = dataset('/home/paulhager/Projects/ROIPredictor/data/{}/split_0.csv'.format(hparams.splits),hparams.target_type)
  dataset_2 = dataset('/home/paulhager/Projects/ROIPredictor/data/{}/split_1.csv'.format(hparams.splits),hparams.target_type)
  dataset_3 = dataset('/home/paulhager/Projects/ROIPredictor/data/{}/split_2.csv'.format(hparams.splits),hparams.target_type)
  dataset_4 = dataset('/home/paulhager/Projects/ROIPredictor/data/{}/split_3.csv'.format(hparams.splits),hparams.target_type)
  dataset_5 = dataset('/home/paulhager/Projects/ROIPredictor/data/{}/split_4.csv'.format(hparams.splits),hparams.target_type)
elif hparams.input == 'image':
  dataset_1 = image_dataset(hparams.splits,hparams.target,hparams.target_type,0,hparams.augment)
  dataset_2 = image_dataset(hparams.splits,hparams.target,hparams.target_type,1,hparams.augment)
  dataset_3 = image_dataset(hparams.splits,hparams.target,hparams.target_type,2,hparams.augment)
  dataset_4 = image_dataset(hparams.splits,hparams.target,hparams.target_type,3,hparams.augment)
  dataset_5 = image_dataset(hparams.splits,hparams.target,hparams.target_type,4,hparams.augment)
  log_dir = os.path.join(log_dir,hparams.model)
else:
  raise ValueError('Invalid input type. Must be csv or image')

all_datasets = [dataset_1,dataset_2,dataset_3,dataset_4,dataset_5]

out_base_dir = os.path.join(os.getcwd(),log_dir,hparams.target)

version = get_next_version(out_base_dir)
cross_val_results = []

def train_tune(config, hparams, tuning=False, dataset_val_i=0):
  if hparams.final:
    i=5

  callbacks = []
  split_folder = os.path.join(out_base_dir,"version_{}".format(version),'split_{}'.format(i))

  if hparams.target_type == 'binary':
    monitor_var = 'val_f1'
    monitor_mode = 'max'
  else:
    monitor_var = 'val_loss'
    monitor_mode = 'min'

  if tuning:
    callbacks.append(TuneReportCallback([monitor_var],on="validation_end"))
    callbacks.append(TQDMProgressBar(refresh_rate=0))
    for k,v in config.items():
      vars(hparams)[k] = v
    logger = TensorBoardLogger(save_dir=tune.get_trial_dir(), default_hp_metric=False)
  else:
    callbacks.append(ModelCheckpoint(monitor=monitor_var,mode=monitor_mode,filename='best_val_loss',dirpath=split_folder))
    callbacks.append(ModelCheckpoint(monitor='val_f1',mode='max',filename='best_val_f1',dirpath=split_folder))
    logger = TensorBoardLogger(log_dir, default_hp_metric=False, name=hparams.target, sub_dir='split_{}'.format(i), version=version)

  dataset_val = all_datasets[dataset_val_i]
  dataset_train = ConcatDataset(all_datasets[:dataset_val_i]+all_datasets[dataset_val_i+1:])
  if hparams.final:
    dataset_train = ConcatDataset(all_datasets)
    dataset_val = ConcatDataset(all_datasets)

  callbacks.append(StochasticWeightAveraging(swa_epoch_start=2001))

  callbacks.append(LearningRateMonitor(logging_interval='epoch'))

  callbacks.append(EarlyStopping(monitor=monitor_var, min_delta=0.00, patience=20, verbose=False, mode=monitor_mode))
  # init model
  predictor = model(hparams)
  train_loader = DataLoader(dataset_train, num_workers=64, batch_size=hparams.batch_size, pin_memory = True, persistent_workers = True)
  val_loader = DataLoader(dataset_val, num_workers=64)

  trainer = pl.Trainer.from_argparse_args(hparams, logger=logger, gpus=1, check_val_every_n_epoch=hparams.val_check_n, callbacks=callbacks)

  trainer.fit(predictor, train_loader, val_loader)

  print('Best val loss for split {} was: {}'.format(dataset_val_i,predictor.best_val_loss))
  cross_val_results.append(predictor.best_val_loss)

  logger.log_hyperparams(predictor.hparams,{'best_val_loss':predictor.best_val_loss})

if hparams.tune:
  config = {
    'num_layers': tune.qrandint(1, 5),
    'num_nodes_per_layer': tune.qrandint(2,100),
    'batch_size': tune.qrandint(1,150),
    'learning_rate': tune.loguniform(1e-6,1e-1),
    #'batch_norm_on': tune.choice([True,False]),
    #'model_name': tune.choice(['resnet10','resnet18','resnet50']),
    'init_type': tune.choice(['normal','kaiming','orthogonal'])
  }

  if hparams.target_type == 'binary':
    monitor_var = 'val_f1'
    monitor_mode = 'max'
  else:
    monitor_var = 'val_loss'
    monitor_mode = 'min'

  train_with_params = tune.with_parameters(train_tune, hparams=hparams, tuning=True)

  hyperopt_search = HyperOptSearch(metric=monitor_var, mode=monitor_mode)

  max_iter_2500_stopper = tune.stopper.MaximumIterationStopper(5000)

  analysis = tune.run(train_with_params,
    resources_per_trial={'cpu':64,'gpu':1},
    metric=monitor_var,
    mode=monitor_mode,
    config=config,
    num_samples=300,
    search_alg=hyperopt_search,
    name='gmvolume_atn_dkt',
    stop=max_iter_2500_stopper
    )

  print('Best hyperparameters found were: ', analysis.best_config)
else:
  for i, dataset_val in enumerate(all_datasets):
    train_tune(None, hparams, dataset_val_i=i)
    if not hparams.cross_val:
      break

  print('Average best val loss over all splits was: {}'.format(sum(cross_val_results)/len(cross_val_results)))

#lr_finder = trainer.tune(predictor, train_loader, val_loader)
#fig = lr_finder['lr_find'].plot()
#fig.savefig('/tmp/lrfinder.png')