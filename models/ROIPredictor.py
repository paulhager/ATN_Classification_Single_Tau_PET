import pytorch_lightning as pl
from torch.nn import init
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics import F1Score, Accuracy

from efficientnet_pytorch_3d import EfficientNet3D
from MedicalNet import MedicalNet
from models.SimpleCNN import SimpleCNN


class RoiPredictor(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.target == 'amyloid' or self.hparams.target == 'tau':
      if 'dkt' in self.hparams.splits:
        target_size = 105
      else:
        target_size = 200
    else:
      target_size = 98
    if self.hparams.target_type == 'binary':
      target_size = 2

    if 'dkt' in self.hparams.splits:
      input_size = 114
    else:
      if self.hparams.target == 'gmvolume_nomm':
        input_size = 200
      else:
        input_size = 209

    if self.hparams.input == 'csv':
      self.net = self.make_linear_model(input_size, target_size)
      self.net.apply(self.init_weights)
    elif self.hparams.input == 'image':
      if self.hparams.model.startswith("efficient"):
        self.net = EfficientNet3D.from_name(self.hparams.model, override_params={'num_classes':target_size}, in_channels=1)
      elif self.hparams.model=='simpleCNN':
        self.net = SimpleCNN(self.hparams)
        self.net.apply(self.init_weights)
      elif self.hparams.model=="resnet10":
        self.net = MedicalNet(path_to_weights='/home/paulhager/Projects/common/MedicalNet/pretrain/resnet_10_23dataset.pth', device=0, target_size=target_size)
        #self.set_grad()
      elif self.hparams.model=="resnet18":
        self.net = MedicalNet(path_to_weights='/home/paulhager/Projects/common/MedicalNet/pretrain/resnet_18_23dataset.pth', device=0, target_size=target_size)
        #self.set_grad()
      elif self.hparams.model=="resnet50":
        self.net = MedicalNet(path_to_weights='/home/paulhager/Projects/common/MedicalNet/pretrain/resnet_50_23dataset.pth', device=0, target_size=target_size)
        #self.set_grad()
      else:
        raise ValueError("Only efficientnet models or resnet18 or resnet50 are accepted.")
    print(self.net)
    self.best_val_loss = float('Inf')
    self.loss_sum = 0

    if (self.hparams.target=='gmvolume' and self.hparams.target_type=='binary'):
      self.weight = torch.tensor([0.2,0.8],dtype=torch.float, device='cuda:0')
    elif (self.hparams.target=='gmvolume' and self.hparams.target_type=='roi' and self.hparams.weighted_loss):
      w = [1 for i in range(98)]
      w[36] = 10 # 4101
      w[37] = 10 # 4102
      self.weight = torch.tensor(w,dtype=torch.float, device='cuda:0')
    else:
      self.weight = None
    
    if self.hparams.target_type == 'roi':
      if self.hparams.weighted_loss:
        self.loss = self.weighted_mse_loss
      else:
        self.loss = F.mse_loss
    else:
      self.loss = F.cross_entropy

    self.f1 = F1Score(multiclass=False)

  @staticmethod
  def add_model_specific_args(parent_parser):
    parser = parent_parser.add_argument_group("Model")
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_nodes_per_layer', type=int, default=208)
    parser.add_argument('--init_type', type=str, default='kaiming')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr_scheduler_gamma', type=float, default=1)
    parser.add_argument('--batch_norm_on', type=bool, default=False)
    parser.add_argument('--weighted_loss', type=bool, default=False)
    parser.add_argument('--lr_scheduler_step_size', type=float, default=10)
    return parent_parser

  def set_grad(self):
    for param_name, param in self.net.named_parameters():
      if param_name.startswith("conv_seg"):
        param.requires_grad = True
      else:
        param.requires_grad = False

  def init_weights(self, m, init_gain=0.02):
    if isinstance(m, nn.Linear):
      if self.hparams.init_type == 'normal':
        init.normal_(m.weight.data, 0, 0.001)
      elif self.hparams.init_type == 'xavier':
        init.xavier_normal_(m.weight.data, gain=init_gain)
      elif self.hparams.init_type == 'kaiming':
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
      elif self.hparams.init_type == 'orthogonal':
        init.orthogonal_(m.weight.data, gain=init_gain)
      if hasattr(m, 'bias') and m.bias is not None:
        torch.nn.init.constant_(m.bias.data, 0.0)

  def make_linear_model(self, input_size, target_size):
    modules = [nn.Linear(input_size,self.hparams.num_nodes_per_layer)]
    
    if self.hparams.batch_norm_on:
      modules.extend([nn.BatchNorm1d(self.hparams.num_nodes_per_layer)])
    
    for _ in range(self.hparams.num_layers-1):
      modules.extend([nn.ReLU(), nn.Linear(self.hparams.num_nodes_per_layer,self.hparams.num_nodes_per_layer)])
      
      if self.hparams.batch_norm_on:
        modules.extend([nn.BatchNorm1d(self.hparams.num_nodes_per_layer)])

    modules.extend([nn.ReLU(), nn.Linear(self.hparams.num_nodes_per_layer,target_size)])
    net = nn.Sequential(*modules)
    return net

  def forward(self, x): # Prediction
    pred = self.net(x)
    return pred

  def predict_step(self, batch, _):
    x, y = batch
    y_hat = self.net(x)
    self.loss_sum += self.loss(y_hat, y)
    return y_hat

  def training_step(self, batch, _):
    output = self.shared_step(batch)
    loss = output['loss']
    self.log("train_loss", loss, on_epoch=True, on_step=False)
    
    return loss

  def validation_step(self, batch, _):
    output = self.shared_step(batch)
    loss = output['loss']
    self.log("val_loss", loss, on_epoch=True, on_step=False)
    return output

  def validation_epoch_end(self, validation_step_outputs):
    stacked = {k: [dic[k] for dic in validation_step_outputs] for k in validation_step_outputs[0]}
    if self.hparams.target_type == 'binary':
      st_p = torch.stack(stacked['preds'])
      st_g = torch.stack(stacked['gt'])
      t_p = torch.flatten(st_p,start_dim=0,end_dim=1)
      t_g = torch.flatten(st_g)
      self.f1(t_p, t_g)
      self.log("val_f1", self.f1, on_epoch=True, on_step=False)
    mean_val_loss = sum(stacked['loss'])/len(stacked['loss'])
    self.best_val_loss = min(mean_val_loss, self.best_val_loss)

  def shared_step(self, batch):
    x, y = batch
    y_hat = self.net(x)
    if self.hparams.target=='gmvolume' and self.hparams.weighted_loss:
      l = self.loss(y_hat, y, weight=self.weight)
    else:
      l = self.loss(y_hat, y)
    return {'loss':l,'preds':y_hat,'gt':y}

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.hparams.lr_scheduler_step_size*x for x in range(1,4)], gamma=self.hparams.lr_scheduler_gamma)
    return [optimizer], [scheduler]

  def weighted_mse_loss(self, input, target, weight):
    return (weight * (input - target) ** 2).mean()