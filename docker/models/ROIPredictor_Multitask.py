import pytorch_lightning as pl
from torch.nn import init
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics import AUROC, PrecisionRecallCurve, PearsonCorrCoef

class RoiPredictor_Multitask(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.target == 'amyloid':
      if 'dkt' in self.hparams.splits:
        target_size = 105
      else:
        target_size = 200
    else:
      target_size = 98

    if 'dkt' in self.hparams.splits:
      input_size = 114
    else:
      input_size = 209

    
    self.encoder, self.regional_head, self.positivity_head = self.make_linear_models(input_size, target_size)
    self.encoder.apply(self.init_weights)
    self.regional_head.apply(self.init_weights)
    self.positivity_head.apply(self.init_weights)
    print("ENCODER:")
    print(self.encoder)
    print("REGIONAL HEAD:")
    print(self.regional_head)
    print("POSITIVITY HEAD:")
    print(self.positivity_head)
    self.best_val_loss = float('Inf')
    self.loss_sum = 0

    if (self.hparams.target=='gmvolume')  and (self.hparams.weighted_loss):
      self.positivity_weight = torch.tensor([0.2,0.8],dtype=torch.float, device='cuda:0')
      w = [1 for i in range(98)]
      w[36] = 10 # 4101
      w[37] = 10 # 4102
      self.regional_weight = torch.tensor(w,dtype=torch.float, device='cuda:0')
      self.regional_loss = self.weighted_mse_loss
    else:
      self.regional_weight = None
      self.positivity_weight = None
      self.regional_loss = F.mse_loss
  
    self.positivity_loss = F.cross_entropy

    #self.f1  = F1Score(multiclass=False)
    self.auc = AUROC(num_classes=None)
    self.prc = PrecisionRecallCurve(num_classes=None)
    self.pearson = PearsonCorrCoef()

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

  def make_linear_models(self, input_size, target_size):
    modules = [nn.Linear(input_size,self.hparams.num_nodes_per_layer)]
    
    if self.hparams.batch_norm_on:
      modules.extend([nn.BatchNorm1d(self.hparams.num_nodes_per_layer)])
    
    for _ in range(self.hparams.num_layers-1):
      modules.extend([nn.ReLU(), nn.Linear(self.hparams.num_nodes_per_layer,self.hparams.num_nodes_per_layer)])
      
      if self.hparams.batch_norm_on:
        modules.extend([nn.BatchNorm1d(self.hparams.num_nodes_per_layer)])

    modules.extend([nn.ReLU()])
    encoder = nn.Sequential(*modules)

    regional_head = nn.Sequential(*[nn.Linear(self.hparams.num_nodes_per_layer,target_size)])

    positivity_head = nn.Sequential(*[nn.Linear(self.hparams.num_nodes_per_layer,2)])

    return encoder, regional_head, positivity_head

  def forward(self, x): # Prediction
    encoded_features = self.encoder(x)
    regional_prediction = self.regional_head(encoded_features)
    positivity_prediction = self.positivity_head(encoded_features)

    return regional_prediction, positivity_prediction

  def predict_step(self, batch, _):
    x = batch
    y_hat_regional, y_hat_positivity = self.forward(x)

    return [y_hat_regional, y_hat_positivity]