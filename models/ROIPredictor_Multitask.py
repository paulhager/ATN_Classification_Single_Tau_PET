import pytorch_lightning as pl
from torch.nn import init
from torch import nn
import torch
import torch.nn.functional as F
from torchmetrics import F1Score, AUROC, PrecisionRecallCurve, PearsonCorrCoef

from efficientnet_pytorch_3d import EfficientNet3D
from MedicalNet_multitask import MedicalNet
from models.SimpleCNN import SimpleCNN


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

    if self.hparams.input == 'csv':
      self.encoder, self.regional_head, self.positivity_head = self.make_linear_models(input_size, target_size)
      self.encoder.apply(self.init_weights)
      self.regional_head.apply(self.init_weights)
      self.positivity_head.apply(self.init_weights)
    elif self.hparams.input == 'image':
      if self.hparams.model.startswith("efficient"):
        self.net = EfficientNet3D.from_name(self.hparams.model, override_params={'num_classes':target_size}, in_channels=1)
      elif self.hparams.model=='simpleCNN':
        self.net = SimpleCNN(self.hparams)
        self.net.apply(self.init_weights)
      elif self.hparams.model=="resnet10":
        self.net = MedicalNet(path_to_weights='/home/paulhager/Projects/common/MedicalNet/pretrain/resnet_10_23dataset.pth', device=0, target_size=target_size)
      elif self.hparams.model=="resnet18":
        self.net = MedicalNet(path_to_weights='/home/paulhager/Projects/common/MedicalNet/pretrain/resnet_18_23dataset.pth', device=0, target_size=target_size)
      elif self.hparams.model=="resnet50":
        self.net = MedicalNet(path_to_weights='/home/paulhager/Projects/common/MedicalNet/pretrain/resnet_50_23dataset.pth', device=0, target_size=target_size)
      else:
        raise ValueError("Only efficientnet models or resnet18 or resnet50 are accepted.")
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
    x, y = batch
    y_hat_regional, y_hat_positivity = self.forward(x)
    self.loss_sum += (self.regional_loss(y_hat_regional, y['regional']) + self.positivity_loss(y_hat_positivity, y['positivity']))

    return [y_hat_regional, y_hat_positivity]

  def training_step(self, batch, _):
    output = self.shared_step(batch)
    loss = output['loss']
    self.log("train_loss", loss, on_epoch=True, on_step=False)
    self.log("positivity_loss", output['positivity_loss'], on_epoch=True, on_step=False)
    self.log("regional_loss", output['regional_loss'], on_epoch=True, on_step=False)

    return loss

  def validation_step(self, batch, _):
    output = self.shared_step(batch)
    loss = output['loss']
    self.log("val_loss", loss, on_epoch=True, on_step=False)

    return output

  def validation_epoch_end(self, validation_step_outputs):

    stacked = {k: [dic[k] for dic in validation_step_outputs] for k in validation_step_outputs[0]}
    
    st_p = [p[1] for p in stacked['preds']]
    st_p = torch.stack(st_p)

    st_g = [g['positivity'] for g in stacked['gt']]
    st_g = torch.stack(st_g)

    st_p_regional = [p[0] for p in stacked['preds']]
    st_p_regional = torch.stack(st_p_regional)
    st_p_regional = torch.flatten(st_p_regional)

    st_g_regional = [g['regional'] for g in stacked['gt']]
    st_g_regional = torch.stack(st_g_regional)
    st_g_regional = torch.flatten(st_g_regional)

    self.pearson(st_p_regional, st_g_regional)

    t_p = torch.flatten(st_p,start_dim=0,end_dim=1)
    t_p_sm = nn.functional.softmax(t_p,dim=1)
    t_p_1 = t_p_sm[:,1]
    t_g = torch.flatten(st_g)

    self.auc(t_p_1, t_g)

    precision, recall, _ = self.prc(t_p_1, t_g)
    f1 = self.harmonic_mean_torch(precision, recall)
    best_val_f1 = torch.max(f1)
    #self.f1(t_p, t_g)

    self.log("val_auc", self.auc, on_epoch=True, on_step=False)
    self.log("val_f1", best_val_f1, on_epoch=True, on_step=False)
    self.log("val_pearson", self.pearson, on_epoch=True, on_step=False)

    mean_val_loss = sum(stacked['loss'])/len(stacked['loss'])
    self.best_val_loss = min(mean_val_loss, self.best_val_loss)

  def shared_step(self, batch):
    x, y = batch
    y_hat_regional, y_hat_positivity = self.forward(x)
    if self.hparams.target=='gmvolume' and self.hparams.weighted_loss:
      l_regional = self.regional_loss(y_hat_regional, y['regional'], weight=self.weight)
    else:
      l_regional = self.regional_loss(y_hat_regional, y['regional'])
    l_positivity = (self.positivity_loss(y_hat_positivity, y['positivity'])/5) # Divide by 5 to scale more range of MSE. Positivity loss starts off less important and becomes more important as training progresses
    l = l_positivity + l_regional

    return {'loss':l, 'positivity_loss':l_positivity, 'regional_loss':l_regional, 'preds':[y_hat_regional, y_hat_positivity], 'gt':y}

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[self.hparams.lr_scheduler_step_size*x*x for x in range(1,4)], gamma=self.hparams.lr_scheduler_gamma)

    return [optimizer], [scheduler]

  def weighted_mse_loss(self, input, target, weight):
    return (weight * (input - target) ** 2).mean()

  def harmonic_mean_torch(self, t1, t2):
    return (2*torch.mul(t1,t2))/(t1+t2+1e-10)