from torch import nn
import torch
import functools

class SimpleCNN(nn.Module):
  def __init__(self, hparams, norm_layer=nn.BatchNorm3d):
    super(SimpleCNN, self).__init__()

    hparams.use_dropout = False
    hparams.maxpool = True
    input_nc = 1

    if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
      use_bias = norm_layer.func == nn.InstanceNorm3d
    else:
      use_bias = norm_layer == nn.InstanceNorm3d

    ndf = 8
    n_layers = 7
    nfc = 64
    dropout_prob = 0.1

    layers_to_flat_mni = {8: 2*2*2, 7: 4*4*3, 6: 7*8*6, 5: 13*15*12}
    #layers_to_flat_standardized = {8: 2*2*2, 7: 3*3*3, 6: 6*6*6, 5: 12*12*12, 4: 24*24*24}

    layers_to_flat = layers_to_flat_mni

    conv_sequence = [nn.Conv3d(input_nc, ndf, kernel_size=5, stride=1, padding=1), nn.LeakyReLU(0.2, True)]
    nf_mult = 1
    nf_mult_prev = 1
    for n in range(1, n_layers):  # gradually increase the number of filters
      nf_mult_prev = nf_mult
      nf_mult = int(min(2 ** n, 512/ndf))
      if hparams.maxpool:
        conv_sequence += [
          nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=1, bias=use_bias),
          nn.LeakyReLU(0.2, True),
          norm_layer(ndf * nf_mult),
          nn.MaxPool3d(2)
        ]
      else:
        conv_sequence += [
          nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=5, stride=2, padding=2, bias=use_bias),
          nn.LeakyReLU(0.2, True),
          norm_layer(ndf * nf_mult)
        ]
    if hparams.maxpool:
      conv_sequence += [nn.AdaptiveAvgPool3d(1)]
    filters = (ndf * nf_mult)
    size = layers_to_flat[n_layers]
    flatLength = filters if hparams.maxpool else int(filters*size)
    fc_sequence = [nn.Linear(flatLength, nfc),
                    nn.ReLU(True)]
    
    if hparams.use_dropout:
      fc_sequence += [nn.Dropout(dropout_prob)]
    
    fc_sequence += [nn.Linear(nfc,2)]

    self.conv_model = nn.Sequential(*conv_sequence)
    self.fc_model = nn.Sequential(*fc_sequence)

  def forward(self, input):
      x = self.conv_model(input)
      x = torch.flatten(x,1)
      x = self.fc_model(x)
      return x