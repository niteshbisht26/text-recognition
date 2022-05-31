import torch.nn as nn
import torchvision.models as models

class YOLONetwork(nn.Module):
  def __init__(self):
    super(YOLONetwork, self).__init__()
    self.resnet = nn.Sequential(*list(models.resnet101().children())[:-3])
    self.regresser = nn.Sequential(
        nn.Conv2d(1024, 512, 5),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 128, 5),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 20, 1)
    )
  
  def forward(self, input_):
    out = self.regresser(self.resnet(input_)).permute(0, 2, 3, 1).contiguous()
    return out

class CRNNetwork(nn.Module):
  def __init__(self, alphanum2ind):
    super(CRNNetwork, self).__init__()
    self.alphanum2ind = alphanum2ind
    
    self.convNet = nn.Sequential(
        nn.Conv2d(1, 64, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, 3, padding='same'),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d((2, 1)),
        nn.Conv2d(256, 512, 3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.Conv2d(512, 512, 3, padding='same'),
        nn.ReLU(),
        nn.BatchNorm2d(512),
        nn.MaxPool2d((2, 1)),
        nn.Conv2d(512, 512, 2),
        nn.ReLU()
    )

    self.lstm1 = nn.LSTM(512, 128, batch_first=True, dropout=0.2, bidirectional=True)
    self.lstm2 = nn.LSTM(256, 128, batch_first=True, dropout=0.2, bidirectional=True)

    self.linear = nn.Linear(256, len(self.alphanum2ind))
  
  def forward(self, input_):
    out = self.convNet(input_).squeeze(2).permute(0, 2, 1)
    out, h_c_n = self.lstm1(out)
    out, h_c_n = self.lstm2(out)
    out = self.linear(out)
    return out