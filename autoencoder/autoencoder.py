import numpy as np
import torch.optim as optim 
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

transforms = transforms.Compose([torchvision.transforms.Grayscale(1),transforms.ToTensor()])

train_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets", train=True, transform = transforms, download=True)

test_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets", train=False, transform = transforms, download=False)
train_loader = DataLoader(train_dataset, batch_size=32)

#encoder and decoder
class deep_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=784, out_features=334)
        self.layer2 = nn.Linear(in_features=334, out_features=128)
        self.layer3 = nn.Linear(in_features=128, out_features=32)

        self.d_layer1 = nn.Linear(in_features=32, out_features=128)
        self.dout_layer = nn.Linear(in_features=128, out_features=784)

    def calculate(self, features):
        
      l1 = self.layer1(features)
      l1 =  torch.relu(l1)

      l2 = self.layer2(l1)
      l2 = torch.relu(l2)

      l3 = self.layer3(l2)
      l3 = torch.relu(l3)
      return l3
    
    def expand(self, l3):
        out1 = self.d_layer1(l3)
        out1 = torch.relu(out1)
        out1 = self.dout_layer(out1)
        expanded = torch.relu(out1)
        
        return expanded
    
    def forward(self, x, select):
        if select=='train':
            x = self.calculate(x)
            x = self.expand(x)

        else:
            x = self.calculate(x)
        return x

device = "cuda"
model = deep_net().to(device)

trainer = optim.Adam(model.parameters(), lr = 1e-3)
lossfnc = nn.MSELoss()

#train the model
for j in range(35):
  loss = 0
           
  for data, i in train_loader:

    data = data.view(-1, 784).to(device)
    trainer.zero_grad()
    result = model(data, "train")
    train_loss = lossfnc(result, data)
    train_loss.backward()
    trainer.step()
    loss += train_loss.item()
   
  loss = loss / len(train_loader)
torch.save(model.state_dict(),"D:/task3/model.pth")