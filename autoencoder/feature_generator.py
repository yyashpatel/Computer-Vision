import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim 
import cv2
import PIL
import torch
import sklearn
from sklearn.manifold import TSNE
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision.models as models
from torchvision.transforms import ToTensor, Lambda
import torchvision.transforms as transforms
import mnist_reader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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

def refine_data(out):
    a = out
    a = a.reshape(-1,784)
    a = torch.from_numpy(a)
    a = a.to(torch.float).to(device)
    a = network_model(a,'abc')
    a = out
    a = a.reshape(-1,784)
    a = torch.from_numpy(a)
    a = a.to(torch.float).to(device)
    a = network_model(a,'train')
    a = a.to('cpu')
    a = a.cpu().detach().numpy()
    a = a.reshape(28,28)
    return a

device = "cuda"

network_model = deep_net().to(device)
path = "D:/task3/model.pth"
network_model.load_state_dict(torch.load(path))
network_model.eval()     

X_train, y_train = mnist_reader.load_mnist('', kind='train', )
X_test, y_test = mnist_reader.load_mnist('', kind='t10k')

X_train = X_train.reshape(60000,28,28)
X_test = X_test.reshape(10000,28,28)

a = X_train[7]
b = X_train[10]

a_op = refine_data(a)
b_op = refine_data(b)

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(a)
axarr[0,1].imshow(a_op)
axarr[1,0].imshow(b)
axarr[1,1].imshow(b_op)

transforms = transforms.Compose([torchvision.transforms.Grayscale(1),transforms.ToTensor()])

test_dataset = torchvision.datasets.FashionMNIST(root="~/torch_datasets", train=False, transform = transforms, download=False)

test_loader = DataLoader(test_dataset, batch_size = 40)

dense_feature_array = torch.empty((40,32), dtype=torch.float32).to(device)

for batch,i in test_loader:
    
    batch = batch.view(-1, 784).to(device)
    outputs = network_model(batch, 'f')
    dense_feature_array = torch.cat((dense_feature_array, outputs),0)
    
dense_feature_array = dense_feature_array[40:10040,:]
dense_feature_array = dense_feature_array.cpu()

dense_feature_array = dense_feature_array.detach().numpy()

print(type(dense_feature_array))

dense_feature_array = dense_feature_array.astype(np.double)
reduced_data = PCA(n_components=2).fit_transform(dense_feature_array)
kmeans = KMeans(init="k-means++", n_clusters=10, n_init=20)
kmeans.fit(reduced_data)

h = 0.02

x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape) 
plt.figure(1)
plt.clf()
plt.imshow(Z,interpolation="nearest",extent=(xx.min(), xx.max(), yy.min(), yy.max()),cmap=plt.cm.Paired,aspect="auto",origin="lower",)
plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0],centers[:, 1],marker="x",s=169,linewidths=3,color="w",zorder=10,)
plt.title("plot")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()