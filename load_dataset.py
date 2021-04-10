# NN
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

# torch
import torch
import torchvision
from torchvision.transforms import transforms

import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------
#                                   load data
# --------------------------------------------------------------------------------
transform = transforms.Compose([transforms.ToTensor(), ])

trainset = torchvision.datasets.CIFAR10(root='CIFAR10', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='CIFAR10', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)  # , num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# for i in range(10):
#     dataiter = iter(testloader)
#     images, labels = dataiter.next()
#     imshowss = torchvision.utils.make_grid(images)
#     plt.imshow(imshowss.transpose(0, -1))
#     plt.pause(10)
#     plt.close()

# --------------------------------------------------------------------------------
#                                   AutoEncoder class
# --------------------------------------------------------------------------------
class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()

        # decoder
        self.de1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.de2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.fc1_2 = nn.Linear(in_features=50176, out_features=256)
        self.fc2_3 = nn.Linear(in_features=256, out_features=32)

        # encoder
        self.fc3_2 = nn.Linear(in_features=32, out_features=256)
        self.fc2_1 = nn.Linear(in_features=256, out_features=50176)
        self.en2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.en1 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1))

        # softmax layer
        self.softmax = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # forward decoder
        xe = F.relu(self.de1(x))
        xe = F.relu(self.de2(xe))

        shp = [xe.shape[0], xe.shape[1], xe.shape[2], xe.shape[3]]
        xe.view(-1, shp[1] * shp[1] * shp[1])

        xe = F.relu(self.fc1_2(xe))
        xe = F.relu(self.fc2_3(xe))

        # forward encoder
        xd = F.relu(self.fc3_2(xe))
        xd = F.relu(self.fc2_1(xd))
        xd = torch.reshape(xd, (shp[0], shp[1], shp[2], shp[3]))
        xd = F.relu(self.conv3(xd))
        x_hat = F.relu(self.conv4(xd))

        y_hat = self.softmax(xe)
        return x_hat, y_hat

