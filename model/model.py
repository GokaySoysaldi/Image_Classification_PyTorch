import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.cnn_layers = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
        nn.Dropout(0.15),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=(2, 2)),
        
            
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), padding=1),
        nn.Dropout(0.15),
        nn.ReLU(),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=(2, 2)),
            
            
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), padding=1),
        nn.Dropout(0.15),
        nn.ReLU(),
        nn.BatchNorm2d(256),
        nn.MaxPool2d(kernel_size=(2, 2)),
        ).to(self.device)
        
        self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.2),
        nn.Linear(4096, 256),
        nn.ReLU(),
            
        nn.BatchNorm1d(256),    
            
        nn.Linear(256, 10),
        #nn.Softmax(),
        ).to(self.device)
        
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.classifier(x)
        return x