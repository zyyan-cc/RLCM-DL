from torch import nn as nn
import torch


class ResDCNN(nn.Module):
    def __init__(self, model_arch=[64, 64, 128, 128, 256]):
        super(ResDCNN, self).__init__()

        downsample1 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, stride=2, bias=False, padding=1),
                nn.BatchNorm1d(128))
        
        downsample2 = nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(256))
        
        self.fc = nn.Linear(model_arch[-1], 1)

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1,
                    padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU())
        
        self.conv2 = Bottleneck(64, 64)
        self.conv3 = Bottleneck(64, 128, stride=2, downsample=downsample1)
        self.conv4 = Bottleneck(128, 128)
        self.conv5 = Bottleneck(128, 256, stride=1, downsample=downsample2)

        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        out = self.avg(x)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
    
class Bottleneck(nn.Module):
    
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
 
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        
        self.downsample = downsample

    def forward(self, x):

        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = out + identity
 
        return out

if __name__ == "__main__":
    input_tensor = torch.rand(size=(1, 1, 8))
    net = ResDCNN()
    out = net(input_tensor)

    print(out.shape)