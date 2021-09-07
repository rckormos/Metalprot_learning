#from common import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
import torch



class ConvBn2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0, dilation=1):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model
# @xhlulu OpenVaccine: Simple GRU Model

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x * torch.sigmoid(x)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        return grad_output * (sigmoid * (1 + x * (1 - sigmoid)))
F_swish = SwishFunction.apply

class Swish(nn.Module):
    def forward(self, x):
        return F_swish(x)






#https://www.nature.com/articles/s41586-019-1923-7.epdf?author_access_token=Z_KaZKDqtKzbE7Wd5HtwI9RgN0jAjWel9jnR3ZoTv0MCcgAwHMgRx9mvLjNQdB2TlQQaa7l420UCtGo8vYQ39gg8lFWR9mAZtvsN_1PrccXfIbc6e-tGSgazNL_XdtQzn1PHfy21qdcxV7Pw-k3htw%3D%3D
class Residual(nn.Module):
    def __init__(self,in_channel, dilation):
        super(Residual, self).__init__()
        self.conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm2d(in_channel)
        #self.act  = nn.ELU(inplace=True)
        self.act  = Swish()

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = x+residual
        return x




class AlphafoldNet(nn.Module):
    def __init__(self):
        super(AlphafoldNet, self).__init__()
        num_target = 1

        self.block_n1 = nn.Sequential(
            ConvBn2d(40, 8, kernel_size=3, padding=1),  # 29 + 6(sol) --> 35
            Swish(), #nn.ELU(inplace=True),#Swish(), #
          # nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
        )
        self.block0 = nn.Sequential(
            ConvBn2d(12, 64, kernel_size=3, padding=1),  # 29 + 6(sol) --> 35
            Swish(), #nn.ELU(inplace=True),#Swish(), #
          # nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),
        )
        self.block1 = nn.Sequential(
            Residual(64, dilation=1),
            Residual(64, dilation=1),
            ConvBn2d(64, 128, kernel_size=1, padding=0),
            Swish(), #nn.ELU(inplace=True),#Swish(), #
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),#
        )
        self.block2 = nn.Sequential(
            Residual(128, dilation=1),
            Residual(128, dilation=1),
            ConvBn2d(128, 256, kernel_size=1, padding=0),
            Swish(), #nn.ELU(inplace=True),#Swish(), #
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Dropout(0.2),#
        )
        self.block3 = nn.Sequential(
            Residual(256, dilation=1),
            Residual(256, dilation=1),
            #ConvBn2d(128, 256, kernel_size=1, padding=0),
          # nn.MaxPool2d(kernel_size=(3,3)),
            # nn.ELU(inplace=True),
            nn.Dropout(0.2),#
        )
        # self.block3 = nn.Sequential(
        #     Residual(256, dilation=1),
        #     Residual(256, dilation=2),
        #     Residual(256, dilation=4),
        #     Residual(256, dilation=8),
        #     nn.MaxPool2d(kernel_size=(1,2)),
        #     nn.ELU(inplace=True),
        # )
        self.linear1 = nn.Linear(256*3*3,512)
        self.linear2 = nn.Linear(512,32)
        self.linear3 = nn.Linear(32,1)

    #https://discuss.pytorch.org/t/clarification-regarding-the-return-of-nn-gru/47363/2
    def forward(self, x):
        batch_size, dim, length, length = x.shape

        x1 = x[:, :4, :, :]                       # ; print("x1:", x1.size())             #torch.Size([8, 5, 257, 257]) 
        x2 = x[:, 4:, :, :]                       # ; print("x2:", x2.size())             #torch.Size([8, 1464, 257, 257]) # --> 156

        x2 = self.block_n1(x2)                      # ; print("xl0:", x2.size())            #torch.Size([8, 11, 257, 257])
        x = torch.cat([x1,x2],1)                  # ; print('x', x.size())                #torch.Size([8, 16, 257, 257])

        x = self.block0(x)     # ; print("xb0 size:", x.size())
        x = self.block1(x)     # ; print("xb1 size:", x.size())
        x = self.block2(x)     # ; print("xb2 size:", x.size())
        x = self.block3(x)     # ; print("xb3 size:", x.size())


        # pool
      # avg = x.mean(3) #x.mean(2) +
      # #max = x.max(-1)[0]
        #x = (avg+max)/2
      # x = avg

        x = F.dropout(x,0.5, training=self.training)
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x


class basicCnn_v0(nn.Module):
    def __init__(self):
        super(basicCnn,self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(104,13,kernel_size=1, padding=0,stride=1),
            nn.BatchNorm2d(13),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(26,16,kernel_size=3, padding=0,stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=0, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3*3*64,32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()


    def forward(self,x):
        batch_size, dim, length, length = x.shape     # ; print("map:", x.size())
      # print("batch_size:", batch_size, "dim:", dim, "length:", length)
      # batch_size: 8 dim: 1469 length: 257

        if 1:
            x1 = x[:, :13, :, :]                       # ; print("x1:", x1.size())             #torch.Size([8, 5, 257, 257]) 
            x2 = x[:, 13:, :, :]                       # ; print("x2:", x2.size())             #torch.Size([8, 1464, 257, 257]) # --> 156

            x2 = self.layer0(x2)                      # ; print("xl0:", x2.size())            #torch.Size([8, 11, 257, 257])
            x = torch.cat([x1,x2],1)                  # ; print('x', x.size())                #torch.Size([8, 16, 257, 257])

        out = self.layer1(x)                          # ; print("xl1:", out.size())           #torch.Size([4, 16, 64, 64])
        out = self.layer2(out)                        # ; print("xl2:", out.size())           #torch.Size([4, 32, 15, 15])
        out = self.layer3(out)                        # ; print("xl3:", out.size())           #torch.Size([4, 64, 3, 3])
        out = out.reshape(out.size(0),-1)             # ; print("x view:", out.size())        #torch.Size([4, 576])
        out = self.relu(self.fc1(out))                # ; print("xf1:", out.size())           #torch.Size([4, 10])
        out = self.fc2(out)                           # ; print("xf2:", out.size())           #torch.Size([4, 2])
        out = torch.sigmoid(out)
        return out


class basicCnn(nn.Module):
    def __init__(self):
        super(basicCnn,self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(40,8, kernel_size=1, padding=0,stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.layer1 = nn.Sequential(
            nn.Conv2d(12,16, kernel_size=3, padding=3,stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32,64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

      # self.fc1 = nn.Linear(3*3*64,32)
        self.fc1 = nn.Linear(2*2*64,8)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(8,1)
        self.relu = nn.ReLU()


    def forward(self,x):
        batch_size, dim, length, length = x.shape     # ; print("map:", x.size())
      # print("batch_size:", batch_size, "dim:", dim, "length:", length)
      # batch_size: 8 dim: 1469 length: 257

        if 1:
            x1 = x[:, :4, :, :]                       # ; print("x1:", x1.size())             #torch.Size([8, 5, 257, 257]) 
            x2 = x[:, 4:, :, :]                       # ; print("x2:", x2.size())             #torch.Size([8, 1464, 257, 257]) # --> 156

            x2 = self.layer0(x2)                      # ; print("xl0:", x2.size())            #torch.Size([8, 11, 257, 257])
            x = torch.cat([x1,x2],1)                  # ; print('x', x.size())                #torch.Size([8, 16, 257, 257])

        out = self.layer1(x)                          # ; print("xl1:", out.size())           #torch.Size([4, 16, 64, 64])
        out = self.layer2(out)                        # ; print("xl2:", out.size())           #torch.Size([4, 32, 32, 32])
        out = self.layer3(out)                        # ; print("xl3:", out.size())           #torch.Size([4, 64, 16, 16])
      # out = self.layer4(out)                          ; print("xl3:", out.size())           #torch.Size([4, 128, 8, 8])
      # out = self.layer5(out)                          ; print("xl3:", out.size())           #torch.Size([4, 128, 4, 4])
        out = out.reshape(out.size(0),-1)             # ; print("x view:", out.size())        #torch.Size([4, 2048])
        out = self.relu(self.fc1(out))                # ; print("xf1:", out.size())           #torch.Size([4, 32])
        out = self.fc2(out)                           # ; print("xf2:", out.size())           #torch.Size([4, 2])
        out = torch.sigmoid(out)
        return out

