import torch
import torch.nn as nn
import torch.optim as optim

# Layer ( kernel_size , filters , strides , padding)
#             3            64       1          0 
model_arch = [
				[3 , 64 , 1 , 1 , 2] , 
				"M" ,
				[3 , 128, 1 , 1 , 2] ,
				"M" ,
				[3 , 256, 1 , 1 , 4] ,
				"M" ,
				[3 , 512, 1 , 1 , 4] ,
				"M" , 
				[3 , 512, 1 , 1 , 4] ,
				"M"
			]

class VGG(nn.Module):
  def __init__(self , arch , in_channels , int_shape , num_classes):
    super(VGG , self).__init__()
    self.in_channels  = in_channels
    self.int_shape   = int_shape
    self.num_classes = num_classes
    self.vgg = self.body(arch)
    self.fcs = self.tail()
  
  def forward(self , x):
    x = torch.flatten(self.vgg(x) , start_dim = 1)
    return self.fcs(x)
  
  def body(self , archs):
    current_channels = self.in_channels
    layers = []
    poolings = 0
    for block in archs:
      if type(block) == list:
        layers+= [nn.Conv2d(current_channels , block[1] , kernel_size = block[0] , stride = block[2] , padding = block[3]),
						  nn.BatchNorm2d(block[1]),
						  nn.ReLU(0.1)]
        current_channels = block[1]
      else:
        layers+= [nn.MaxPool2d(kernel_size = (2 , 2) , stride = (2 , 2))]
        poolings += 1
    self.out = int(self.int_shape / 2**poolings)
    return nn.Sequential(*layers)
  
  def tail(self):
    return nn.Sequential(
				nn.Linear(512 * self.out**2 , 4096) ,
				nn.ReLU(0.1) ,
				nn.Dropout(0.5)  ,
				nn.Linear(4096 , 4096) ,
				nn.ReLU(0.1) ,
				nn.Dropout(0.5) ,
				nn.Linear(4096 , self.num_classes))