import torch.nn as nn
import torchvision

class resnet50_slices(nn.Module):
    def __init__(self):
        super(resnet50_slices, self).__init__()
        self.model =  torchvision.models.resnet50()  
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)      
        self.model.fc = nn.Sequential(
                            nn.Dropout(0.5),
                            nn. Linear(in_features=2048, out_features=3, bias=True),
                        )
        
    def forward(self, x1):
     
        return self.model(x1)
