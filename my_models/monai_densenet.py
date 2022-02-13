import torch
import monai
import torch.nn as nn

class monai_densenet_three_views(nn.Module):
    def __init__(self, l1=150528//(128), l2=150528//(256)): # was 128, 512, best 128, 256
        super(monai_densenet_three_views, self).__init__()
        self.model = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        
        self.model2 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        
        self.model3 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features

        self.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(in_features=150528, out_features=l1, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=l1, out_features=l2, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=l2, out_features=2, bias=True),
                        )
        
    def forward(self, x1, x2, x3):
        x1 = (self.model(x1))
        x2 = (self.model2(x2))
        x3 = (self.model3(x3))

        combined = torch.cat((x1, x2, x3),dim=1)
        combined = combined.reshape(combined.size(0), -1)
        logits = self.fc(combined) 
 
        return logits

class monai_densenet_single_view(nn.Module):
    def __init__(self, l1=50176//(128), l2=50176//(256)):
        super(monai_densenet_single_view, self).__init__()
        self.model =  monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features

        
        self.avrg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(in_features=50176, out_features=l1, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=l1, out_features=l2, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=l2, out_features=2, bias=True),
                        )
        
    def forward(self, x1):
        x1 = (self.model(x1))

        x1 = x1.reshape(x1.size(0), -1)
        logits = self.fc(x1)     
        return logits
    
class multi_mri_monai_densenet_block(nn.Module):
    def __init__(self):
        super(multi_mri_monai_densenet_block, self).__init__()

        self.model = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        
        self.model2 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        
        self.model3 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features

    def forward(self, x1, x2, x3):
        x1 = (self.model(x1))
        x2 = (self.model2(x2))
        x3 = (self.model3(x3))
        features = torch.cat((x1, x2, x3),dim=1)
        return features

class multi_mri_fc_layers(nn.Module):
    def __init__(self):
        super(multi_mri_fc_layers, self).__init__()

        self.mri_1 = multi_mri_monai_densenet_block()
        self.mri_2 = multi_mri_monai_densenet_block()

        self.fc = nn.Sequential(
                            # nn.Dropout(0.3),
                            # nn.Linear(in_features=150528*2, out_features=(150528*2)//128, bias=True),
                            # nn.Dropout(0.3),
                            # nn.Linear(in_features=(150528*2)//128, out_features=(150528*2)//256, bias=True),
                            # nn.Dropout(0.3),
                            # nn.Linear(in_features=(150528*2)//256, out_features=2, bias=True),
                            nn.Linear(in_features=(150528*2), out_features=2, bias=True)
                        )
    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = (self.mri_1(x1, x2, x3))
        x2 = (self.mri_2(x4, x5, x6))

        x= torch.cat((x1, x2),dim=1)
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x) 
        return logits


class multi_mri_monai_densenet_block_single_axis(nn.Module):
    def __init__(self):
        super(multi_mri_monai_densenet_block_single_axis, self).__init__()

        self.densenet1 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features
        self.densenet2 = monai.networks.nets.DenseNet121(pretrained= True, spatial_dims=2, in_channels=1,out_channels=2).features

        self.fc = nn.Sequential(
                            nn.Dropout(0.3),
                            nn.Linear(in_features=(50176*2), out_features=(50176*2)//32, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=(50176*2)//32, out_features=(50176*2)//128, bias=True),
                            nn.Dropout(0.3),
                            nn.Linear(in_features=(50176*2)//128, out_features=2, bias=True),
                        )

    def forward(self, x1, x2):
        x1 = (self.densenet1(x1))
        x2 = (self.densenet2(x2))
        x = torch.cat((x1, x2),dim=1)
        x = x.reshape(x.size(0), -1)
        logits = self.fc(x) 

        return logits




