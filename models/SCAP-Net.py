import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from mamba_ssm import Mamba
from mmcv.cnn import build_norm_layer
from timm.models.layers import DropPath
import math
from typing import Optional, Union, Sequence
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule, constant_init
from mmengine.model.weight_init import trunc_normal_init, normal_init
from mmengine.logging import MMLogger

import torch.nn as nn
from einops.layers.torch import Rearrange

from tools import RCA, CAB, PVM, PVMLayer, MCPM











class SCMCNet(nn.Module):
    
    def __init__(self, num_classes=1, input_channels=3, c_list=[8,16,24,32,48,64],
                split_att='fc', bridge=True):
        super().__init__()

        self.bridge = bridge
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
        self.encoder2 =nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        ) 
        self.encoder3 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )
        self.encoder4 = nn.Sequential(
            MCPM(c_list[2]),
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            MCPM(c_list[3]),
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            MCPM(c_list[4]),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )

        if bridge: 
            self.cab = CAB(c_list, split_att)

        self.rca_1=RCA(dim=c_list[0])
        self.rca_2=RCA(dim=c_list[1])
        self.rca_3=RCA(dim=c_list[2])
        self.rca_4=RCA(dim=c_list[3])
        self.rca_5=RCA(dim=c_list[4])

        
        self.decoder1 = nn.Sequential(
            MCPM(c_list[5]),
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        ) 
        self.decoder2 = nn.Sequential(
            MCPM(c_list[4]),
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        ) 
        self.decoder3 = nn.Sequential(
            MCPM(c_list[3]),
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )  
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )  
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )  
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])
        self.dbn1 = nn.GroupNorm(4, c_list[4])
        self.dbn2 = nn.GroupNorm(4, c_list[3])
        self.dbn3 = nn.GroupNorm(4, c_list[2])
        self.dbn4 = nn.GroupNorm(4, c_list[1])
        self.dbn5 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        t1 = out 

        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out

        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out
        
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out
        
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        t5 = out


        t1 = self.rca_1(t1)
        t2 = self.rca_2(t2)
        t3 = self.rca_3(t3)
        t4 = self.rca_4(t4)
        t5 = self.rca_5(t5)
        if self.bridge: t1, t2, t3, t4, t5 = self.cab(t1, t2, t3, t4, t5)
        
        out = F.gelu(self.encoder6(out)) 
        
        out5 = F.gelu(self.dbn1(self.decoder1(out))) 
        out5 = torch.add(out5, t5) 
        
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out4 = torch.add(out4, t4)
        
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) 
        out3 = torch.add(out3, t3) 
        
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)),scale_factor=(2,2),mode ='bilinear',align_corners=True)) 
        out2 = torch.add(out2, t2) 
        
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)),scale_factor=(2,2),mode ='bilinear',align_corners=True))
        out1 = torch.add(out1, t1) 
        
        out0 = F.interpolate(self.final(out1),scale_factor=(2,2),mode ='bilinear',align_corners=True) 
        
        return torch.sigmoid(out0)
