#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.



from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He

from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He

from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He

class Discriminator(nn.Module):
    def __init__(self, weightInitializer=InitWeights_He(1e-2), in_channels=1, out_channels=2):
        super(Discriminator, self).__init__()

        self.weightInitializer = weightInitializer
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv3d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm3d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.conv = nn.Sequential(
            *discriminator_block(in_channels * 2, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            # nn.ZeroPad3d((1, 0, 1, 0)),
        )
        self.final = nn.Conv3d(512, 1, 4, padding=1, bias=False)
        
        self.linear_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, out_channels)
        )

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        intermediate = self.conv(img_input)
        pad = nn.functional.pad(intermediate, pad=(1,0,1,0,1,0))
        model_output = self.final(pad)
        model_output_flatten = model_output.flatten(start_dim=1)
        linear_output = self.linear_layers(model_output_flatten)
        return linear_output
    