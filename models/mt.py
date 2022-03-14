import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

from .encoder import encoder
from .decoder.ccnet import CCNet, RCCAModule
from .decoder.bts import bts


class MTmodel(nn.Module):
    def __init__(self, params):
        '''
        params : dict
        '''
        super(MTmodel, self).__init__()

        self.encoder = encoder(params)

        # Semantic
        self.recurrence = 2 # For 2 loop in RRCAModule
        self.semantic_decoder = CCNet(inplanes=self.encoder.feat_out_channels[-1], num_classes=params.num_classes, recurrence=self.recurrence)
        # self.semantic_decoder = RCCAModule(self.encoder.feat_out_channels[-1], 512, self.num_classes)

        # Depth
        bts_size = 512
        self.depth_decoder = bts(params, self.encoder.feat_out_channels, bts_size)

    def forward(self, x):
        feature_maps = self.encoder(x) # five feature maps
        
        res = []
        res.append(self.semantic_decoder(feature_maps[-1], self.recurrence)) # use the last

        depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth = self.depth_decoder(feature_maps)
        res.append(final_depth)
        
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTmodel(params).to(device)
    summary(model, (3, 768, 384))
    