import argparse
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

from .decoder.ccnet import RCCAModule
from .decoder.bts import encoder, bts

class MTmodel(nn.Module):
    def __init__(self, params):
        '''
        params : dict
        '''
        super(MTmodel, self).__init__()

        params.encoder = 'densenet161_bts'
        params.max_depth = 80
        self.encoder = encoder(params)

        self.recurrence = 2 # For 2 loop in RRCAModule
        self.num_classes = 19
        self.semantic_decoder = RCCAModule(self.encoder.feat_out_channels[-1], 512, self.num_classes)

        # depth
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
    summary(model, (3, 640, 420))