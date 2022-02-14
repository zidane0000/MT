import torch
import torch.nn as nn

from decoder import bts

class MTmodel(nn.Module):
    def __init__(self, params):
        '''
        params : dict
        '''
        super(MTmodel, self).__init__()
        self.encoder = encoder(params)

        self.decoders = []
        # semantic
        semantic_decoder = semantic_decoder()
        self.decoders.append(semantic_decoder)

        # depth
        bts_size = 512 # initial num_filters in bts
        depth_decoder = bts(params, self.encoder.feat_out_channels, bts_size)
        self.decoders.append(depth_decoder)

    def forward(self, x):
        feature_map = self.encoder(x)
        
        res = []
        for decoder in self.decoders:
            res.append(decoder(feature_map))
        
        return res
