import argparse
import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary

try:
    from .encoder import encoder
    from .decoder.ccnet import CCNet, RCCAModule
    from .decoder.hrnet_ocr import HighResolutionDecoder, cfg
    from .decoder.bts import bts
except:
    from encoder import encoder
    from decoder.ccnet import CCNet, RCCAModule
    from decoder.hrnet_ocr import HighResolutionDecoder, cfg
    from decoder.bts import bts

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
        # self.semantic_decoder = RCCAModule(self.encoder.feat_out_channels[-1], 512, params.num_classes)
        # self.semantic_decoder = HighResolutionDecoder(cfg, self.encoder.feat_out_channels[-4:])

        # Depth
        bts_size = 512
        self.depth_decoder = bts(params, self.encoder.feat_out_channels, bts_size)

    def forward(self, x):
        feature_maps = self.encoder(x) # five feature maps
        
        res = []
        res.append(self.semantic_decoder(feature_maps[-1], self.recurrence)) # use the last
        # res.append(self.semantic_decoder(feature_maps[-4:]))

        depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth = self.depth_decoder(feature_maps)
        res.append(final_depth)
        
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',         type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--encoder',            type=str, help='Choose Encoder in MT', default='densenet161')
    parser.add_argument('--summary', action='store_true', help='Summary Model')
    # Semantic Segmentation
    parser.add_argument('--num_classes',        type=int, help='Number of classes to predict (including background).', default=19)

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    params = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MTmodel(params).to(device)
    print("load model to device")
    
    if params.summary:
        summary(model, (3, 32, 32))
        
    if True:
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )
        print('for HRNet')
    
    ran = torch.rand((4, 3, 64, 64)).to(device)
    model.forward(ran)
    print("pass")
    