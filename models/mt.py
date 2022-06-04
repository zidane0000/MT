import argparse
import math
import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary

try:
    from .encoder import encoder
    from .yolo import YOLOR_backbone, Detect, IDetect
    from .decoder.ccnet import CCNet, RCCAModule
    from .decoder.hrnet_ocr import HighResolutionDecoder, cfg
    from .decoder.bts import bts, bts_4channel
    from .decoder.espnet import ESPNet_Decoder
except:
    from encoder import encoder
    from yolo import YOLOR_backbone, Detect, IDetect
    from decoder.ccnet import CCNet, RCCAModule
    from decoder.hrnet_ocr import HighResolutionDecoder, cfg
    from decoder.bts import bts, bts_4channel
    from decoder.espnet import ESPNet_Decoder


class MTmodel(nn.Module):
    def __init__(self, params):
        '''
        params : dict
        '''
        super(MTmodel, self).__init__()
        
        if params.encoder.lower() == 'yolor':
            self.encoder = YOLOR_backbone(params)
        else:
            self.encoder = encoder(params)

        # Semantic
        self.semantic_head = params.semantic_head.lower()
        if self.semantic_head == "ccnet":
            self.recurrence = 2 # For 2 loop in RRCAModule
            self.semantic_decoder = CCNet(inplanes=self.encoder.feat_out_channels[-1], num_classes=params.num_classes, recurrence=self.recurrence)
            # self.semantic_decoder = RCCAModule(self.encoder.feat_out_channels[-1], 512, params.num_classes)
        elif self.semantic_head == "hrnet":
            self.semantic_decoder = HighResolutionDecoder(cfg, self.encoder.feat_out_channels[-4:])
        elif self.semantic_head == "espnet":
            self.semantic_decoder = ESPNet_Decoder(classes=params.num_classes, input_channels=self.encoder.feat_out_channels[:3])

        # Depth
        self.depth_head = params.depth_head.lower()
        if self.depth_head == "bts":
            bts_size = 512
            if params.encoder.lower() == 'yolor':
                self.depth_decoder = bts_4channel(params, self.encoder.feat_out_channels, bts_size)
            else:
                self.depth_decoder = bts(params, self.encoder.feat_out_channels, bts_size)
            
        # Object detection
        self.obj_head = params.obj_head.lower()
        if self.obj_head == "yolo":
            self.object_detection_decoder = IDetect(ch=self.encoder.feat_out_channels)
            m = self.object_detection_decoder            
            s = 256  # 2x min stride
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, 3, s, s))[-1]])  # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            
            # Check anchor order against stride order for Detect() module m, and correct if necessary
            a = m.anchors.prod(-1).view(-1)  # anchor area
            da = a[-1] - a[0]  # delta a
            ds = m.stride[-1] - m.stride[0]  # delta s
            if da.sign() != ds.sign():  # same order
                print('obj_head Reversing anchor order')
                m.anchors[:] = m.anchors.flip(0)
            
            # initialize_biases -> https://arxiv.org/abs/1708.02002 section 3.3
            for mi, s in zip(m.m, m.stride):  # from
                b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
                b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
                b.data[:, 5:] += math.log(0.6 / (m.nc - 0.999999))  # cls
                mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x):
        feature_maps = self.encoder(x) # five feature maps
        
        res = []
        if self.semantic_head:
            if self.semantic_head == "ccnet":
                res.append(self.semantic_decoder(feature_maps[-1], self.recurrence)) # use the last
            elif self.semantic_head == "hrnet":
                res.append(self.semantic_decoder(feature_maps[-4:]))
            elif self.semantic_head == "espnet":
                res.append(self.semantic_decoder(feature_maps[:3]))
            else:
                raise Exception(f'ERROR: Unkown Semnatic head {self.semantic_head}')

        if self.depth_head:
            depth_8x8_scaled, depth_4x4_scaled, depth_2x2_scaled, reduc1x1, final_depth = self.depth_decoder(feature_maps)
            res.append(final_depth)
        
        if self.obj_head:
            objs = self.object_detection_decoder(feature_maps[:4]) # input -> H/2, H/4, H/8, H/16
            res.append(objs)
        
        return res
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank',         type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--encoder',            type=str, help='Choose Encoder in MT', default='densenet161')
    parser.add_argument('--summary', action='store_true', help='Summary Model')
    # Semantic Segmentation
    parser.add_argument('--num_classes',        type=int, help='Number of classes to predict (including background).', default=19)
    parser.add_argument('--semantic_head',      type=str, help='Choose method for semantic head(CCNet/HRNet/ESPNet)', default='CCNet')

    # Depth Estimation
    parser.add_argument('--min_depth',     type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth',     type=float, help='maximum depth for evaluation', default=80.0)
    parser.add_argument('--depth_head',    type=str, help='Choose method for depth estimation head', default='bts') 
    
    # Object detection
    parser.add_argument('--obj_head',      type=str, help='Choose method for obj detection head', default='yolo')    
    params = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = MTmodel(params).to(device)
    print("load model to device")
    
    if params.summary:
        summary(model, (3, 32, 32))
        
    if params.semantic_head == 'HRNet':
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(backend="nccl", init_method="env://",)
        print('for HRNet')
    
    ran = torch.rand((4, 3, 384, 768)).to(device)    
    output = model.forward(ran)    
    print("pass")
    