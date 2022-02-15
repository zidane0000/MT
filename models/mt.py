import argparse
import torch
import torch.nn as nn
import torchvision.models as models

from torchsummary import summary


class MTmodel(nn.Module):
    def __init__(self, params):
        '''
        params : dict
        '''
        super(MTmodel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=7, kernel_size=3),
            nn.ReLU(),            
        )

        self.decoders = []
        # semantic
        self.semantic_decoder = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=3, kernel_size=3),
            nn.ReLU(),          
        )
        self.decoders.append(self.semantic_decoder)

        # depth
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=3, kernel_size=3),
            nn.ReLU(),            
        )
        self.decoders.append(self.depth_decoder)

    def forward(self, x):
        feature_map = self.encoder(x)
        
        res = []

        res.append(self.semantic_decoder(feature_map))
        res.append(self.depth_decoder(feature_map))
        
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MTmodel(params).to(device)

    summary(model, (3, 640, 420))