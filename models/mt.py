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
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1),
            nn.ReLU(),
        )

        self.decoders = []
        # semantic
        self.num_classes = 19
        self.semantic_decoder = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=self.num_classes, kernel_size=1),
            nn.ReLU(),          
        )
        self.decoders.append(self.semantic_decoder)

        # depth
        self.depth_decoder = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Sigmoid(),
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