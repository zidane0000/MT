import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Neck(nn.Module):
    def __init__(self, ch_in, ch_out):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        assert len(ch_in) == len(ch_out) == 4, 'Neck need same num channel'

        layers = []
        layers.append(Conv(ch_in[3], 512, 1, 1)) # 4
        layers.append(SPP(512, 512, (5, 9, 13)))
        layers.append(Conv(512, 384, 1, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        layers.append(Conv(ch_in[2], 384, 1, 1)) # 8
        layers.append(Concat())
        layers.append(BottleneckCSP(768, 384))
        layers.append(Conv(384, 256, 1, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        layers.append(Conv(ch_in[1], 256, 1, 1)) # 13
        layers.append(Concat())
        layers.append(BottleneckCSP(512, 256))
        layers.append(Conv(256, 128, 1, 1))
        layers.append(nn.Upsample(scale_factor=2, mode='nearest'))

        layers.append(Conv(ch_in[0], 128, 1, 1)) # 18
        layers.append(Concat())

        # PAN
        layers.append(BottleneckCSP(256, 128))
        layers.append(Conv(128, 256, 3, 2))
        layers.append(Concat())

        layers.append(BottleneckCSP(512, 256))   # 23
        layers.append(Conv(256, 384, 3, 2))
        layers.append(Concat())

        layers.append(BottleneckCSP(768, 384))   # 26
        layers.append(Conv(384, 512, 3, 2))
        layers.append(Concat())

        layers.append(BottleneckCSP(1024, 512))  # 29

        layers.append(Conv(128,ch_out[0],3,1))   # 30
        layers.append(Conv(256,ch_out[1],3,1))   # 31
        layers.append(Conv(384,ch_out[2],3,1))   # 32
        layers.append(Conv(512,ch_out[3],3,1))   # 33

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        layers = self.layers
        x.append(layers[0](x[-1]))
        x.append(layers[1](x[-1]))
        x.append(layers[2](x[-1]))
        x.append(layers[3](x[-1]))

        x.append(layers[4](x[2]))
        x.append(layers[5]([x[-1], x[-2]]))
        x.append(layers[6](x[-1]))
        x.append(layers[7](x[-1]))
        x.append(layers[8](x[-1]))

        x.append(layers[9](x[1]))
        x.append(layers[10]([x[-1], x[-2]]))
        x.append(layers[11](x[-1]))
        x.append(layers[12](x[-1]))
        x.append(layers[13](x[-1]))

        x.append(layers[14](x[0]))
        x.append(layers[15]([x[-1], x[-2]]))
        x.append(layers[16](x[-1]))
        x.append(layers[17](x[-1]))

        x.append(layers[18]([x[-1], x[15]]))
        x.append(layers[19](x[-1]))
        x.append(layers[20](x[-1]))
        
        x.append(layers[21]([x[-1], x[10]]))
        x.append(layers[22](x[-1]))
        x.append(layers[23](x[-1]))
        
        x.append(layers[24]([x[-1], x[5]]))
        x.append(layers[25](x[-1]))

        x.append(layers[26](x[20]))
        x.append(layers[27](x[23]))
        x.append(layers[28](x[26]))
        x.append(layers[29](x[29]))

        x = x[-4:]
        return x