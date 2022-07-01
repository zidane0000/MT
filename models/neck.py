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
    def __init__(self, ch_in, ch_out, pan=True):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        assert len(ch_in) == len(ch_out), 'Neck need same num channel'
        
        self.pan = pan # PAN architecture

        ch_len = len(ch_in)
        self.ch_len = ch_len
        ch_neck = [128 + 128 * i for i in range(ch_len)] # Equivariance

        layers = []
        # FPN
        layers.append(Conv(ch_in[-1], ch_neck[ch_len - 1], 1, 1))
        layers.append(SPP(ch_neck[-1], ch_neck[ch_len - 1], (5, 9, 13)))

        for i in range(ch_len - 1):
            layers.append(Conv(ch_neck[ch_len - 1 - i], ch_neck[ch_len - 2 - i], 1, 1))
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(Conv(ch_in[ch_len - 2 - i], ch_neck[ch_len - 2 - i], 1, 1))
            layers.append(Concat())
            if not (self.pan and i == (ch_len - 2)):
                layers.append(BottleneckCSP(ch_neck[ch_len - 2 - i] * 2, ch_neck[ch_len - 2 - i]))
        
        # PAN
        if self.pan:
            for i in range(ch_len - 1):
                layers.append(BottleneckCSP(ch_neck[i] * 2, ch_neck[i]))
                layers.append(Conv(ch_neck[i], ch_neck[i+1], 3, 2))
                layers.append(Concat())
            layers.append(BottleneckCSP(ch_neck[ch_len - 1] * 2, ch_neck[ch_len - 1]))

        for i in range(ch_len):
            layers.append(Conv(ch_neck[i],ch_out[i],3,1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        layers = self.layers
        ch_len = self.ch_len

        posi = [] # record SPP and BottleneckCSP posi for concat

        x.append(layers[0](x[-1]))
        x.append(layers[1](x[-1]))
        posi.append(len(x) - 1)

        for i in range(ch_len - 1):
            x.append(layers[2 + 5 * i](x[-1]))
            x.append(layers[3 + 5 * i](x[-1]))
            x.append(layers[4 + 5 * i](x[ch_len - 2 - i]))
            x.append(layers[5 + 5 * i]([x[-1], x[-2]]))
            if not (self.pan and i == (ch_len - 2)):
                x.append(layers[6 + 5 * i](x[-1]))                
                posi.append(len(x) - 1)
        fpn_last = 5 + 5 * i

        if self.pan:
            posi_fpn = posi.copy()
            for i in range(ch_len - 1):
                x.append(layers[fpn_last + 1 + 3 * i](x[-1]))              
                posi.append(len(x) - 1)
                x.append(layers[fpn_last + 2 + 3 * i](x[-1]))
                posi_fpn[ch_len - 2 - i]
                x.append(layers[fpn_last + 3 + 3 * i]([x[-1], x[posi_fpn[ch_len - 2 - i]]]))
            x.append(layers[fpn_last + 4 + 3 * i](x[-1]))
            posi.append(len(x) - 1)

        for i in range(ch_len):
            print(posi[len(posi_fpn) + i])
            print(len(x))
            x.append(layers[i - ch_len](x[posi[len(posi_fpn) + i]]))

        x = x[-4:]
        return x