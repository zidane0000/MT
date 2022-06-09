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
        assert len(ch_in) == len(ch_out), 'Neck need same num channel'

        self.num_inputs = len(ch_in) # number of inputs

        layers = []
        for i in range(self.num_inputs):
            if i == 0:
                layers.append(BottleneckCSP2(ch_in[i], ch_in[i]))
            else:
                layers.append(BottleneckCSP2(ch_out[i-1] + ch_in[i], ch_in[i]))
            layers.append(Conv(ch_in[i], ch_in[i]))
            layers.append(Conv(ch_in[i], ch_out[i]))

            if i != (self.num_inputs-1):
                layers.append(Concat())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i in range(self.num_inputs):
            Bottleneck_out = self.layers[i*4](x[i])
            Conv1_out = self.layers[i*4+1](Bottleneck_out)
            x[i] = self.layers[i*4+2](Conv1_out)

            if i != (self.num_inputs-1):
                if x[i].shape[2:] != x[i+1].shape[2:]:
                    x[i+1] = self.layers[i*4+3]((F.interpolate(x[i], size=x[i+1].shape[2:], mode='bilinear'), x[i+1]))
                else:
                    x[i+1] = self.layers[i*4+3]((x[i], x[i+1]))
        return x