import torch
import torch.nn as nn
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        else:
            pass


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        weight_init(self)

    def forward(self, x):
        return self.conv_bn(x)


class Reduction(nn.Module):
    def __init__(self, in_channel=32, out_channel=32):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            BasicConv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        weight_init(self)

    def forward(self, x):
        return self.reduce(x)


class ConvUpsample(nn.Module):
    def __init__(self, channel=32):
        super(ConvUpsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, kernel_size=1)
        weight_init(self)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class PPM(nn.Module):
    # pyramid pooling module
    def __init__(self, channel):
        super(PPM, self).__init__()
        self.scales = [1, 2, 4, 8]
        self.poolings = [nn.AdaptiveAvgPool2d((s, s)) for s in self.scales]
        self.convs = nn.ModuleList([BasicConv2d(channel, channel, kernel_size=3, padding=1)
                                    for i in range(len(self.scales))])
        self.cat = BasicConv2d(len(self.scales) * channel, channel, 1)
        weight_init(self)

    def forward(self, x):
        pool_x = []
        for i, pooling in enumerate(self.poolings):
            pool_x.append(self.convs[i](pooling(x)))

        inp_x = []
        for i in range(len(self.scales)):
            inp_x.append(F.upsample(pool_x[i], size=x.size()[2:], mode='bilinear', align_corners=True))
        return x + self.cat(torch.cat(inp_x, dim=1))


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        weight_init(self)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        weight_init(self)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        weight_init(self)

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x


class FusionOutput(nn.Module):
    def __init__(self, channel):
        super(FusionOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsampled1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catd1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_catd2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_catd3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.outputd = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.att_1 = Attention(channel)
        self.att_2 = Attention(channel)
        self.att_3 = Attention(channel)
        self.att_4 = Attention(channel)

        weight_init(self)

    def forward(self, s1, s2, s3, s4, d1, d2, d3, d4):
        s4 = s4 * self.att_4(d4) + s4

        d3 = torch.cat((d3, self.conv_upsampled1(self.upsample(d4))), 1)
        d3 = self.conv_catd1(d3)

        s3 = torch.cat((s3, self.conv_upsample1(self.upsample(s4))), 1)
        s3 = self.conv_cat1(s3)
        s3 = s3 * self.att_3(d3) + s3

        d2 = torch.cat((d2, self.conv_upsampled2(self.upsample(d3))), 1)
        d2 = self.conv_catd2(d2)

        s2 = torch.cat((s2, self.conv_upsample2(self.upsample(s3))), 1)
        s2 = self.conv_cat2(s2)
        s2 = s2 * self.att_2(d2) + s2

        d1 = torch.cat((d1, self.conv_upsampled3(self.upsample(d2))), 1)
        d1 = self.conv_catd3(d1)

        s1 = torch.cat((s1, self.conv_upsample3(self.upsample(s2))), 1)
        s1 = self.conv_cat3(s1)
        s1 = s1 * self.att_1(d1) + s1

        d = self.outputd(d1)
        s = self.output(s1)
        return s, d


class Attention(nn.Module):
    def __init__(self, channel=32):
        super(Attention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channel, channel, kernel_size=1)
        self.activation = nn.Sigmoid()
        weight_init(self)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class FusionOutput_v2(nn.Module):
    def __init__(self, channel):
        super(FusionOutput_v2, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.output_s1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_s2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_s3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_s4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsampled1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catd1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_catd2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_catd3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.output_d1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_d2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_d3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_d4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsamplee1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cate1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cate2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_cate3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            nn.ReLU(inplace=True)
        )
        self.output_e1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_e2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_e3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.output_e4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.att_1_s = Attention(channel)
        self.att_2_s = Attention(channel)
        self.att_3_s = Attention(channel)
        self.att_4_s = Attention(channel)
        self.att_1_e = Attention(channel)
        self.att_2_e = Attention(channel)
        self.att_3_e = Attention(channel)
        self.att_4_e = Attention(channel)
        self.att_1_d = Attention(channel)
        self.att_2_d = Attention(channel)
        self.att_3_d = Attention(channel)
        self.att_4_d = Attention(channel)

        weight_init(self)

    def forward(self, s1, s2, s3, s4, d1, d2, d3, d4, e1, e2, e3, e4):
        tmp_s4, tmp_d4, tmp_e4 = s4, d4, e4
        s4 = tmp_s4 * tmp_e4 + tmp_s4 * self.att_4_s(tmp_d4) + tmp_s4
        e4 = tmp_e4 * tmp_s4 * self.att_4_e(tmp_d4)
        d4 = tmp_d4 * tmp_e4 + tmp_d4 * self.att_4_d(tmp_s4) + tmp_d4

        d3 = torch.cat((d3, self.conv_upsampled1(self.upsample(d4))), 1)
        d3 = self.conv_catd1(d3)

        e3 = torch.cat((e3, self.conv_upsamplee1(self.upsample(e4))), 1)
        e3 = self.conv_cate1(e3)

        s3 = torch.cat((s3, self.conv_upsample1(self.upsample(s4))), 1)
        s3 = self.conv_cat1(s3)
        tmp_s3, tmp_d3, tmp_e3 = s3, d3, e3
        s3 = tmp_s3 * tmp_e3 + tmp_s3 * self.att_3_s(tmp_d3) + tmp_s3
        e3 = tmp_e3 * tmp_s3 * self.att_3_e(tmp_d3)
        d3 = tmp_d3 * tmp_e3 + tmp_d3 * self.att_3_d(tmp_s3) + tmp_d3

        d2 = torch.cat((d2, self.conv_upsampled2(self.upsample(d3))), 1)
        d2 = self.conv_catd2(d2)

        e2 = torch.cat((e2, self.conv_upsamplee2(self.upsample(e3))), 1)
        e2 = self.conv_cate2(e2)

        s2 = torch.cat((s2, self.conv_upsample2(self.upsample(s3))), 1)
        s2 = self.conv_cat2(s2)
        tmp_s2, tmp_d2, tmp_e2 = s2, d2, e2
        s2 = tmp_s2 * tmp_e2 + tmp_s2 * self.att_2_s(tmp_d2) + tmp_s2
        e2 = tmp_e2 * tmp_s2 * self.att_2_e(tmp_d2)
        d2 = tmp_d2 * tmp_e2 + tmp_d2 * self.att_2_d(tmp_s2) + tmp_d2

        d1 = torch.cat((d1, self.conv_upsampled3(self.upsample(d2))), 1)
        d1 = self.conv_catd3(d1)

        e1 = torch.cat((e1, self.conv_upsamplee3(self.upsample(e2))), 1)
        e1 = self.conv_cate3(e1)

        s1 = torch.cat((s1, self.conv_upsample3(self.upsample(s2))), 1)
        s1 = self.conv_cat3(s1)
        tmp_s1, tmp_d1, tmp_e1 = s1, d1, e1
        s1 = tmp_s1 * tmp_e1 + tmp_s1 * self.att_1_s(tmp_d1) + tmp_s1
        e1 = tmp_e1 * tmp_s1 * self.att_1_e(tmp_d1)
        d1 = tmp_d1 * tmp_e1 + tmp_d1 * self.att_1_d(tmp_s4) + tmp_d1

        s1 = self.output_s1(s1)
        s2 = self.output_s2(s2)
        s3 = self.output_s3(s3)
        s4 = self.output_s4(s4)
        e1 = self.output_e1(e1)
        e2 = self.output_e2(e2)
        e3 = self.output_e3(e3)
        e4 = self.output_e4(e4)
        d1 = self.output_d1(d1)
        d2 = self.output_d2(d2)
        d3 = self.output_d3(d3)
        d4 = self.output_d4(d4)
        return s1, s2, s3, s4, e1, e2, e3, e4, d1, d2, d3, d4


class MultiFusionOutput(nn.Module):
    def __init__(self, channel):
        super(MultiFusionOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsamples1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamples2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamples3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_downsamples4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamples2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamples3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cats1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cats2 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cats3 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cats4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.outputs = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsampled1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_downsampled4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsampled2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsampled3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_catd1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_catd2 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_catd3 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_catd4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.outputd = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.conv_upsamplee1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsamplee3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_downsamplee4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamplee2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_downsamplee3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cate1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cate2 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cate3 = nn.Sequential(
            BasicConv2d(3 * channel, 3 * channel, 3, padding=1),
            BasicConv2d(3 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.conv_cate4 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1),
            # nn.ReLU(inplace=True)
        )
        self.outpute = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

        self.att_1 = Attention(channel)
        self.att_2 = Attention(channel)
        self.att_3 = Attention(channel)
        self.att_4 = Attention(channel)

        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        weight_init(self)

    def forward(self, s1, s2, s3, s4, d1, d2, d3, d4, e1, e2, e3, e4):
        d4 = torch.cat((d4, self.conv_downsampled4(self.pooling(d3))), 1)
        d4 = self.conv_catd4(d4)

        e4 = torch.cat((e4, self.conv_downsamplee4(self.pooling(e3))), 1)
        e4 = self.conv_cate4(e4)

        s4 = torch.cat((s4, self.conv_downsamples4(self.pooling(s3))), 1)
        s4 = self.conv_cats4(s4)
        s4 = s4 * e4 + s4 * self.att_4(d4) + s4

        d3 = torch.cat((d3, self.conv_upsampled3(self.upsample(d4)), self.conv_downsampled3(self.pooling(d2))), 1)
        d3 = self.conv_catd3(d3)

        e3 = torch.cat((e3, self.conv_upsamplee3(self.upsample(e4)), self.conv_downsamplee3(self.pooling(e2))), 1)
        e3 = self.conv_cate3(e3)

        s3 = torch.cat((s3, self.conv_upsamples3(self.upsample(s4)), self.conv_downsamples3(self.pooling(s2))), 1)
        s3 = self.conv_cats3(s3)
        s3 = s3 * e3 + s3 * self.att_3(d3) + s3

        d2 = torch.cat((d2, self.conv_upsampled2(self.upsample(d3)), self.conv_downsampled2(self.pooling(d1))), 1)
        d2 = self.conv_catd2(d2)

        e2 = torch.cat((e2, self.conv_upsamplee2(self.upsample(e3)), self.conv_downsamplee2(self.pooling(e1))), 1)
        e2 = self.conv_cate2(e2)

        s2 = torch.cat((s2, self.conv_upsamples2(self.upsample(s3)), self.conv_downsamples2(self.pooling(s1))), 1)
        s2 = self.conv_cats2(s2)
        s2 = s2 * e2 + s2 * self.att_2(d2) + s2

        d1 = torch.cat((d1, self.conv_upsampled1(self.upsample(d2))), 1)
        d1 = self.conv_catd1(d1)

        e1 = torch.cat((e1, self.conv_upsamplee1(self.upsample(e2))), 1)
        e1 = self.conv_cate1(e1)

        s1 = torch.cat((s1, self.conv_upsamples1(self.upsample(s2))), 1)
        s1 = self.conv_cats1(s1)
        s1 = s1 * e1 + s1 * self.att_1(d1) + s1

        e = self.outpute(e1)
        s = self.outputs(s1)
        return s, e
