from timm.models.layers import trunc_normal_
import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import List


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))
class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class group_aggregation_bridge(nn.Module):
    def __init__(self, dim_xh, dim_xl, num_classes, k_size=3, d_list=[1, 2, 5, 7]):
        super().__init__()
        self.pre_project = nn.Conv2d(dim_xh, dim_xl, 1)
        group_size = dim_xl // 2
        self.g0 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + num_classes, data_format='channels_first'),
            nn.Conv2d(group_size + num_classes, group_size + num_classes, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[0] - 1)) // 2,
                      dilation=d_list[0], groups=group_size + num_classes)
        )
        self.g1 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + num_classes, data_format='channels_first'),
            nn.Conv2d(group_size + num_classes, group_size + num_classes, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[1] - 1)) // 2,
                      dilation=d_list[1], groups=group_size + num_classes)
        )
        self.g2 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + num_classes, data_format='channels_first'),
            nn.Conv2d(group_size + num_classes, group_size + num_classes, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[2] - 1)) // 2,
                      dilation=d_list[2], groups=group_size + num_classes)
        )
        self.g3 = nn.Sequential(
            LayerNorm(normalized_shape=group_size + num_classes, data_format='channels_first'),
            nn.Conv2d(group_size + num_classes, group_size + num_classes, kernel_size=3, stride=1,
                      padding=(k_size + (k_size - 1) * (d_list[3] - 1)) // 2,
                      dilation=d_list[3], groups=group_size + num_classes)
        )
        self.tail_conv = nn.Sequential(
            LayerNorm(normalized_shape=dim_xl * 2 + 4 * num_classes, data_format='channels_first'),
            nn.Conv2d(dim_xl * 2 + 4 * num_classes, dim_xl, 1)
        )

    def forward(self, xh, xl, mask):
        xh = self.pre_project(xh)
        mask = F.interpolate(mask, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = F.interpolate(xh, size=[xl.size(2), xl.size(3)], mode='bilinear', align_corners=True)
        xh = torch.chunk(xh, 4, dim=1)
        xl = torch.chunk(xl, 4, dim=1)

        x0 = self.g0(torch.cat((xh[0], xl[0], mask), dim=1))
        x1 = self.g1(torch.cat((xh[1], xl[1], mask), dim=1))
        x2 = self.g2(torch.cat((xh[2], xl[2], mask), dim=1))
        x3 = self.g3(torch.cat((xh[3], xl[3], mask), dim=1))
        x = torch.cat((x0, x1, x2, x3), dim=1)
        x = self.tail_conv(x)
        return x


class EnCoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, att=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(4, out_channels)
        )
        self.att_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            CoordAtt(inp=out_channels, oup=out_channels),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.GroupNorm(4, out_channels)
        )
        # 添加残差连接的卷积层，如果输入和输出通道数不一致
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.residual_conv = nn.Identity()
        self.att = att
        self.activation = nn.GELU()  # 添加一个激活函数，在残差连接之后使用
    def forward(self, x):
        if self.att:
            out = self.att_encoder(x)
        else:
            out = self.encoder(x)
        # 残差连接
        residual = self.residual_conv(x)
        out += residual
        # 激活函数
        out = self.activation(out)
        return out


class DeCoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            CoordAtt(inp=out_channels, oup=out_channels),
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.GroupNorm(4, out_channels)
        )
        # 添加残差连接的卷积层，如果输入和输出通道数不一致
        if in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        else:
            self.residual_conv = nn.Identity()
        self.activation = nn.GELU()  # 添加一个激活函数，在残差连接之后使用
    def forward(self, x):
        out = self.decoder(x)
        # 残差连接
        residual = self.residual_conv(x)
        out += residual
        # 激活函数
        out = self.activation(out)
        return out

class self_net(nn.Module):

    def __init__(self, n_classes=4, input_channels=3, c_list=[16,24,32,48,64,80], bridge=True, gt_ds=True):
        super().__init__()
        self.n_classes = n_classes
        self.bridge = bridge
        self.gt_ds = gt_ds
        self.input_channels = input_channels

        self.encoder1 = EnCoderBlock(input_channels,c_list[0],att=True)
        self.encoder2 = EnCoderBlock(c_list[0], c_list[1],att=True)
        self.encoder3 = EnCoderBlock(c_list[1], c_list[2],att=True)
        self.encoder4 = EnCoderBlock(c_list[2], c_list[3], att=True)
        self.encoder5 = EnCoderBlock(c_list[3], c_list[4], att=True)
        self.encoder6 = EnCoderBlock(c_list[4], c_list[5], att=True)


        if bridge:
            self.GAB1 = group_aggregation_bridge(c_list[1], c_list[0], num_classes=n_classes)
            self.GAB2 = group_aggregation_bridge(c_list[2], c_list[1], num_classes=n_classes)
            self.GAB3 = group_aggregation_bridge(c_list[3], c_list[2], num_classes=n_classes)
            self.GAB4 = group_aggregation_bridge(c_list[4], c_list[3], num_classes=n_classes)
            self.GAB5 = group_aggregation_bridge(c_list[5], c_list[4], num_classes=n_classes)
            print('group_aggregation_bridge was used')
        if gt_ds:
            self.gt_conv1 = nn.Sequential(nn.Conv2d(c_list[4], n_classes, 1))
            self.gt_conv2 = nn.Sequential(nn.Conv2d(c_list[3], n_classes, 1))
            self.gt_conv3 = nn.Sequential(nn.Conv2d(c_list[2], n_classes, 1))
            self.gt_conv4 = nn.Sequential(nn.Conv2d(c_list[1], n_classes, 1))
            self.gt_conv5 = nn.Sequential(nn.Conv2d(c_list[0], n_classes, 1))
            print('gt deep supervision was used')

        self.decoder1 = DeCoderBlock(c_list[5],c_list[4])
        self.decoder2 = DeCoderBlock(c_list[4],c_list[3])
        self.decoder3 = DeCoderBlock(c_list[3],c_list[2])
        self.decoder4 = DeCoderBlock(c_list[2],c_list[1])
        self.decoder5 = DeCoderBlock(c_list[1],c_list[0])


        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(c_list[0], n_classes, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_backbone(self):
        for name,param in self.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print("Backbone frozen. Only 'classifier' layer is trainable.")

    def unfreeze_backbone(self):
        for name,param in self.named_parameters():
            param.requires_grad = True
        print("All layers trainable.")

    def forward(self, x):
        out = F.max_pool2d(self.encoder1(x),2)
        t1 = out  # b, c0, H/2, W/2

        out = F.max_pool2d(self.encoder2(out),2)
        t2 = out  # b, c1, H/4, W/4

        out = F.max_pool2d(self.encoder3(out),2)
        t3 = out  # b, c2, H/8, W/8

        out = F.max_pool2d(self.encoder4(out),2)
        t4 = out  # b, c3, H/16, W/16
        #out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))
        out = F.max_pool2d(self.encoder5(out),2)
        t5 = out  # b, c4, H/32, W/32
        #out = F.gelu(self.encoder6(out))  # b, c5, H/32, W/32
        out = self.encoder6(out)
        t6 = out
        out5 = self.decoder1(out)  # b, c4, H/32, W/32
        if self.gt_ds:
            gt_pre5 = self.gt_conv1(out5)
            t5 = self.GAB5(t6, t5, gt_pre5)
            gt_pre5 = F.interpolate(gt_pre5, scale_factor=32, mode='bilinear', align_corners=True)
        else:
            t5 = self.GAB5(t6, t5)
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32

        # out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
        #                             align_corners=True))  # b, c3, H/16, W/16
        out4 = F.interpolate(self.decoder2(out5), scale_factor=(2, 2),mode='bilinear', align_corners=True)
        if self.gt_ds:
            gt_pre4 = self.gt_conv2(out4)
            t4 = self.GAB4(t5, t4, gt_pre4)
            gt_pre4 = F.interpolate(gt_pre4, scale_factor=16, mode='bilinear', align_corners=True)
        else:
            t4 = self.GAB4(t5, t4)
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        # out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
        #                             align_corners=True))  # b, c2, H/8, W/8
        out3 = F.interpolate(self.decoder3(out4), scale_factor=(2, 2),mode='bilinear', align_corners=True)
        if self.gt_ds:
            gt_pre3 = self.gt_conv3(out3)
            t3 = self.GAB3(t4, t3, gt_pre3)
            gt_pre3 = F.interpolate(gt_pre3, scale_factor=8, mode='bilinear', align_corners=True)
        else:
            t3 = self.GAB3(t4, t3)
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        # out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
        #                             align_corners=True))  # b, c1, H/4, W/4
        out2 = F.interpolate(self.decoder4(out3), scale_factor=(2, 2),mode='bilinear', align_corners=True)
        if self.gt_ds:
            gt_pre2 = self.gt_conv4(out2)
            t2 = self.GAB2(t3, t2, gt_pre2)
            gt_pre2 = F.interpolate(gt_pre2, scale_factor=4, mode='bilinear', align_corners=True)
        else:
            t2 = self.GAB2(t3, t2)
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        # out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
        #                             align_corners=True))  # b, c0, H/2, W/2
        out1 = F.interpolate(self.decoder5(out2), scale_factor=(2, 2),mode='bilinear', align_corners=True)
        if self.gt_ds:
            gt_pre1 = self.gt_conv5(out1)
            t1 = self.GAB1(t2, t1, gt_pre1)
            gt_pre1 = F.interpolate(gt_pre1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            t1 = self.GAB1(t2, t1)
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2
        out0 = F.interpolate(self.classifier(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W

        if self.gt_ds:  # return logits
            return (gt_pre5, gt_pre4, gt_pre3, gt_pre2, gt_pre1), out0
        else:
            return out0