from nets.UNet import UNet, UNet_Nested, \
    UNet_Nested_dilated, ResUnet, Unet_with_layer, layer_head,CLAN
import torch.nn as nn
affine_par = True
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

def net_builder(args):
    if args == 'layer_head':
        net = layer_head(n_classes=2)
    elif args.name == 'ResUnet':
        net = ResUnet(channel=3)
    elif args.name == 'CLAN':
        net = CLAN(Bottleneck, [3, 4, 23, 3],args.number_classes)
    elif args.name == 'unet':
        net = UNet(n_classes=args.number_classes, feature_scale=4)
    elif args.name == 'Unet_with_layer':
        net = Unet_with_layer(n_classes=args.number_classes, feature_scale=4)
    elif args.name == 'unet_nested':
        net = UNet_Nested(feature_scale=4)
    elif args.name == 'unet_nested_dilated':
        net = UNet_Nested_dilated(feature_scale=4)
    # elif args.name == 'TransUnet':
    #     config_vit = CONFIGS_ViT_seg[args.Transmodel]
    #     net = ViT_seg(config_vit, img_size=args.resize[0], num_classes=args.number_classes).cuda()
    else:
        raise NameError("Unknow Model Name!")
    return net
