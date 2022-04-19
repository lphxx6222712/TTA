import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.layers import unetConv2,unetUp,unetConv2_dilation
from models.utils.init_weights import init_weights
from models.utils.Resmodules import ResidualConv, Upsample, Squeeze_Excite_Block, ASPP, AttentionBlock, Upsample_
import numpy as np
from torch.nn.modules.utils import _pair
# class UNet(nn.Module):
#
#     def __init__(self, in_channels=3,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True):
#         super(UNet, self).__init__()
#         self.is_deconv = is_deconv
#         self.in_channels = in_channels
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#
#         filters = [64, 128, 256, 512, 1024]
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2)
#
#         self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
#
#         self.cls = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Conv2d(256,n_classes-1,1),
#             nn.AdaptiveMaxPool2d(1),
#             nn.Sigmoid())
#
#         self.mask_cls = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Conv2d(16,n_classes-1,1),
#             #nn.AdaptiveMaxPool2d(1),
#             nn.Sigmoid())
#
#         # upsampling
#         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
#         self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
#         self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
#         self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
#
#         # final conv (without any concat)
#         self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
#         # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
#         # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
#
#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init_weights(m, init_type='kaiming')
#
#     def forward(self, inputs):
#         conv1 = self.conv1(inputs)       # 16*512*1024
#         maxpool1 = self.maxpool1(conv1)  # 16*256*512
#
#         conv2 = self.conv2(maxpool1)     # 32*256*512
#         maxpool2 = self.maxpool2(conv2)  # 32*128*256
#
#         conv3 = self.conv3(maxpool2)     # 64*128*256
#         maxpool3 = self.maxpool3(conv3)  # 64*64*128
#
#         conv4 = self.conv4(maxpool3)     # 128*64*128
#         maxpool4 = self.maxpool4(conv4)  # 128*32*64
#
#         center = self.center(maxpool4)   # 256*32*64
#         cls_branch = self.cls(center).squeeze()
#
#
#         up4 = self.up_concat4(center,conv4)  # 128*64*128
#         up3 = self.up_concat3(up4,conv3)     # 64*128*256
#         up2 = self.up_concat2(up3,conv2)     # 32*256*512
#         up1 = self.up_concat1(up2,conv1)     # 16*512*1024
#
#         final_1 = self.final_1(up1)
#         # final_2 = self.final_2(up1)
#         # final_3 = self.final_3(up1)
#
#         #global average pooling
#         GAP = torch.nn.functional.adaptive_avg_pool2d(up1,output_size=1)
#         mask_cls = self.mask_cls(GAP).squeeze()
#
#         return F.log_softmax(final_1,dim=1),cls_branch, mask_cls

class Unet_with_layer(nn.Module):

    def __init__(self, in_channels=3, n_classes=9, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(Unet_with_layer, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, n_classes - 1, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)

        # upsampling
        self.up_concat2_4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat2_3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2_2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat2_1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_2 = nn.Conv2d(filters[0], 9, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64
        cls_branch = self.cls(center).squeeze()

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        final_1 = self.final_1(up1)

        up_2_4 = self.up_concat2_4(center, conv4)
        up_2_3 = self.up_concat2_3(up_2_4, conv3)
        up_2_2 = self.up_concat2_2(up_2_3, conv2)
        up_2_1 = self.up_concat2_1(up_2_2, conv1)

        final_2 = self.final_2(up_2_1)

        # final_2 = self.final_2(up1)
        # final_3 = self.final_3(up1)

        # return F.log_softmax(final_1,dim=1),cls_branch, mask_cls
        return F.log_softmax(final_1, dim=1), cls_branch, F.log_softmax(final_2, dim=1), up1

class layer_head(nn.Module):

    def     __init__(self, in_channels=3, n_classes=2, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(layer_head, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]


        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)

        # upsampling
        self.up_concat2_4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat2_3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2_2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat2_1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.bottleneck = unetConv2(filters[0], filters[0], self.is_batchnorm)
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):

        inputs = self.bottleneck(inputs)
        final_2 = self.final_2(inputs)

        # final_2 = self.final_2(up1)
        # final_3 = self.final_3(up1)

        # return F.log_softmax(final_1,dim=1),cls_branch, mask_cls
        return F.log_softmax(final_2, dim=1), inputs

class UNet(nn.Module):

    def __init__(self, in_channels=3, n_classes=3, feature_scale=4, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256, n_classes - 1, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*1024
        maxpool1 = self.maxpool1(conv1)  # 16*256*512

        conv2 = self.conv2(maxpool1)  # 32*256*512
        maxpool2 = self.maxpool2(conv2)  # 32*128*256

        conv3 = self.conv3(maxpool2)  # 64*128*256
        maxpool3 = self.maxpool3(conv3)  # 64*64*128

        conv4 = self.conv4(maxpool3)  # 128*64*128
        maxpool4 = self.maxpool4(conv4)  # 128*32*64

        center = self.center(maxpool4)  # 256*32*64
        cls_branch = self.cls(center).squeeze()

        up4 = self.up_concat4(center, conv4)  # 128*64*128
        up3 = self.up_concat3(up4, conv3)  # 64*128*256
        up2 = self.up_concat2(up3, conv2)  # 32*256*512
        up1 = self.up_concat1(up2, conv1)  # 16*512*1024

        final_1 = self.final_1(up1)

        # final_2 = self.final_2(up1)
        # final_3 = self.final_3(up1)

        # return F.log_softmax(final_1,dim=1),cls_branch, mask_cls
        return F.log_softmax(final_1, dim=1), cls_branch, center, up1

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=3,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds=True):
        super(UNet_Nested, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256,3,1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv,3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv,3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv,3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv,4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv,4)
        
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv,5)
        
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)       
        maxpool0 = self.maxpool0(X_00)  
        X_10= self.conv10(maxpool0)     
        maxpool1 = self.maxpool1(X_10) 
        X_20 = self.conv20(maxpool1)    
        maxpool2 = self.maxpool2(X_20)  
        X_30 = self.conv30(maxpool2)     
        maxpool3 = self.maxpool3(X_30)  
        X_40 = self.conv40(maxpool3)   
        cls_branch = self.cls(X_40).squeeze()
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1+final_2+final_3+final_4)/4

        if self.is_ds:
            return F.log_softmax(final,dim=1),cls_branch
        else:
            return F.log_softmax(final_4),cls_branch


class UNet_Nested_dilated(nn.Module):

    def __init__(self, in_channels=3,n_classes=4,feature_scale=4, is_deconv=True, is_batchnorm=True,is_ds=True):
        super(UNet_Nested_dilated, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool0 = nn.MaxPool2d(kernel_size=2)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        self.dilated = unetConv2_dilation(filters[4],filters[4],self.is_batchnorm)

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(256,3,1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv,3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv,3)
        self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv,3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv,4)
        self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv,4)
        
        self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv,5)
        
        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)      
        maxpool0 = self.maxpool0(X_00)  
        X_10= self.conv10(maxpool0)     
        maxpool1 = self.maxpool1(X_10)  
        X_20 = self.conv20(maxpool1)     
        maxpool2 = self.maxpool2(X_20)  
        X_30 = self.conv30(maxpool2)     
        maxpool3 = self.maxpool3(X_30)  
        X_40 = self.conv40(maxpool3)   
        X_40_d = self.dilated(X_40)
        cls_branch = self.cls(X_40_d).squeeze()
        # column : 1
        X_01 = self.up_concat01(X_10,X_00)
        X_11 = self.up_concat11(X_20,X_10)
        X_21 = self.up_concat21(X_30,X_20)
        X_31 = self.up_concat31(X_40_d,X_30)
        # column : 2
        X_02 = self.up_concat02(X_11,X_00,X_01)
        X_12 = self.up_concat12(X_21,X_10,X_11)
        X_22 = self.up_concat22(X_31,X_20,X_21)
        # column : 3
        X_03 = self.up_concat03(X_12,X_00,X_01,X_02)
        X_13 = self.up_concat13(X_22,X_10,X_11,X_12)
        # column : 4
        X_04 = self.up_concat04(X_13,X_00,X_01,X_02,X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1+final_2+final_3+final_4)/4

        # if self.is_ds:
        #     return F.log_softmax(final),cls_branch,
        # else:
        return F.log_softmax(final_4),cls_branch, X_40_d, final_4

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 4, 1, 1),
            nn.Sigmoid(),
        )

        self.cls = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, 4 - 1, 1),
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid())

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)

        out_cls = self.cls(x4).squeeze()

        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return F.log_softmax(output, dim=1), out_cls, x4, x10
    #F.log_softmax(final_1,dim=1)

class ResUnetPlusPlus(nn.Module):
    def __init__(self, channel, filters=[32, 64, 128, 256, 512]):
        super(ResUnetPlusPlus, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.squeeze_excite1 = Squeeze_Excite_Block(filters[0])

        self.residual_conv1 = ResidualConv(filters[0], filters[1], 2, 1)

        self.squeeze_excite2 = Squeeze_Excite_Block(filters[1])

        self.residual_conv2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.squeeze_excite3 = Squeeze_Excite_Block(filters[2])

        self.residual_conv3 = ResidualConv(filters[2], filters[3], 2, 1)

        self.aspp_bridge = ASPP(filters[3], filters[4])

        self.attn1 = AttentionBlock(filters[2], filters[4], filters[4])
        self.upsample1 = Upsample_(2)
        self.up_residual_conv1 = ResidualConv(filters[4] + filters[2], filters[3], 1, 1)

        self.attn2 = AttentionBlock(filters[1], filters[3], filters[3])
        self.upsample2 = Upsample_(2)
        self.up_residual_conv2 = ResidualConv(filters[3] + filters[1], filters[2], 1, 1)

        self.attn3 = AttentionBlock(filters[0], filters[2], filters[2])
        self.upsample3 = Upsample_(2)
        self.up_residual_conv3 = ResidualConv(filters[2] + filters[0], filters[1], 1, 1)

        self.aspp_out = ASPP(filters[1], filters[0])

        self.output_layer = nn.Conv2d(filters[0], 4, 1)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)

        x2 = self.squeeze_excite1(x1)
        x2 = self.residual_conv1(x2)

        x3 = self.squeeze_excite2(x2)
        x3 = self.residual_conv2(x3)

        x4 = self.squeeze_excite3(x3)
        x4 = self.residual_conv3(x4)

        x5 = self.aspp_bridge(x4)

        x6 = self.attn1(x3, x5)
        x6 = self.upsample1(x6)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.up_residual_conv1(x6)

        x7 = self.attn2(x2, x6)
        x7 = self.upsample2(x7)
        x7 = torch.cat([x7, x2], dim=1)
        x7 = self.up_residual_conv2(x7)

        x8 = self.attn3(x1, x7)
        x8 = self.upsample3(x8)
        x8 = torch.cat([x8, x1], dim=1)
        x8 = self.up_residual_conv3(x8)

        x9 = self.aspp_out(x8)
        out = self.output_layer(x9)

        return out
affine_par = True

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out

class CLAN(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(CLAN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.layer5(x)
        x2 = self.layer6(x)
        return x1, x2

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        #b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]