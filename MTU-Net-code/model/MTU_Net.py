import torch
import torch.nn as nn

from model.vit import ViT

from einops import rearrange

class _FCNHead(nn.Module):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, momentum, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=inter_channels,kernel_size=3, padding=1, bias=False),
        norm_layer(inter_channels, momentum=momentum),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=inter_channels, out_channels=channels,kernel_size=1)
        )
    # pylint: disable=arguments-differ
    def forward(self, x):
        return self.block(x)


class Res_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # self.fca = FCA_Layer(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out



class res_UNet(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter):
        super(res_UNet, self).__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)


        self.conv0_0_0 = self._make_layer(block, input_channels, input_channels)
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
        # for name,value in self.conv0_0.named_parameters():
        #     value.requires_grad = False
        self.conv1_0 = self._make_layer(block, nb_filter[0],   nb_filter[1], num_blocks[0])
        # for name,value in self.conv1_0.named_parameters():
        #     value.requires_grad = False
        self.conv2_0 = self._make_layer(block, nb_filter[1],   nb_filter[2], num_blocks[1])
        # for name,value in self.conv2_0.named_parameters():
        #     value.requires_grad = False
        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])
        # for name,value in self.conv3_0.named_parameters():
        #     value.requires_grad = False
        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])
        # for name,value in self.conv4_0.named_parameters():
        #     value.requires_grad = False

        self.vit5 = ViT(img_dim=1024, in_channels=nb_filter[0], embedding_dim=nb_filter[2], head_num=1, mlp_dim=64 * 64,
                        block_num=1, patch_dim=16, classification=False, num_classes=1)

        self.vit4 = ViT(img_dim=512, in_channels=nb_filter[1], embedding_dim=nb_filter[2], head_num=1, mlp_dim=64 * 64,
                        block_num=1, patch_dim=8, classification=False, num_classes=1)
        self.vit3 = ViT(img_dim=256, in_channels=nb_filter[2], embedding_dim=nb_filter[2], head_num=1, mlp_dim=64 * 64,
                        block_num=1, patch_dim=4, classification=False, num_classes=1)
        self.vit2 = ViT(img_dim=128, in_channels=nb_filter[3], embedding_dim=nb_filter[2],head_num=1, mlp_dim=64*64,
                        block_num=1, patch_dim=2, classification=False,num_classes=1)
        self.vit1 = ViT(img_dim=64, in_channels=nb_filter[4], embedding_dim=nb_filter[4], head_num=1, mlp_dim=64*64,
                        block_num=1, patch_dim=1, classification=False, num_classes=1)

        # self.conv4_1 = self._make_layer(block, nb_filter[4] + nb_filter[4], nb_filter[4])

        self.conv3_1_1 = self._make_layer(block, nb_filter[2] + nb_filter[2] + nb_filter[2] + nb_filter[2] + nb_filter[4],
                                        nb_filter[4])


        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])
        # for name,value in self.conv3_1.named_parameters():
        #     value.requires_grad = False
        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
        # for name,value in self.conv2_2.named_parameters():
        #     value.requires_grad = False
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
        # for name,value in self.conv1_3.named_parameters():
        #     value.requires_grad = False
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])
        # for name,value in self.conv0_4.named_parameters():
        #     value.requires_grad = False

        # self.final1 = nn.Conv2d(nb_filter[0]+3, num_classes, kernel_size=1)
        self.head = _FCNHead(nb_filter[0], channels=num_classes, momentum=0.9)

        # self.final2 = nn.BatchNorm2d(num_classes)
        self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        # for name,value in self.final.named_parameters():
        #     value.requires_grad = False

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)






    def forward(self, input):
        # x0_0_0 = self.conv0_0_0(input)
        x0_0 = self.conv0_0(input)
        # (4,16,256,256)
        x1_0 = self.conv1_0(self.pool(x0_0))
        # (4,32,128,128)
        x2_0 = self.conv2_0(self.pool(x1_0))
        # (4,64,64,64)
        x3_0 = self.conv3_0(self.pool(x2_0)) # (4,128,32,32)

        out = self.conv4_0(self.pool(x3_0))
        # (4,256,16,16)

        out = torch.cat([rearrange(self.vit2(x3_0), "b (x y) c -> b c x y", x=64, y=64),rearrange(self.vit3(x2_0), "b (x y) c -> b c x y", x=64, y=64),rearrange(self.vit4(x1_0), "b (x y) c -> b c x y", x=64, y=64),rearrange(self.vit5(x0_0), "b (x y) c -> b c x y", x=64, y=64),out], 1)
    
        out = self.conv3_1_1(out)

   
        out = self.conv3_1(torch.cat([x3_0, self.up(out)], 1))
  
        out = self.conv2_2(torch.cat([x2_0, self.up(out)], 1))

        out = self.conv1_3(torch.cat([x1_0, self.up(out)], 1))
        
        out = self.conv0_4(torch.cat([x0_0, self.up(out)], 1))

        out = self.final(out)

        return out
