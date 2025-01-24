import torch
import torch.nn as nn
import torch.nn.functional as F

from model.warp_m import warp
from model.galerkin_module import OverlapPatchEmbed,MotionFormerFullsize

def convrelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias), 
        nn.PReLU(out_channels)
    )

cor = {}
def get_cor( shape, device):
    k = (str(shape), str(device))
    if k not in cor:
        tenHorizontal = torch.linspace(-1.0, 1.0, shape[2], device=device).view(
            1, 1, 1, shape[2]).expand(shape[0], -1, shape[1], -1).permute(0, 2, 3, 1)
        tenVertical = torch.linspace(-1.0, 1.0, shape[1], device=device).view(
            1, 1, shape[1], 1).expand(shape[0], -1, -1, shape[2]).permute(0, 2, 3, 1)
        cor[k] = torch.cat([tenHorizontal, tenVertical], -1).to(device)
    return cor[k]
def resize(x, scale_factor):
    return F.interpolate(x, scale_factor=scale_factor, mode="bilinear", align_corners=False)
class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(in_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
            nn.PReLU(side_channels)
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)
        out[:, -self.side_channels:, :, :] = self.conv2(out[:, -self.side_channels:, :, :].clone())
        out = self.conv3(out)
        out[:, -self.side_channels:, :, :] = self.conv4(out[:, -self.side_channels:, :, :].clone())
        out = self.prelu(x + self.conv5(out))
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
       
        self.pyramid2 = nn.Sequential(
            convrelu(3, 36, 3, 2, 1), 
            convrelu(36, 36, 3, 1, 1)
        )
        self.pyramid3 = nn.Sequential(
            convrelu(36, 54, 3, 2, 1), 
            convrelu(54, 54, 3, 1, 1)
        )
        self.pyramid4 = nn.Sequential(
            convrelu(54, 72, 3, 2, 1), 
            convrelu(72, 72, 3, 1, 1)
        )
        
    def forward(self, img):
        f2 = self.pyramid2(img)
        f3 = self.pyramid3(f2)
        f4 = self.pyramid4(f3)
        return  f2, f3, f4


class Decoder4(nn.Module):
    def __init__(self):
        super(Decoder4, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(656+1, 144), 
            ResBlock(144, 24), 
            nn.ConvTranspose2d(144, 58+4, 4, 2, 1, bias=True)
        )
        self.patch_embed = OverlapPatchEmbed(patch_size=3,
                                                    stride = 1,
                                                    in_chans=72,
                                                    embed_dim=128)
        
       
        self.motionformer = MotionFormerFullsize(dim=128, motion_dim=128, num_heads=8,
                            mlp_ratio=4, qkv_bias=True, qk_scale=None,
                            drop=0, attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm,att_flag = '0'
        
        )
          
    def forward(self, f0, f1, embt):
        B, c, h, w = f0.shape

        x = torch.cat([f0,f1],dim=0)
        x, H, W = self.patch_embed(x)
        this_cor = get_cor((x.shape[0], H, W), x.device)
        x, x_motion = self.motionformer(x, this_cor, H, W, B)
        x =  x.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_motion = x_motion.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        f0_feat = x[:B]
        f1_feat = x[B:]
        mf_feat0 = x_motion[:B]
        mf_feat1 = x_motion[B:]
        
        f_in = torch.cat([embt*mf_feat0,(1-embt)*mf_feat1,f0, f0_feat,f1,f1_feat], 1)

        f_in = torch.cat([f_in, embt], 1)
        f_out = self.convblock(f_in)

        return f_out


class Decoder3(nn.Module):
    def __init__(self):
        super(Decoder3, self).__init__()
       
        self.convblock = nn.Sequential(
            convrelu(678, 162), 
            ResBlock(162, 24), 
            nn.ConvTranspose2d(162, 40+4, 4, 2, 1, bias=True)
        )
        self.patch_embed = OverlapPatchEmbed(patch_size=3,
                                                stride = 1,
                                                in_chans=54,
                                                embed_dim=128)
    
        
        self.motionformer = MotionFormerFullsize(dim=128, motion_dim=128, num_heads=8,
                                mlp_ratio=4, qkv_bias=True, qk_scale=None,
                                drop=0, attn_drop=0, drop_path=0, norm_layer=nn.LayerNorm,att_flag = '0'
            
            )
          

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        
        B, c, h, w = f0.shape
        x = torch.cat([f0_warp,f1_warp],dim=0)
        x, H, W = self.patch_embed(x)
        this_cor = get_cor((x.shape[0], H, W), x.device)
        x, x_motion = self.motionformer(x, this_cor, H, W, B)
        x =  x.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x_motion = x_motion.reshape(2*B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        f0_feat = x[:B]
        f1_feat = x[B:]
        mf_feat0 = x_motion[:B]
        mf_feat1 = x_motion[B:]
    
        f_in = torch.cat([ft_, f0_warp,f0_feat, f1_warp,f1_feat, up_flow0,mf_feat0, up_flow1,mf_feat1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder2(nn.Module):
    def __init__(self,out_c = 38):
        super(Decoder2, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(112, 108), 
            ResBlock(108, 24), 
            nn.ConvTranspose2d(108, out_c, 4, 2, 1, bias=True)
        )

    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class Decoder1(nn.Module):
    def __init__(self):
        super(Decoder1, self).__init__()
        self.convblock = nn.Sequential(
            convrelu(76, 72), 
            ResBlock(72, 24), 
            nn.ConvTranspose2d(72, 8, 4, 2, 1, bias=True)
        )
        
    def forward(self, ft_, f0, f1, up_flow0, up_flow1):
        f0_warp = warp(f0, up_flow0)
        f1_warp = warp(f1, up_flow1)
        f_in = torch.cat([ft_, f0_warp, f1_warp, up_flow0, up_flow1], 1)
        f_out = self.convblock(f_in)
        return f_out


class IFRnet_Model(nn.Module):
    def __init__(self):
        super(IFRnet_Model, self).__init__()
        self.encoder = Encoder()
        self.decoder4 = Decoder4()
        self.decoder3 = Decoder3()
      
        self.decoder2 = Decoder2()
        self.reduce = convrelu(32, 3)


    def forward(self,img0, img1, embt):
        mean_ = torch.cat([img0, img1], 2).mean(1, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
        img0 = img0 - mean_
        img1 = img1 - mean_


        f0_2, f0_3, f0_4 = self.encoder(img0)
        f1_2, f1_3, f1_4 = self.encoder(img1)

        out4 = self.decoder4(f0_4, f1_4, embt)
        up_flow0_4 = out4[:, 0:2]
        up_flow10_4 = out4[:, 2:4]
        up_flow1_4 = out4[:, 4:6]
        up_flow01_4 = out4[:, 6:8]
        ft_3_ = out4[:, 8:]

        
        out3 = self.decoder3(ft_3_, f0_3, f1_3, up_flow0_4, up_flow1_4)
      
        up_flow0_3 = out3[:, 0:2] + 2.0 * resize(up_flow0_4, scale_factor=2.0)
        up_flow10_3 = out3[:, 2:4] + 2.0 * resize(up_flow10_4, scale_factor=2.0)
        up_flow1_3 = out3[:, 4:6] + 2.0 * resize(up_flow1_4, scale_factor=2.0)
        up_flow01_3 = out3[:, 6:8] + 2.0 * resize(up_flow01_4, scale_factor=2.0)
        ft_2_ = out3[:, 8:]

        out2 = self.decoder2(ft_2_, f0_2, f1_2, up_flow0_3, up_flow1_3)
        up_flow0_2 = out2[:, 0:2] + 2.0 * resize(up_flow0_3, scale_factor=2.0)
        up_flow10_2 = out2[:, 2:4] + 2.0 * resize(up_flow10_3, scale_factor=2.0)
        up_flow1_2 = out2[:, 4:6] + 2.0 * resize(up_flow1_3, scale_factor=2.0)
        up_flow01_2 = out2[:, 6:8] + 2.0 * resize(up_flow01_3, scale_factor=2.0)
        up_mask_1 = torch.sigmoid(out2[:, 8:9])
        up_res_1 = out2[:, 9:]
        # F_{t->0},F_{t->1},F_{1->0},F_{0->1}
        flow_list = [up_flow0_2,up_flow1_2,up_flow10_2,up_flow01_2]

        img0_warp = warp(img0, up_flow0_2)
        img1_warp = warp(img1, up_flow1_2)
        imgt_merge = up_mask_1 * img0_warp + (1 - up_mask_1) * img1_warp + mean_

        out_feat = torch.cat([imgt_merge,up_res_1],dim=1)
       

        imgt_pred = self.reduce(out_feat)
        imgt_pred = torch.clamp(imgt_pred, 0, 1)
        return imgt_pred,flow_list
        
