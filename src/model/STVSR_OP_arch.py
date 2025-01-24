import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from model.general_module import ResidualBlocksWithInputConv, PixelShufflePack

from model.warp_m import warp
from model.flownet_module import IFRnet_Model
from model.general_module import make_coord

class Flow_STSR(nn.Module):
   

    def __init__(self,
                 mid_channels=64,
                 num_blocks=30,
                 padding=2,
                ):

        super().__init__()
        # flownet

        self.mid_channels = mid_channels
        self.padding = padding
        self.flownet = IFRnet_Model()
      
        self.backward_branch = ResidualBlocksWithInputConv(
            mid_channels + 3, mid_channels, num_blocks)

        self.forward_branch = ResidualBlocksWithInputConv(
            2 * mid_channels + 3, mid_channels, num_blocks)

        self.conv00 =  nn.Conv2d((64 + 2)*4+2, 256, 1)
        self.fc1 = nn.Conv2d(256, 256, 1)
        self.fc2 = nn.Conv2d(256, 3, 1)
        self.short_cut  = nn.Sequential(nn.Conv2d(mid_channels, mid_channels, 1),
                                        nn.ReLU(),
                                        nn.Conv2d(mid_channels, 3, 1)) 


       
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def load_flow_weight(self,flow_p):
        weight_dict = { strKey: tenWeight for strKey, tenWeight in torch.load(flow_p).items()  }
        self.flownet.load_state_dict(weight_dict)
    def compute_biflow(self, lrs,timestep):

        n, t, c, h, w = lrs.size()
        lrs_1 = lrs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lrs_2 = lrs[:, 1:, :, :, :].reshape(-1, c, h, w)
        timestep = torch.tensor(timestep, dtype=torch.float).cuda()
        timestep = timestep.reshape(1,1,1,1).repeat((t-1)*n, 1, h//8, w//8)
        insert_frame,flow_list  = self.flownet(lrs_1,lrs_2,timestep)
        flows_backward,flows_forward,flow_out_backward_half,flow_out_forward_half = flow_list[3],flow_list[2],flow_list[1],flow_list[0]

        insert_frame = insert_frame.view(n,t-1,3,h,w).contiguous()
        flows_backward = flows_backward.view(n,t-1,2,h,w).contiguous()
        flows_forward = flows_forward.view(n,t-1,2,h,w).contiguous()
        flow_out_backward_half = flow_out_backward_half.view(n,t-1,2,h,w).contiguous()
        flow_out_forward_half = flow_out_forward_half.view(n,t-1,2,h,w).contiguous()
        return insert_frame,flows_forward, flows_backward,flow_out_forward_half,flow_out_backward_half
    def make_grid(self,x):
        b,c,h,w = x.shape
        grid = make_coord((h,w),flatten=False).unsqueeze(0).repeat(b,1,1,1).permute(0,3,1,2).cuda()
        return grid
    def query_rgb(self, feat,target_size):

        feat_in = feat.clone()
        scale_max = 4
        H,W = target_size[0],target_size[1]
        scale_h = H / feat.shape[-2]
        scale_w = W / feat.shape[-1]
        coord = make_coord(target_size, flatten=False).cuda()
        coord = coord.unsqueeze(0).repeat(feat.shape[0],1,1,1)

        cell = torch.ones(1,2).cuda()
        cell[:, 0] *= 2 / H
        cell[:, 1] *= 2 / W
        cell_factor_h = max(scale_h/scale_max, 1)
        cell_factor_w = max(scale_w/scale_max, 1)
        cell[0][0] = cell[0][0]*cell_factor_h
        cell[0][1] = cell[0][1]*cell_factor_w

        pos_lr = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2
        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        rel_coords = []
        feat_s = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx + eps_shift
                coord_[:, :, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                feat_ = F.grid_sample(feat, coord_.flip(-1), mode='nearest', align_corners=False)
                old_coord = F.grid_sample(pos_lr, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - old_coord
                rel_coord[:, 0, :, :] *= feat.shape[-2]
                rel_coord[:, 1, :, :] *= feat.shape[-1]

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

                rel_coords.append(rel_coord)
                feat_s.append(feat_)
                
        rel_cell = cell.clone()
        rel_cell[:,0] *= feat.shape[-2]
        rel_cell[:,1] *= feat.shape[-1]

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]; areas[0] = areas[3]; areas[3] = t
        t = areas[1]; areas[1] = areas[2]; areas[2] = t

        for index, area in enumerate(areas):
            feat_s[index] = feat_s[index] * (area / tot_area).unsqueeze(1)
    
        
        grid = torch.cat([*rel_coords, *feat_s, \
            rel_cell.unsqueeze(-1).unsqueeze(-1).repeat(feat.shape[0],1,coord.shape[1],coord.shape[2])],dim=1)
       
        x = self.conv00(grid)
        ret = self.fc2(F.gelu(self.fc1(x)))
        
        short_cut =  self.short_cut(feat_in)
        ret = ret + F.grid_sample(short_cut, coord.flip(-1), mode='bilinear',\
                                padding_mode='border', align_corners=False)
        return ret
    def muti_feat_extract(self,lrs,timestep):
        n, t, c, h_input, w_input = lrs.size()
        h, w = lrs.size(3), lrs.size(4)
        insert_frame,flows_forward, flows_backward,flow_out_forward_half,flow_out_backward_half = self.compute_biflow(lrs,timestep)

        outputs = []
        outputs_lr = []
        mix_out = []

        HSF = lrs.new_zeros(n, self.mid_channels, h, w)
        for i in range(t - 1, -1, -1):
            # ----------------for lrs ---------------
            lr_curr = lrs[:, i, :, :, :]

            if i < t - 1:  
                flow = flows_backward[:, i, :, :, :]
                HSF = warp(HSF, flow)
           
            HSF = torch.cat([lr_curr, HSF], dim=1)
            HSF = self.backward_branch(HSF)
            outputs.append(HSF)
            # ----------------for inserted lrs ---------------
  
            if i < t - 1:
                # get warped_lr
                flow = flows_backward[:, i, :, :, :]
                flow_half = flow_out_backward_half[:,i,:,:,:]
                insert_HSF = warp(outputs[-2], flow_half)
                insert_HSF = torch.cat([insert_frame[:,i,:,:,:], insert_HSF], dim=1)
                insert_HSF = self.backward_branch(insert_HSF)
                outputs_lr.append([insert_frame[:,i,:,:,:],insert_HSF])
        for ix in range(2 * t - 1):
            if ix % 2 == 0:
                mix_out.append(outputs[ix // 2])
            else:
                mix_out.append(outputs_lr[ix // 2])
        outputs = mix_out[::-1]

        HSF = torch.zeros_like(HSF)
        grid = self.make_grid(HSF)
        for i in range(0, t):
            lr_curr = lrs[:, i, :, :, :]
            if i > 0:  
               
                flow = flows_forward[:, i - 1, :, :, :]
                previous_insert_HSF = HSF.clone()
                HSF = warp(HSF, flow)

            HSF = torch.cat([lr_curr, outputs[2 * i], HSF], dim=1)
            HSF = self.forward_branch(HSF)
            outputs[i * 2] = HSF

            # ----------------for inserted lrs ---------------
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                flow_half = flow_out_forward_half[:, i - 1, :, :, :]
                insert_HSF = warp(previous_insert_HSF, flow_half)
                outputs_feat = outputs[2 * i - 1][1]
                insert_HSF = torch.cat([insert_frame[:,i-1,:,:,:],outputs_feat , insert_HSF], dim=1)
                insert_HSF = self.forward_branch(insert_HSF)
                outputs[2 * i - 1] = insert_HSF
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.view(n*outputs.shape[1], self.mid_channels, h_input, w_input)
        return outputs
    def forward(self, lrs,timestep,target_size):
        n = lrs.shape[0]
        outputs = self.muti_feat_extract(lrs,timestep)
        outputs = self.query_rgb(outputs,target_size)
        outputs = outputs.view(n,-1,3,target_size[0],target_size[1])
        return outputs
    
 


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    (n, t, c, h, w) = 1, 3, 3, 64, 64
    fstsr = Flow_STSR()
  
    ivsr = fstsr.cuda()
    in_data = torch.zeros(n, t, c, h, w)
    in_data = in_data.cuda()
    time_step = 0.5
    target_size = (128,128)
    out_data = ivsr(in_data,time_step,target_size)
    print(out_data.shape)
