import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
from data.Vid4 import Vid4
from model.STVSR_OP_arch import Flow_STSR

def single_forward(model, imgs_in, time_ls, target_size, scale, crop_shape):
    out_ls = []

    with torch.no_grad():
        b, n, c, h, w = imgs_in.size()
        bordered_size = (round(scale * (h)), round(scale * (w)))

        multi_feat = model.muti_feat_extract(imgs_in.cuda(), time_ls)
        features_cpu = multi_feat.cpu()
        for frame_feat in features_cpu:
            frame_feat = frame_feat.unsqueeze(0)  
            frame_feat = frame_feat.to(device)

            output = model.query_rgb(frame_feat, bordered_size)
            output = output.view(-1, 3, bordered_size[0], bordered_size[1]).squeeze(0)
            x1, x2, y1, y2 = crop_shape
            output = output[:, x1:x2, y1:y2]
            out_ls.append(output.detach().cpu())
    return out_ls

def vid_test(data_dir, output_p, scale):
    vid_data = Vid4(data_dir, scale)
    Vid_ST_dl = DataLoader(vid_data, batch_size=1, shuffle=False)

    for ix, (GT, LR, name, crop_shape) in enumerate(Vid_ST_dl):  
        target_size = (GT.shape[-2], GT.shape[-1])
        seq_name = name[0]
        this_p = os.path.join(output_p, seq_name)
        if not os.path.exists(this_p):
            os.makedirs(this_p)
        LR = LR.to(device)
        
        out = single_forward(model, LR, 0.5, target_size, scale, crop_shape)
            
        for i in range(len(out)):
            img = out[i].permute(1, 2, 0).numpy() * 255.0
            img = img[:, :, ::-1]
            img_path = os.path.join(this_p, f"{str(i).zfill(8)}.png")
            print(img_path)
            cv2.imwrite(img_path, img)

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=4 python test_vid4.py
    # python test_vid4.py --device cuda --test_p /path/to/test/output --weight_p /path/to/weights --data_dir /path/to/Vid4/GT --scale 4.0
    parser = argparse.ArgumentParser(description="STVSR_OP testing for Vid4 dataset")
    parser.add_argument('--device', default='cuda', help='Device to use (e.g., cuda, cpu)')
    parser.add_argument('--test_p', default='xxx', help='Path to test output')
    parser.add_argument('--weight_p', default='xxx', help='Path to model weights')
    parser.add_argument('--data_dir', default='xxx', help='Path to Vid4 GT dataset')
    parser.add_argument('--scale', type=float, default=4.0, help='Scale factor for the video')
    
    args = parser.parse_args()

    device = torch.device(args.device)
    model = Flow_STSR().to(device)
    state_dict = torch.load(args.weight_p)
    model.load_state_dict(state_dict)

    vid_test(args.data_dir, args.test_p, args.scale)
