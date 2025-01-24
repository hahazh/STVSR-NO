import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import argparse
import cv2
import shutil
import torch
import torch.nn.functional as F
from data.SPMCS import SPMCS_arb
from torch.utils.data import DataLoader, Dataset
from model.STVSR_OP_arch import Flow_STSR

def single_forward(model, imgs_in, time_ls, target_size, scale):
    out_ls = []
    def calculate_padding(height, width):
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        return (0, pad_w, 0, pad_h)

    with torch.no_grad():
        b, n, c, h, w = imgs_in.size()
        padding = calculate_padding(h, w)
        pad_h, pad_w = padding[3], padding[1]
        reshaped_tensor = imgs_in.view(-1, h, w)
        padded_reshaped_tensor = F.pad(reshaped_tensor, padding, mode='reflect')
        
        padded_tensor = padded_reshaped_tensor.view(b, n, c, h + pad_h, w + pad_w)

        bordered_size = (round(scale * (h + pad_h)), round(scale * (w + pad_w)))

        multi_feat = model.muti_feat_extract(padded_tensor, time_ls)
        features_cpu = multi_feat.cpu()
        for frame_feat in features_cpu:
            frame_feat = frame_feat.unsqueeze(0)
            frame_feat = frame_feat.to(device)  

            output = model.query_rgb(frame_feat, bordered_size)
            output = output.view(-1, 3, bordered_size[0], bordered_size[1]).squeeze(0)
            output = output[:, :target_size[0], :target_size[1]]
            out_ls.append(output.detach().cpu())
    return out_ls

def SPMCS_test(data_dir, output_p):
    in_bewteen_scale_list = [2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
    out_bewteen_scale_list = [8.0, 7.0, 6.0, 5.0]
    scale_list = out_bewteen_scale_list + in_bewteen_scale_list
    scale_list = [2.0]
    time_list = [2, 4]
    input_two = False
    for scale in scale_list:
        for tempo in time_list:
            if input_two:
                modulate_factor = 'two_' + (str(scale) + '_' + str(tempo)).replace('.', 'p')
            else:
                modulate_factor = 'mul_' + (str(scale) + '_' + str(tempo)).replace('.', 'p')
            print(modulate_factor)

            out_p = output_p + modulate_factor

            vid_data = SPMCS_arb(data_dir, scale, tempo)
            Vid_ST_dl = DataLoader(vid_data, batch_size=1, shuffle=False)
            time_ls = [int(j) / tempo for j in range(1, tempo)]

            for ix, (GT, LR, name, crop_shape, output_list) in enumerate(Vid_ST_dl):  
                print(output_list)
                target_size = (GT.shape[-2], GT.shape[-1])
                seq_name = name[0]
                this_p = out_p + '/' + seq_name
                if not os.path.exists(this_p):
                    os.makedirs(this_p)
                LR = LR.to(device)
                for idx, ti in enumerate(time_ls):
                    out = single_forward(model, LR, ti, target_size, scale)
                    for i in range(len(out)):
                        img = out[i].permute(1, 2, 0).numpy() * 255.0
                        img = img[:, :, ::-1]
                        if i % 2 == 0:
                            cv2.imwrite(this_p + '/' + output_list[i // 2][0].split('/')[-1], img)
                            print(output_list[i // 2][0].split('/')[-1])
                        else:
                            base = output_list[i // 2][0].split('/')[-1].split('.')[0]
                            this_idx = int(base) + idx + 1
                            this_name = this_p + '/' + str(this_idx).zfill(len(base)) + '.png'
                            print(this_name)
                            cv2.imwrite(this_name, img)

if __name__ == '__main__':
    # python test.py --device cuda --test_p /path/to/test/output --weight_p /path/to/weights --data_dir /path/to/data --output_p /path/to/output

    parser = argparse.ArgumentParser(description="STVSR_OP testing")
    parser.add_argument('--device', default='cuda', help='Device to use (e.g. cuda, cpu)')
    parser.add_argument('--test_p', default='xxx', help='Path to test output')
    parser.add_argument('--weight_p', default='xxx', help='Path to model weights')
    parser.add_argument('--data_dir', default='xxx', help='Path to dataset')
    parser.add_argument('--output_p', default='xxx', help='Path to output folder')

    args = parser.parse_args()

    device = torch.device(args.device)
    model = Flow_STSR().to(device)
    state_dict = torch.load(args.weight_p)
    model.load_state_dict(state_dict)

    SPMCS_test(args.data_dir, args.output_p)
