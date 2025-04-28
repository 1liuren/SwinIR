import argparse
import cv2
import numpy as np
import os
import torch
import requests
from collections import OrderedDict
from models.network_swinir import SwinIR as net
from utils import util_calculate_psnr_ssim as util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='real_sr', help='classical_sr, lightweight_sr, real_sr, '
                                                                   'gray_dn, color_dn, jpeg_car, color_jpeg_car')
    parser.add_argument('--scale', type=int, default=4, help='scale factor: 1, 2, 3, 4, 8') # 1 for dn and jpeg car
    parser.add_argument('--noise', type=int, default=15, help='noise level: 15, 25, 50')
    parser.add_argument('--jpeg', type=int, default=40, help='scale factor: 10, 20, 30, 40')
    parser.add_argument('--training_patch_size', type=int, default=128, help='patch size used in training SwinIR')
    parser.add_argument('--large_model', action='store_true', help='use large model, only provided for real image sr')
    parser.add_argument('--model_path', type=str, default='model_zoo/swinir/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth')
    parser.add_argument('--input_path', type=str, required=True, help='input image path')
    parser.add_argument('--output_path', type=str, help='output image path')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置输出路径
    if args.output_path is None:
        output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        args.output_path = os.path.join(output_dir, os.path.basename(args.input_path).split('.')[0] + '_SwinIR.png')

    # 下载或加载模型
    if os.path.exists(args.model_path):
        print(f'loading model from {args.model_path}')
    else:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        url = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{}'.format(os.path.basename(args.model_path))
        r = requests.get(url, allow_redirects=True)
        print(f'downloading model {args.model_path}')
        open(args.model_path, 'wb').write(r.content)

    # 加载模型
    model = define_model(args)
    model.eval()
    model = model.to(device)

    # 读取图像
    img_lq = read_img(args)
    
    # 转换图像格式
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HWC-BGR to CHW-RGB
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(device)  # CHW-RGB to NCHW-RGB

    # 设置window size
    window_size = 7 if args.task in ['jpeg_car', 'color_jpeg_car'] else 8

    # 推理
    with torch.no_grad():
        # pad input image to be a multiple of window_size
        _, _, h_old, w_old = img_lq.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        output = model(img_lq)
        output = output[..., :h_old * args.scale, :w_old * args.scale]

    # 保存图像
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HWC-BGR
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    cv2.imwrite(args.output_path, output)
    print(f'结果已保存到: {args.output_path}')

def read_img(args):
    """读取输入图像并根据任务类型进行预处理"""
    if args.task in ['real_sr', 'classical_sr', 'lightweight_sr']:
        img_lq = cv2.imread(args.input_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    elif args.task in ['gray_dn']:
        img_lq = cv2.imread(args.input_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        img_lq = np.expand_dims(img_lq, axis=2)
    elif args.task in ['color_dn']:
        img_lq = cv2.imread(args.input_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    elif args.task in ['jpeg_car']:
        img_lq = cv2.imread(args.input_path, cv2.IMREAD_UNCHANGED)
        if img_lq.ndim != 2:
            img_lq = util.bgr2ycbcr(img_lq, y_only=True)
        img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
    elif args.task in ['color_jpeg_car']:
        img_lq = cv2.imread(args.input_path)
        img_lq = img_lq.astype(np.float32) / 255.
    return img_lq

def define_model(args):
    """定义模型结构"""
# 001 classical image sr
    if args.task == 'classical_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=args.training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = 'params'

    # 002 lightweight image sr
    # use 'pixelshuffledirect' to save parameters
    elif args.task == 'lightweight_sr':
        model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'

    # 003 real-world image sr
    elif args.task == 'real_sr':
        if not args.large_model:
            # use 'nearest+conv' to avoid block artifacts
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        else:
            # larger model size; use '3conv' to save parameters and memory; use ema for GAN training
            model = net(upscale=args.scale, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
        param_key_g = 'params_ema'

    # 004 grayscale image denoising
    elif args.task == 'gray_dn':
        model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 005 color image denoising
    elif args.task == 'color_dn':
        model = net(upscale=1, in_chans=3, img_size=128, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 grayscale JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'jpeg_car':
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    # 006 color JPEG compression artifact reduction
    # use window_size=7 because JPEG encoding uses 8x8; use img_range=255 because it's sligtly better than 1
    elif args.task == 'color_jpeg_car':
        model = net(upscale=1, in_chans=3, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'

    pretrained_model = torch.load(args.model_path)
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    return model

if __name__ == '__main__':
    main()