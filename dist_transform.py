import os
import matplotlib.pyplot as plt
from os.path import join
from collections import OrderedDict

import cv2
import random
import numpy as np
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import PILToTensor, ToPILImage

from model import Generator
from config import GANConfig

import PIL
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description="Alias-free GAN Inversion")

    # Alias-free GAN parameter
    parser.add_argument(
        "--seed", type=int, default=1, help="fix random seed"
    )
    parser.add_argument(
        "--n_img", type=int, default=32, help="number of images to be generated"
    )
    parser.add_argument(
        "--iter", type=int, default=100, help="number of iteration" 
    )
    parser.add_argument(
        "--truncation", type=float, default=0.5, help="truncation ratio"
    )
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt", metavar="CKPT", type=str, help="path to the model checkpoint",
        default='./checkpoint/310000.pt')

    parser.add_argument('--gpu', type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Set random seed.
    if args.seed > -1:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    conf = GANConfig(**ckpt["conf"])
    generator = conf.generator.make().to(DEVICE)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    eps = 1e-8

    # Get initial latent
    degrees = []
    t_xs = []
    t_ys = []
    with torch.no_grad():
        for _ in range(args.iter):
            mean_latent = generator.mean_latent(args.truncation_mean)

            z = torch.randn(args.n_img, conf.generator["style_dim"], device=DEVICE)

            affine = generator.get_transform(
                z, truncation=args.truncation, truncation_latent=mean_latent
            )
            norm = torch.norm(affine[:, :2], dim=-1, keepdim=True)
            affine = affine / (norm + eps)

            r_c = affine[:, 0]
            r_s = affine[:, 1]
            t_x = affine[:, 2]
            t_y = affine[:, 3]

            degree = torch.rad2deg(torch.atan2(r_s, r_c)) + 90
            degrees.append(degree.cpu().numpy())
            t_xs.append(t_x.cpu().numpy())
            t_ys.append(t_y.cpu().numpy())

    degrees = np.concatenate(degrees)
    t_xs = np.concatenate(t_xs)
    t_ys = np.concatenate(t_ys)

    num_bin = 100
    fig, axs = plt.subplots(1, 3, figsize=(8,3))

    axs[0].hist(degrees, bins=num_bin)
    axs[0].set_title('rotation')
    axs[0].set_xlabel('degree')

    axs[1].hist(t_xs, bins=num_bin)
    axs[1].set_title('x translation')

    axs[2].hist(t_ys, bins=num_bin)
    axs[2].set_title('y translation')
    plt.suptitle('Transformation Parameter Distribution (Sample: %dEA)' 
            % (args.iter * args.n_img))
    plt.show()

if __name__ == '__main__':
    main()
