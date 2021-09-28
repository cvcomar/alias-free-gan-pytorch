import os
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


_MEAN_STATS = (103.939, 116.779, 123.68)

def bool_parser(arg):
    """Parses an argument to boolean."""
    if isinstance(arg, bool):
        return arg
    if arg.lower() in ['1', 'true', 't', 'yes', 'y']:
        return True
    if arg.lower() in ['0', 'false', 'f', 'no', 'n']:
        return False
    raise argparse.ArgumentTypeError(
        f'`{arg}` cannot be converted to boolean!')


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description="Alias-free GAN Inversion")

    # Alias-free GAN parameter
    parser.add_argument(
        "--seed", type=int, default=1, help="fix random seed"
    )
    parser.add_argument(
        "--n_img", type=int, default=1, help="number of images to be generated"
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
        default='./checkpoint/310000.pt'
    )

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

    # Get initial latent
    with torch.no_grad():
        mean_latent = generator.mean_latent(args.truncation_mean)

        z = torch.randn(args.n_img, conf.generator["style_dim"], device=DEVICE)

        transform_p = generator.get_transform(
            z, truncation=args.truncation, truncation_latent=mean_latent
        )

    import matplotlib.pyplot as plt
    plt.legend()
    plt.savefig(join(args.path_inversion_result, 'loss_graph.png'))


if __name__ == '__main__':
    main()
