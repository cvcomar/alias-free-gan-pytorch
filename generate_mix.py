import argparse
import os
import random

import torch
from torchvision import utils
import torchvision.transforms as transforms
import cv2
from tqdm import tqdm
import numpy as np

from model import Generator
from config import GANConfig

if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--seed", type=int, default=2, help="fix random seed"
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
    parser.add_argument("--n_frame", type=int, default=120)
    parser.add_argument("--radius", type=float, default=30)
    parser.add_argument(
        "--ckpt", metavar="CKPT", type=str, default='./checkpoint/140000.pt',
        help="path to the model checkpoint"
    )

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    conf = GANConfig(**ckpt["conf"])
    generator = conf.generator.make().to(device)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    mean_latent = generator.mean_latent(args.truncation_mean)

    if args.seed >= 0:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)

    num_wp = 16

    x = torch.randn(2, conf.generator["style_dim"],
            device=device)

    img = generator(x,
            truncation=args.truncation,
            truncation_latent=mean_latent,
            )
    wp = generator.z2w(x)

    w1 = wp[0].expand(num_wp // 2, -1)
    w2 = wp[1].expand(num_wp // 2, -1)
    wps = torch.cat([w1, w2], dim=0).unsqueeze(0)

    img_mix = generator.forward_wp(wps)
    img = torch.cat([img, img_mix], dim=0)

    grid = utils.make_grid(
        img.cpu(), normalize=True, nrow=3, range=(-1, 1)
    )

    im = transforms.ToPILImage()(grid)
    im.show()
