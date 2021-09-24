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
        "--seed", type=int, default=1, help="fix random seed"
    )

    parser.add_argument(
        "--n_img", type=int, default=8, help="number of images to be generated"
    )
    parser.add_argument(
        "--n_row", type=int, default=4, help="number of samples per row"
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
        "ckpt", metavar="CKPT", type=str, help="path to the model checkpoint"
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

    x = torch.randn(args.n_img, conf.generator["style_dim"], device=device)

    theta = np.radians(np.linspace(0, 360, args.n_frame))
    x_2 = np.cos(theta) * args.radius
    y_2 = np.sin(theta) * args.radius

    trans_x = x_2.tolist()
    trans_y = y_2.tolist()

    images = []

    transform_p = generator.get_transform(
        x, truncation=args.truncation, truncation_latent=mean_latent
    )

    transform_p[:, 0] = 0
    transform_p[:, 1] = -1
    transform_p[:, 2:] = 0

    img = generator(
        x,
        truncation=args.truncation,
        truncation_latent=mean_latent,
        transform=transform_p,
    )

    grid = utils.make_grid(
        img.cpu(), normalize=True, nrow=args.n_row, range=(-1, 1)
    )

    im = transforms.ToPILImage()(grid)
    im.save('./sample_original.jpg')
