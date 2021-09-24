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

    theta = np.radians(np.linspace(-90, 270, args.n_frame))
    rotate_c = np.cos(theta)
    rotate_s = np.sin(theta)

    rotate_c = rotate_c.tolist()
    rotate_s = rotate_s.tolist()

    images = []

    transform_p = generator.get_transform(
        x, truncation=args.truncation, truncation_latent=mean_latent
    )

    with torch.no_grad():
        for i, (r_c, r_s) in enumerate(tqdm(zip(rotate_c, rotate_s), total=args.n_frame)):
            transform_p[:, 0] = r_c
            transform_p[:, 1] = r_s
            transform_p[:, 2] = 0
            transform_p[:, 3] = 0

            img = generator(
                x,
                truncation=args.truncation,
                truncation_latent=mean_latent,
                transform=transform_p,
            )
            images.append(
                utils.make_grid(
                    img.cpu(), normalize=True, nrow=args.n_row, range=(-1, 1)
                )
                .mul(255)
                .permute(1, 2, 0)
                .numpy()
                .astype(np.uint8)
            )

    videodims = (images[0].shape[1], images[0].shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"VP90")
    video = cv2.VideoWriter("sample_rotation.webm", fourcc, 24, videodims)

    for i in tqdm(images):
        video.write(cv2.cvtColor(i, cv2.COLOR_RGB2BGR))

    video.release()
