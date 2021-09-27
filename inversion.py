import os
from os.path import join
from collections import OrderedDict

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

from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)


_MEAN_STATS = (103.939, 116.779, 123.68)


class VGG16(nn.Sequential):
    # Input : -1~1 , RGB, NCHW,
    """Defines the VGG16 structure as the perceptual network.

    This models takes `RGB` images with pixel range [-1, 1] and data format `NCHW`
    as raw inputs. This following operations will be performed to preprocess the
    inputs (as defined in `keras.applications.imagenet_utils.preprocess_input`):
    (1) Shift pixel range to [0, 255].
    (3) Change channel order to `BGR`.
    (4) Subtract the statistical mean.

    NOTE: The three fully connected layers on top of the model are dropped.
    """

    def __init__(self, output_layer_idx=23, min_val=-1.0, max_val=1.0):
        """Defines the network structure.

        Args:
          output_layer_idx: Index of layer whose output will be used as perceptual
            feature. (default: 23, which is the `block4_conv3` layer activated by
            `ReLU` function)
          min_val: Minimum value of the raw input. (default: -1.0)
          max_val: Maximum value of the raw input. (default: 1.0)
        """
        sequence = OrderedDict({
            # block 1
            'layer0': nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            'layer1': nn.ReLU(inplace=True),
            'layer2': nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            'layer3': nn.ReLU(inplace=True),
            'layer4': nn.MaxPool2d(kernel_size=2, stride=2),
            # block 2
            'layer5': nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            'layer6': nn.ReLU(inplace=True),
            'layer7': nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            'layer8': nn.ReLU(inplace=True),
            'layer9': nn.MaxPool2d(kernel_size=2, stride=2),
            # block3
            'layer10': nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            'layer11': nn.ReLU(inplace=True),
            'layer12': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            'layer13': nn.ReLU(inplace=True),
            'layer14': nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            'layer15': nn.ReLU(inplace=True),
            'layer16': nn.MaxPool2d(kernel_size=2, stride=2),
            # block4
            'layer17': nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            'layer18': nn.ReLU(inplace=True),
            'layer19': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer20': nn.ReLU(inplace=True),
            'layer21': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer22': nn.ReLU(inplace=True),
            #
            'layer23': nn.MaxPool2d(kernel_size=2, stride=2),
            'layer24': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer25': nn.ReLU(inplace=True),
            'layer26': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer27': nn.ReLU(inplace=True),
            'layer28': nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            'layer29': nn.ReLU(inplace=True),
            'layer30': nn.MaxPool2d(kernel_size=2, stride=2),
        })
        self.output_layer_idx = output_layer_idx
        self.min_val = min_val
        self.max_val = max_val
        self.mean = torch.from_numpy(np.array(_MEAN_STATS)).view(1, 3, 1, 1)
        self.mean = self.mean.type(torch.FloatTensor)
        super().__init__(sequence)

    def forward(self, x):
        x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
        x = x[:, [2, 1, 0], :, :]
        x = x - self.mean.to(x.device)
        for i in range(self.output_layer_idx):
            x = self.__getattr__(f'layer{i}')(x)
        return x

    def forward_network(self, x):
        features = []
        # change range to 0~255.
        x = (x - self.min_val) * 255.0 / (self.max_val - self.min_val)
        x = x[:, [2, 1, 0], :, :]  # change to BGR
        x = x - self.mean.to(x.device)
        for i in range(self.output_layer_idx):
            x = self.__getattr__(f'layer{i}')(x)
            if f'layer{i}' == 'layer1':
                features.append(x)  # conv1_1
            elif f'layer{i}' == 'layer3':
                features.append(x)  # conv1_2
            elif f'layer{i}' == 'layer13':
                features.append(x)  # conv3_2
            elif f'layer{i}' == 'layer20':
                features.append(x)  # conv4_2
        return features


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
    parser.add_argument("--n_frame", type=int, default=120)
    parser.add_argument("--radius", type=float, default=30)
    parser.add_argument(
        "--ckpt", metavar="CKPT", type=str, help="path to the model checkpoint",
        default='./checkpoint/140000.pt'
    )

    # stylegan
    parser.add_argument('--model_name', type=str,
                        help='Name to the pre-trained model.', default='stylegan2_church256')
    parser.add_argument('--generate_html', type=bool_parser, default=False,
                        help='Whether to use HTML page to visualize the '
                             'synthesized results. (default: %(default)s)')
    parser.add_argument('--save_raw_synthesis', type=bool_parser, default=True,
                        help='Whether to save raw synthesis. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')

    # IO
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num', type=int, default=100,
                        help='Number of samples to synthesize. ''(default: %(default)s)')
    parser.add_argument('--save_dir', type=str, default='reproduce_image2stylegan',
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/inversion/` by default. '
                             '(default: %(default)s)')
    parser.add_argument('--job_name', type=str, default='TT')
    parser.add_argument('--target_images_path', type=str, default='./srcs/sample03.jpg',
                        help='target image folder path, if black it means random image generated by GAN')
    parser.add_argument('--path_inversion_result', type=str, default='./inversion',
                        help='target image folder path, if black it means random image generated by GAN')
    # Settings
    parser.add_argument('--num_iters', type=int, default=5000)
    parser.add_argument('--mse_weight', type=float, default=1.)
    parser.add_argument('--perceptual_weight', type=float, default=1.)
    # parser.add_argument('--inter_feature_weight', type=float, default=1.)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='Adam or RAdam')

    return parser.parse_args()


def load_perceptual_model():
    vgg_weight_path = 'VGG16.pth'
    perceptual_model = VGG16()
    perceptual_model.load_state_dict(torch.load(vgg_weight_path))
    perceptual_model.eval().cuda()
    perceptual_model.requires_grad_(False)

    return perceptual_model


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

    # Load models
    perceptual_model = load_perceptual_model()

    ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
    conf = GANConfig(**ckpt["conf"])
    generator = conf.generator.make().to(DEVICE)
    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    # Get initial latent
    mean_latent = generator.mean_latent(args.truncation_mean)

    z = torch.randn(args.n_img, conf.generator["style_dim"], device=DEVICE)

    transform_p = generator.get_transform(
        z, truncation=args.truncation, truncation_latent=mean_latent
    )

    transform_p[:, 0] = 0
    transform_p[:, 1] = -1
    transform_p[:, 2:] = 0

    # Load inversion target
    # with torch.no_grad():
    target = Image.open(args.target_images_path)
    target.save(join(args.path_inversion_result, 'target.jpg'))
    target = PILToTensor()(target).unsqueeze(0).to(DEVICE)

    # Optimization setting 
    optimizing_variable = []

    if True: #w
        with torch.no_grad():
            w = generator.z2w(
                z,
                truncation=args.truncation,
                truncation_latent=mean_latent,
            )
        w.requires_grad=True
        optimizing_variable.append(w)
    elif False: #wp
        pass
    elif False: # 
        pass
    elif False: #z
        pass

    optimizer = torch.optim.Adam(optimizing_variable, lr=args.lr)

    # Do inversion
    losses = []
    mse_losses = []
    lpips_losses = []

    for i in tqdm(range(args.num_iters)):
        loss = 0.

        x_rec = generator.forward_w(
            w,
            truncation=args.truncation,
            truncation_latent=mean_latent,
            transform=transform_p,
        )

        # x_rec = (x_rec + 1) / 2
        # x_rec = x_rec.clip(0, 1)
        x_rec = x_rec - x_rec.min()
        x_rec = x_rec / x_rec.max()

        # mse
        mse_loss = torch.mean((x_rec-target)**2)
        mse_losses.append(mse_loss.item())
        loss += mse_loss * args.mse_weight

        # lpips
        target_features = perceptual_model.forward_network(target)
        x_rec_features = perceptual_model.forward_network(x_rec)

        lpips_loss = 0.
        for feature_idx in range(4):
            lpips_loss += torch.mean(
                (target_features[feature_idx]-x_rec_features[feature_idx])**2)

        lpips_losses.append(lpips_loss.item())
        loss += lpips_loss * args.perceptual_weight

        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % 10 == 0:
            im_inv = ToPILImage()(x_rec.squeeze(0))
            im_inv.save(join(args.path_inversion_result,
                'inversion_%06d.jpg' % i))


    im_inv = ToPILImage()(x_rec.squeeze(0))
    im_inv.save('inversion_last.jpg')

    import matplotlib.pyplot as plt
    plt.clf()
    plt.plot(losses, color='black', label=f'losses_lr{args.lr}')
    plt.plot(mse_losses, color='red', label=f'mse_losses')
    plt.plot(lpips_losses, color='blue', label=f'lpips_losses')
    plt.legend()
    plt.savefig('loss_graph.png')


if __name__ == '__main__':
    main()
