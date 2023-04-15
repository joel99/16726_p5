# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe), modified by Zhiqiu Lin (zl279@cornell.edu)
# --------------------------------------------------------
from __future__ import print_function

import argparse
import os
import os.path as osp
import numpy as np

from LBFGS import FullBatchLBFGS

import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import torchvision.utils as vutils
from torchvision.models import vgg19

import pytorch_lightning as pl
import wandb


from dataloader import get_data_loader

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def build_model(name):
    if name.startswith('vanilla'):
        z_dim = 100
        model_path = 'pretrained/%s.ckpt' % name
        pretrain = torch.load(model_path)
        from vanilla.models import DCGenerator
        model = DCGenerator(z_dim, 32, 'instance')
        model.load_state_dict(pretrain)

    elif name == 'stylegan':
        model_path = 'pretrained/%s.ckpt' % name
        import sys
        sys.path.insert(0, 'stylegan')
        from stylegan import dnnlib, legacy
        with dnnlib.util.open_url(model_path) as f:
            model = legacy.load_network_pkl(f)['G_ema']
            z_dim = model.z_dim
    else:
         return NotImplementedError('model [%s] is not implemented', name)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, z_dim


class Wrapper(nn.Module):
    """The wrapper helps to abstract stylegan / vanilla GAN, z / w latent"""
    def __init__(self, args, model, z_dim):
        super().__init__()
        self.model, self.z_dim = model, z_dim
        self.latent = args.latent
        self.is_style = args.model == 'stylegan'

    def forward(self, param):
        if self.latent == 'z':
            if self.is_style:
                image = self.model(param, None)
            else:
                image = self.model(param)
        # w / wp
        else:
            assert self.is_style
            if self.latent == 'w':
                param = param.repeat(1, self.model.mapping.num_ws, 1)
            image = self.model.synthesis(param)
        return image


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


class PerceptualLoss(nn.Module):
    def __init__(self, add_layer=['conv_5']):
        device = get_device()
        super().__init__()
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.norm = Normalization(cnn_normalization_mean, cnn_normalization_std)
        cnn = vgg19(pretrained=True).features.to(device).eval()

        conv_num = 0
        last_layer_end = 0
        pieces = []
        assert len(add_layer) > 0, 'add_layer should not be empty'
        add_layer = sorted(add_layer)
        for i in range(len(cnn)):
            if isinstance(cnn[i], nn.Conv2d):
                if 'conv_%d' % conv_num in add_layer:
                    pieces.append(cnn[last_layer_end:i+1])
                    last_layer_end = i+1
                conv_num += 1
            if len(pieces) == len(add_layer):
                break
        self.model = nn.ModuleList(pieces)

    def forward(self, pred, target):

        if isinstance(target, tuple):
            target, mask = target
        else:
            mask = None
        pred = self.norm(pred)
        target = self.norm(target)

        loss = 0.
        for net in self.model:
            pred = net(pred)
            target = net(target)
            if mask is not None:
                resize_mask = F.adaptive_avg_pool2d(mask, pred.shape[2:])
            cur_loss = F.mse_loss(pred, target, reduction='none')
            if mask is not None:
                cur_loss = cur_loss * resize_mask
                # cur_loss = cur_loss[resize_mask]
            loss = loss + cur_loss.mean()
        return loss

class Criterion(nn.Module):
    def __init__(self, args, mask=False, layer=['conv_5']):
        super().__init__()
        self.perc_wgt = args.perc_wgt
        self.l1_wgt = args.l1_wgt # weight for l1 loss/mask loss
        self.mask = mask

        self.perc = PerceptualLoss(layer)

    def forward(self, pred, target):
        """Calculate loss of prediction and target. in p-norm / perceptual  space"""
        perc_loss = self.perc(pred, target)
        if self.mask:
            target, mask = target
        else:
            mask = None
        l1_loss = F.l1_loss(pred, target, reduction='none')
        if mask is not None:
            l1_loss = l1_loss * mask
        l1_loss = l1_loss.mean()
        return self.perc_wgt * perc_loss + self.l1_wgt * l1_loss, self.perc_wgt * perc_loss, self.l1_wgt * l1_loss


def save_images(image, fname, col=8, step=0, save_wandb=True, caption=""):
    image = image.cpu().detach()
    image = image / 2 + 0.5

    image = vutils.make_grid(image, nrow=col)  # (C, H, W)
    image = image.numpy().transpose([1, 2, 0])
    image = np.clip(255 * image, 0, 255).astype(np.uint8)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        imageio.imwrite(fname + '.png', image)
    if save_wandb:
        if caption == "":
            caption = fname
        wandb.log({'samples': [wandb.Image(image, caption=caption)]}, step=step)
    return image


def save_gifs(image_list, fname, col=1):
    """
    :param image_list: [(N, C, H, W), ] in scale [-1, 1]
    """
    image_list = [save_images(each, None, col) for each in image_list]
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    imageio.mimsave(fname + '.gif', image_list)


def sample_noise(dim, device, latent, model, N=1, from_mean=False):
    """
    To generate a noise vector, just sample from a normal distribution.
    To generate a style latent, you need to map the noise (z) to the style (W) space given the `model`.
    You will be using model.mapping for this function.
    Specifically,
    if from_mean=False,
        sample N noise vector (z) or N style latent(w/w+) depending on latent value.
    if from_mean=True
        if latent == 'z': Return zero vectors since zero is the mean for standard gaussian
        if latent == 'w'/'w+': You should sample N=10000 z to generate w/w+ and then take the mean.
    Some hint on the z-mapping can be found at stylegan/generate_gif.py L70:81.
    Additionally, you can look at stylegan/training/networks.py class Generator L477:500
    :return: Tensor on device in shape of (N, dim) if latent == z
             Tensor on device in shape of (N, 1, dim) if latent == w
             Tensor on device in shape of (N, nw, dim) if latent == w+
    """
    def sample_w(N, draws=1, w_plus=False):
        z = torch.randn(N * draws, dim, device=device) # out = N nw dim
        w: torch.Tensor = model.mapping(z, None)
        w = w.view(N, draws, w.size(1), w.size(2))
        if not w_plus:
            w = w[:, :, :1]
        return w.mean(dim=1) # N D Ws Dim -> N Ws Dim
    if latent == 'z':
        vector = torch.randn(N, dim, device=device) if not from_mean else torch.zeros(N, dim, device=device)
    elif latent in ['w', 'w+']:
        if from_mean:
            vector = sample_w(N, 10000, w_plus=latent == 'w+')
        else:
            vector = sample_w(N, w_plus=latent == 'w+')
    else:
        raise NotImplementedError('%s is not supported' % latent)
    return vector


def optimize_para(
    wrapper,
    param,
    target,
    criterion,
    num_step,
    save_prefix=None,
    res=False,
    l2_weight=0.,
    caption_prefix="",
):
    """
    wrapper: image = wrapper(z / w/ w+): an interface for a generator forward pass.
    param: z / w / w+
    target: (1, C, H, W)
    criterion: loss(pred, target)
    """
    device = get_device()
    delta = torch.zeros_like(param)
    delta = delta.requires_grad_().to(device)
    optimizer = FullBatchLBFGS([delta], lr=.1, line_search='Wolfe')
    iter_count = [0]
    def closure():
        optimizer.zero_grad()

        image = wrapper(param + delta)
        loss, perc_loss, l1_loss = criterion(image, target)
        if l2_weight:
            l2_loss = l2_weight * torch.sum(delta ** 2)
            loss += l2_loss
        else:
            l2_loss = 0

        iter_count[0] += 1
        if iter_count[0] % 500 == 0:
            # visualization code
            log_dict = {'loss': loss.item()}
            if l2_weight:
                log_dict['l2_loss'] = l2_loss.item()
            if l1_loss:
                log_dict['l1_loss'] = l1_loss.item()
            if perc_loss:
                log_dict['perc_loss'] = perc_loss.item()
            wandb.log(log_dict, step=iter_count[0])
            print('iter count {} loss {:4f}'.format(iter_count, loss.item()))
            if save_prefix is not None:
                iter_result = image.data.clamp_(-1, 1)
                uses_losses = []
                if l1_loss:
                    uses_losses.append('L1')
                if perc_loss:
                    uses_losses.append('Perc')
                if l2_weight:
                    uses_losses.append('L2 Norm')
                caption = f"{'+'.join(uses_losses)}: "
                if caption_prefix:
                    caption = caption_prefix + ': ' + caption
                save_images(
                    iter_result, save_prefix + '_%d' % iter_count[0], step=iter_count[0],
                    caption=caption
                )
        return loss

    loss = closure()
    loss.backward()
    while iter_count[0] <= num_step:
        options = {'closure': closure, 'max_ls': 10}
        loss, _, lr, _, F_eval, G_eval, _, _ = optimizer.step(options)
    image = wrapper(param + delta)
    image.data.clamp_(-1, 1)
    return param + delta, image


def sample(args):
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    batch_size = 16
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    noise = sample_noise(z_dim, device, args.latent, model, batch_size)
    image = wrapper(noise)
    fname = os.path.join('output/forward/%s_%s' % (args.model, args.mode))
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    save_images(image, fname)


def project(args):
    # load images
    device = get_device()
    loader = get_data_loader(args.input, args.resolution, is_train=False)

    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)
    print('model {} loaded'.format(args.model))
    criterion = Criterion(args)
    # project each image
    for idx, (data, _) in enumerate(loader):
        target = data.to(device)
        save_images(data, 'output/project/%d_data' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, model)
        caption_prefix = f"{args.latent}"
        latents, out = optimize_para(wrapper, param, target, criterion, args.n_iters,
                      'output/project/%d_%s_%s_perc-%g_l1-%g-l2-%g' % (idx, args.model, args.latent, args.perc_wgt, args.l1_wgt, args.l2_wgt), l2_weight=args.l2_wgt, caption_prefix=caption_prefix)
        if idx >= 0:
            break


def draw(args):
    device = get_device()
    # define and load the pre-trained model
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution, alpha=True)
    criterion = Criterion(args, True)
    for idx, (rgb, mask) in enumerate(loader):
        rgb, mask = rgb.to(device), mask.to(device)
        save_images(rgb, 'output/draw/%d_data' % idx, 1)
        save_images(mask, 'output/draw/%d_mask' % idx, 1)
        param = sample_noise(z_dim, device, args.latent, model, from_mean=True)
        latents, out = optimize_para(wrapper, param, (rgb, mask), criterion, args.n_iters,
            'output/draw/%d_%s_%s_perc-%g_l1-%g-l2-%g' % (idx, args.model, args.latent, args.perc_wgt, args.l1_wgt, args.l2_wgt),
            l2_weight=args.l2_wgt
        )
        # if idx >= 0:
            # break


def interpolate(args):
    device = get_device()
    model, z_dim = build_model(args.model)
    wrapper = Wrapper(args, model, z_dim)

    # load the target and mask
    loader = get_data_loader(args.input, args.resolution)
    criterion = Criterion(args)
    for idx, (image, _) in enumerate(loader):
        save_images(image, 'output/interpolate/%d' % (idx))
        target = image.to(device)
        param = sample_noise(z_dim, device, args.latent, model)
        param, recon = optimize_para(wrapper, param, target, criterion, args.n_iters, l2_weight=args.l2_wgt)
        save_images(recon, 'output/interpolate/%d_%s_%s_%g' % (idx, args.model, args.latent, args.l2_wgt))
        if idx % 2 == 0:
            src = param
            continue
        dst = param
        alpha_list = np.linspace(0, 1, 50)
        image_list = []
        with torch.no_grad():
            for alpha in alpha_list:
                image_list.append(wrapper(alpha * src + (1 - alpha) * dst))
        save_gifs(image_list, 'output/interpolate/%d_%s_%s_%g' % (idx, args.model, args.latent, args.l2_wgt))
        if idx >= 3:
            break
    return


def parse_arg():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='stylegan', choices=['vanilla', 'stylegan'])
    parser.add_argument('--mode', type=str, default='sample', choices=['sample', 'project', 'draw', 'interpolate'])
    parser.add_argument('--latent', type=str, default='z', choices=['z', 'w', 'w+'])
    parser.add_argument('--n_iters', type=int, default=8000, help="number of optimization steps in the image projection")
    parser.add_argument('--perc_wgt', type=float, default=0.01, help="perc loss weight")
    parser.add_argument('--l1_wgt', type=float, default=10., help="L1 pixel loss weight")
    parser.add_argument('--l2_wgt', type=float, default=0.0, help="L2 delta norm weight")
    parser.add_argument('--resolution', type=int, default=64, help='Resolution of images')
    parser.add_argument('--input', type=str, default='data/sketch/*.png', help="path to the input image") # Quick hack, easier to replace default than figure out how to launch correctly through slurm...
    # parser.add_argument('--input', type=str, default='data/cat/*.png', help="path to the input image")
    return parser.parse_args()


if __name__ == '__main__':
    # pl.seed_everything(0)
    args = parse_arg()
    wandb.init(
        project='16726_p5',
        config=args,
        name=f'{args.model}-{args.mode}-{args.latent}-perc_{args.perc_wgt}-l1_{args.l1_wgt}-l2_{args.l2_wgt}',
    )
    if args.mode == 'sample':
        sample(args)
    elif args.mode == 'project':
        project(args)
    elif args.mode == 'draw':
        draw(args)
    elif args.mode == 'interpolate':
        interpolate(args)