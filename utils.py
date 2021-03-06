from typing import List, Any, Union

import torch
from numpy.core._multiarray_umath import ndarray
from torch.autograd import Variable
import numpy as np
import generator as g
import discriminator as d
import params
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
import os


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
opt = params.opt


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(FloatTensor(y_cat))


def init_GAN():
    # models
    generator = g.Generator()
    discriminator = d.Discriminator()
    # Loss functions
    adversarial_loss = torch.nn.MSELoss()
    categorical_loss = torch.nn.CrossEntropyLoss()
    continuous_loss = torch.nn.MSELoss()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        categorical_loss.cuda()
        continuous_loss.cuda()

    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)
    return generator, discriminator, adversarial_loss, categorical_loss, continuous_loss


def get_MNIST_loader():
    # Configure data loader
    os.makedirs("../../data/mnist", exist_ok=True)
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../data/mnist",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(params.opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=params.opt.batch_size,
        shuffle=True,
    )
    return dataloader


def get_static_gen_input():
    # Static generator inputs for sampling
    static_z = Variable(FloatTensor(np.zeros((opt.latent_dim ** 2, opt.latent_dim))))
    static_code = Variable(FloatTensor(np.zeros((opt.latent_dim ** 2, opt.code_dim))))
    return static_z, static_code


def sample_image2(generator, n_row, batches_done):
    static_z, static_code = get_static_gen_input()
    zeros = np.zeros((n_row ** 2, 1))
    c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
    for i in range(opt.code_dim):
        l = [zeros] * opt.code_dim
        l[i] = c_varied
        c = Variable(FloatTensor(np.concatenate(tuple(l), -1)))
        sample = generator(static_z, c)
        name = 'images/varying_c' + str(i + 1) + '/%d.png' % batches_done
        save_image(sample.data, name, nrow=n_row, normalize=True)


def get_structure_loss(loss_function, code_input, pred_code, negative_edges):
    # negative edges are the edges to punish. edge: [source, destination]
    loss = 0
    for edge in negative_edges:
        loss += loss_function(pred_code[:, edge[1]], code_input[:, edge[0]])

    return loss / float(len(negative_edges))

# this needs to further implement positive edges



