from comet_ml import Experiment
import os
import numpy as np
import itertools
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import torch
import params
import utils

# reproducibility
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

cuda = utils.cuda
opt = params.opt

# create folders based on code dimension
os.makedirs("images/static/", exist_ok=True)
for i in range(opt.code_dim):
    name = 'images/varying_c' + str(i+1) + '/'
    os.makedirs(name, exist_ok=True)


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator, discriminator, adversarial_loss, categorical_loss, continuous_loss = utils.init_GAN()
dataloader = utils.get_MNIST_loader()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_info = torch.optim.Adam(
    itertools.chain(generator.parameters(), discriminator.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2)
)

static_z, static_label, static_code = utils.get_static_gen_input()


experiment = Experiment(api_key="plg42bGPkFkyBcCXbg7RC8xys", project_name="bn-infogan", workspace="egebeyazit93")
experiment.log_parameters(vars(opt))
#  Training
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = utils.to_categorical(labels.numpy(), num_columns=opt.n_classes)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = utils.to_categorical(np.random.randint(0, opt.n_classes, batch_size), num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        # Generate a batch of images
        gen_imgs = generator(z, label_input, code_input)

        # Loss measures generator's ability to fool the discriminator
        validity, _, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)
        experiment.log_metric("g_loss", g_loss.item(), step=(epoch + 1) * i)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, _, _ = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid)

        # Loss for fake images
        fake_pred, _, _ = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        experiment.log_metric("d_loss", d_loss.item(), step=(epoch + 1) * i)
        d_loss.backward()
        optimizer_D.step()

        # ------------------
        # Information Loss
        # ------------------

        optimizer_info.zero_grad()

        # Sample labels
        sampled_labels = np.random.randint(0, opt.n_classes, batch_size)

        # Ground truth labels
        gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

        # Sample noise, labels and code as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        label_input = utils.to_categorical(sampled_labels, num_columns=opt.n_classes)
        code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, opt.code_dim))))

        gen_imgs = generator(z, label_input, code_input)
        _, pred_label, pred_code = discriminator(gen_imgs)

        # negative edges
        negative_edges = [
            [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]
        ]
        positive_edges = [ # not implemented yet, but info_loss does it already.
            [0, 0], [1, 1], [2, 2]
        ]
        structure_loss = utils.get_structure_loss(continuous_loss, code_input, pred_code, negative_edges)
        experiment.log_metric("structure_loss", structure_loss.item(), step=(epoch + 1) * i)


        info_loss = params.lambda_cat * categorical_loss(pred_label, gt_labels) + \
                    params.lambda_con * (continuous_loss(pred_code, code_input) -
                                            0.4 * structure_loss)

        experiment.log_metric("info_loss", info_loss.item(), step=(epoch + 1) * i)
        info_loss.backward()
        optimizer_info.step()

        # --------------
        # Log Progress
        # --------------
        if i == len(dataloader) - 1:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
            )
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            utils.sample_image2(generator=generator, n_row=10, batches_done=batches_done)

torch.save({'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'parameters': opt}, './trained_models/model_final_{}'.format(opt.n_epochs))


experiment.log_asset('./trained_models/model_final_{}'.format(opt.n_epochs))
experiment.log_asset_folder('.', step=None, log_file_name=False, recursive=False)
experiment.log_asset_folder('./images', step=None, log_file_name=True, recursive=True)
