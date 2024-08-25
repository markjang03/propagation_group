import argparse
import os
import numpy as np

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=2e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=4, help="dimensionality of the latent space")
parser.add_argument("--conditional_info_dim", type=int, default=3005, help="number of classes for dataset")
opt = parser.parse_args()
print(opt)
os.makedirs("images_" + str(opt), exist_ok=True)

cuda = True if torch.cuda.is_available() else False

def LOSNLOS(tx_total_height, rx_total_height, d_array, e_array):
    LOS = 1
    nonzero_e_array_indices = [e_array != 0]
    d_array_nonzero = d_array[tuple(nonzero_e_array_indices)]
    e_array_nonzero = e_array[tuple(nonzero_e_array_indices)]
    slope = (rx_total_height - tx_total_height) / max(d_array_nonzero)
    for i in range(len(e_array_nonzero) - 1):
        # if there is a blockage, LOS = 0, (NLOS = 1)
        if (d_array_nonzero[i] * slope + tx_total_height) < (e_array_nonzero[i]):
            LOS = 0
    return LOS

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(2 + opt.latent_dim, 8, normalize=False),
            nn.Linear(8, 1)
        )

        self.model2 = nn.Sequential(
            *block(3001, 8),
            *block(8, 1),
            nn.Sigmoid()
        )

        self.model3 = nn.Sequential(
            nn.Linear(2, 1)
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        dist = torch.sqrt(
            torch.abs(labels[:, 2] - labels[:, 0]) ** 2 + torch.abs(labels[:, 3] - labels[:, 1]) ** 2).unsqueeze(1)
        elevation_info = labels[:, 4:]
        elevation = self.model2(elevation_info)
        gen_input = torch.cat([dist, elevation, noise], -1)
        power = self.model(gen_input)
        #power = self.model3(torch.cat([distance, elevation], -1))
        return power


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.model2 = nn.Sequential(
            nn.Linear(3001, 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

        self.model3 = nn.Sequential(
            nn.Linear(2, 1),
        )

    def forward(self, power, conditional_info):
        # Concatenate label embedding and image to produce input
        dist = torch.sqrt(torch.abs(conditional_info[:, 2] - conditional_info[:, 0]) ** 2 + torch.abs(
            conditional_info[:, 3] - conditional_info[:, 1]) ** 2).unsqueeze(1)
        elevation = self.model2(conditional_info[:, 4:])
        validity = self.model(torch.cat([power, dist, elevation], -1))
        #d_in = torch.cat((power, d_info), -1)
        #validity = self.model(d_in)
        return validity


# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Configure data loader
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if cuda:
            self.x_data = torch.tensor(X, dtype=torch.float32).to("cuda")
            self.y_data = torch.tensor(y, dtype=torch.float32).to("cuda")
        else:
            self.x_data = torch.tensor(X, dtype=torch.float32).to("cpu")
            self.y_data = torch.tensor(y, dtype=torch.float32).to("cpu")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx, :]  # or just [idx]
        out_y = self.y_data[idx, :]
        return (preds, out_y)  # tuple of two matrices


train_X = torch.load('.././train_X_newww.pt', map_location=(torch.device("cuda") if cuda else torch.device("cpu")))
train_y = torch.load('.././train_y_neww.pt', map_location=(torch.device("cuda") if cuda else torch.device("cpu")))
custom_dataset = custom_dataset(train_X, train_y)
custom_dataloader = torch.utils.data.DataLoader(custom_dataset, batch_size=opt.batch_size, shuffle=False)


def find_dist(l, cellsize):
    a = np.linalg.norm((l[2] - l[0]) * cellsize) ** 2 + np.linalg.norm((l[3] - l[1]) * cellsize) ** 2
    return np.sqrt(a)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=3 * opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr / 8, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor  # Float
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor  # Integer

# ----------
#  Training
# ----------
d_loss_list = []
g_loss_list = []
image_counter = 0

max_x_coord = 10165
max_y_coord = 12545
max_elev = 1555.86
min_elev = 1420.92

d_real_list = []
gan_real_rss_list = []
los_indices = np.where(train_X[:, 0] == 1)[0]
nlos_indices = np.where(train_X[:, 0] == 0)[0]
for i, (x, y) in enumerate(zip(train_X[:, 1:5], train_y)):
    fake_info_reconst = x
    fake_info_reconst[0] *= max_x_coord
    fake_info_reconst[1] *= max_y_coord
    fake_info_reconst[2] *= max_x_coord
    fake_info_reconst[3] *= max_y_coord

    d = find_dist(fake_info_reconst, 0.5)

    if d < 1000:
        print(d)
        d_real_list.append(d)
        gan_real_rss_list.append(y)

d_real_list = np.array(d_real_list)
print(d_real_list / max(d_real_list))
gan_real_rss_list = np.array(gan_real_rss_list)

d_real_list_los = d_real_list[los_indices]
gan_real_rss_list_nlos = gan_real_rss_list[nlos_indices]
gan_real_rss_list_los = gan_real_rss_list[los_indices]
d_real_list_nlos = d_real_list[nlos_indices]

rx_height = 2
tx_height = 5
for epoch in range(opt.n_epochs):
    for i, (conditional_info, power) in enumerate(custom_dataloader):
        batch_size = conditional_info.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_conditional_info = Variable(conditional_info[:, 1:].type(FloatTensor))
        real_power = Variable(power.type(FloatTensor))
        for _ in range(5):
            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise and labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))  # Noise
            gen_labels = Variable(
                FloatTensor(np.random.uniform(0, 1, (batch_size, opt.conditional_info_dim))))  # Conditional Info

            # Generate a batch of images
            gen_powers = generator(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_powers, gen_labels)
            g_loss = adversarial_loss(validity, valid)
            g_loss_list.append(g_loss)
            g_loss.backward()
            optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_power, real_conditional_info)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_powers.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss_list.append(d_loss)
        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(custom_dataloader) + i
        if batches_done % 400 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(custom_dataloader), d_loss.item(), g_loss.item())
            )

            with torch.no_grad():
                noise = torch.randn(len(d_real_list), opt.latent_dim).to("cpu")
                fake_info = torch.rand(len(d_real_list), opt.conditional_info_dim).to("cpu")
                fake_info[:, [0 ,1 ,2 ,3]] = torch.zeros(len(d_real_list), 4).to("cpu")
                fake_info[:, 0] = torch.tensor(d_real_list / max(d_real_list))
                generated_data = generator(noise, fake_info).to("cpu").view(len(d_real_list), 1)

                dist_list_los = []
                dist_list_nlos = []
                gan_rss_list_los = []
                gan_rss_list_nlos = []
                LOS_gen = []
                for i, x in enumerate(generated_data):
                    fake_info_reconst = fake_info[i].detach().to("cpu").numpy()
                    d = fake_info_reconst[0] * max(d_real_list)
                    if d < 1000:
                        fake_info_reconst[4:] *= (max_elev - min_elev)
                        fake_info_reconst[4:] += min_elev
                        rx_total_height = rx_height + fake_info_reconst[-1]
                        tx_total_height = tx_height + fake_info_reconst[4]
                        d_array = np.arange(0, 0.25*(len(fake_info_reconst[4:])), 0.25)
                        LOS = LOSNLOS(tx_total_height, rx_total_height, d_array, fake_info_reconst[4:])
                        LOS_gen.append(LOS)
                        if LOS == 1:
                            dist_list_los.append(d)
                            gan_rss_list_los.append(x.detach().to("cpu").numpy()[0])
                        else:
                            dist_list_nlos.append(d)
                            gan_rss_list_nlos.append(x.detach().to("cpu").numpy()[0])
            a, b = np.polyfit(np.log10(dist_list_los +dist_list_nlos) , gan_rss_list_los + gan_rss_list_nlos, 1)
            plt.figure()
            #plt.scatter(np.log10(dist_list), gan_real_rss_list, alpha=0.2)

            plt.scatter(np.log10(d_real_list_los), gan_real_rss_list_los, c="blue", s=10, alpha=0.2)
            plt.scatter(np.log10(d_real_list_nlos), gan_real_rss_list_nlos, c="red", s=10, alpha=0.2)
            plt.scatter(np.log10(dist_list_los), gan_rss_list_los, c="green", s=10, alpha=0.2)
            plt.scatter(np.log10(dist_list_nlos), gan_rss_list_nlos, c="black", s=10, alpha=0.2)
            plt.xlabel("Logarithmic Distance log10(d[m])")
            plt.ylabel("Power")
            plt.legend(["Real LOS", "Real NLOS", "Generated LOS", "Generated NLOS"])
            plt.title(
                "Path Loss Exponent: %.3f  d_loss: %.3f  g_loss: %.3f" % (abs(a / 10), d_loss.item(), g_loss.item()))
            plt.savefig("images_" + str(opt) + "/image_" + str(image_counter) + ".png")
            image_counter = image_counter + 1
            plt.close("all")