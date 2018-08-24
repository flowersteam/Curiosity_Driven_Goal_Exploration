import numpy as np
import datetime
import os

import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F

import visdom

from latentgoalexplo.actors.meta_actors import *


CUDA = torch.cuda.is_available()
device = torch.device("cuda" if CUDA else "cpu")


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Channelize(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 32, 4, 4)


class BetaVAE(nn.Module):
    def __init__(self, n_input, n_latents, Ta, capacity, capacity_change_duration, network_type='fc', n_channels=1,
                 visdom_env="main", visdom_record=True):
        super(BetaVAE, self).__init__()

        self.n_input = n_input
        self.n_latents = n_latents

        self.n_iters = 0
        self.Ta = Ta
        self.capacity = capacity
        self.capacity_change_duration = capacity_change_duration

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.network_type = network_type
        self.n_channels = n_channels
        self.generate_architecture(network_type)

        self.visdom_record = visdom_record
        if self.visdom_record:
            self.visdom = visdom.Visdom(env=visdom_env)
        self.visdom_wins = dict()

    def generate_architecture(self, type='fc'):
        if type == 'fc':
            self.encoder = nn.Sequential(
                    Flatten(),
                    nn.Linear(self.n_input, 1200),
                    nn.ReLU(),
                    nn.Linear(1200, 1200),
                    nn.ReLU(),
                    nn.Linear(1200, 2 * self.n_latents)
            )

            self.decoder = nn.Sequential(
                    nn.Linear(self.n_latents, 1200),
                    nn.ReLU(),
                    nn.Linear(1200, 1200),
                    nn.ReLU(),
                    nn.Linear(1200, self.n_input)
            )

        if type == 'cnn':
            # Works only for 1 * 64 * 64 images
            self.encoder = nn.Sequential(
                    nn.Conv2d(self.n_channels, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 32, 32
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 16, 16
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 4, 4
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(32 * 4 * 4, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * self.n_latents)
            )

            self.decoder = nn.Sequential(
                    nn.Linear(self.n_latents, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4 * 4 * 32),
                    nn.ReLU(),
                    Channelize(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 16, 16
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 32, 32
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=1)  # b, n_channels, 64, 64
            )

        if type == 'cnn128':
            # Works only for 1 * 64 * 64 images
            self.encoder = nn.Sequential(
                    nn.Conv2d(self.n_channels, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 64, 64
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 32, 32
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 16, 16
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
                    nn.ReLU(),
                    Flatten(),
                    nn.Linear(32 * 8 * 8, 512),
                    nn.ReLU(),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2 * self.n_latents)
            )

            self.decoder = nn.Sequential(
                    nn.Linear(self.n_latents, 256),
                    nn.ReLU(),
                    nn.Linear(256, 4 * 4 * 32),
                    nn.ReLU(),
                    Channelize(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 8, 8
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 16, 16
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 32, 32
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),  # b, 32, 64, 64
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, self.n_channels, kernel_size=4, stride=2, padding=1)  # b, n_channels, 128, 128
            )

    def encode(self, x):
        # h = self.relu(self.fc1(x))
        # h = self.relu(self.fc2(h))
        # return self.fc31(h), self.fc32(h)
        return torch.chunk(self.encoder(x), 2, dim=1)

    def decode(self, z):
        # h = self.relu(self.fc4(z))
        # h = self.relu(self.fc5(h))
        #         h = self.tanh(self.fc6(h))
        # return self.fc7(h)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # eps = Variable(std.data.new(std.size()).normal_())
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def loss_function(self, recon_x, x, mu, logvar, beta, c, capacity):
        BCE = F.binary_cross_entropy_with_logits(recon_x, x, size_average=False) / x.size()[0]

        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)

        # Version 1
        # We force the latents of to have a KL=capacity, on average on a minibatch
        KLD_loss = c * beta * torch.abs(KLD.mean() - capacity)

        # Version 2
        # We force the latents of each sample to have a KL=capacity
        # KLD_loss = torch.abs(KLD - capacity).mean()
        # KLD_loss = c * beta * KLD_loss

        return BCE, KLD_loss, KLD.mean()

    def KLD(self, data_loader):
        self.eval()
        KLD = torch.zeros(self.n_latents).to(device)
        with torch.no_grad():
            for i, (data, _) in enumerate(data_loader):
                if i > 100:
                    break
                data = data.to(device)
                _, mu, logvar = self(data)
                # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                KLD += -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=0)
        KLD /= (i + 1)
        return KLD

    def _perform_epoch(self, train_loader, optimizer, beta):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            self.n_iters += 1
            c = min([1., 0.01 + self.n_iters / self.Ta])
            capacity = self.calc_encoding_capacity()
            optimizer.zero_grad()
            recon_batch, mu, logvar = self.forward(data)
            recon_loss, kld_loss, kld = self.loss_function(recon_batch, target, mu, logvar, beta, c, capacity)
            if self.n_iters % 200 == 0 and self.visdom_record:
                self.plot_losses(recon_loss, kld_loss, kld)
                self.plot_latents_kld_values(self.KLD(train_loader).cpu().detach().numpy(), self.n_iters)
                self.train()
            loss = recon_loss + kld_loss
            loss.backward()
            optimizer.step()
        return

    def test(self, X_test, beta, capacity):
        self.eval()
        recon_X, mu, logvar = self(X_test)
        recon_loss, kld_loss, _ = self.loss_function(recon_X, X_test, mu, logvar, beta, 1, capacity)
        test_loss = recon_loss + kld_loss
        test_loss = test_loss.item()
        test_loss /= X_test.size()[0]
        return test_loss

    def plot_losses(self, recon_loss, kld_loss, kld):
        if 'total_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['total_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=(kld_loss + recon_loss).cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['total_loss'], update='append')
        else:
            self.visdom_wins['total_loss'] = self.visdom.line(X=torch.zeros(1), Y=(recon_loss + kld_loss).cpu().detach().unsqueeze(0),
                                                              opts={'title': 'Total loss'})

        if 'recon_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['recon_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=recon_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['recon_loss'], update='append')
        else:
            self.visdom_wins['recon_loss'] = self.visdom.line(X=torch.zeros(1), Y=recon_loss.cpu().detach().unsqueeze(0),
                                                              opts={'title': 'Reconstruction loss'})

        if 'kld_loss' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['kld_loss']):
            self.visdom.line(X=torch.ones(1) * self.n_iters, Y=kld_loss.cpu().detach().unsqueeze(0),
                             win=self.visdom_wins['kld_loss'], update='append')
        else:
            self.visdom_wins['kld_loss'] = self.visdom.line(X=torch.zeros(1), Y=kld_loss.cpu().detach().unsqueeze(0),
                                                            opts={'title': 'KLD loss with capacity and beta'})

        if 'kld' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['kld']):
            self.visdom.line(X=torch.ones(1) * self.n_iters,
                             Y=torch.cat([kld.cpu().detach().unsqueeze(0), torch.from_numpy(
                                 np.array([self.calc_encoding_capacity()])).float()]).unsqueeze(0),
                             win=self.visdom_wins['kld'], update='append')
        else:
            self.visdom_wins['kld'] = self.visdom.line(X=torch.zeros(1),
                                                       Y=torch.cat([kld.cpu().detach().unsqueeze(0), torch.from_numpy(np.array(
                                                               [self.calc_encoding_capacity()])).float()]).unsqueeze(0),
                                                       opts={'title': 'KLD'})

    def plot_latent_space(self, data_loader, sorted_latents):
        self.eval()
        indices = sorted_latents
        z_means = []
        z_sigmas = []
        for batch_idx, (data, target) in enumerate(data_loader):
            data = data.to(device)
            if batch_idx * data.size(0) > 1000:
                break
            _, mu, logvar = self.forward(data)
            z_means.append(mu)
            z_sigmas.append(logvar.exp())
        z_means = torch.cat(z_means)
        z_means = z_means[:, indices[:3]]
        z_means = z_means.cpu().detach()
        # Scatter plot of latent space
        if 'latent_space' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['latent_space']):
            self.visdom.scatter(z_means, win=self.visdom_wins['latent_space'],
                                opts={'title': 'latent space', 'markersize': 10})
        else:
            self.visdom_wins['latent_space'] = self.visdom.scatter(z_means,
                                                                   opts={'title': 'latent space', 'markersize': 10})
        # Values of the variance of latents
        z_sigmas = torch.cat(z_sigmas).mean(dim=0)

        # Values of the variance of the latents of one image
        # z_sigmas = z_sigmas[0][0]
        zss_str = "capacity=" + "{0:.2f}".format(self.calc_encoding_capacity()) + ", n_iters=" + str(self.n_iters)
        for i, zss in enumerate(z_sigmas.detach()):
            string = "z{0}={1:.4f}".format(i, zss)
            zss_str +=  ", " + string
        if 'latents_var' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['latents_var']):
            self.visdom.text(zss_str, win=self.visdom_wins['latents_var'], append=True)
        else:
            self.visdom_wins['latents_var'] = self.visdom.text(zss_str)

    def plot_reconstruction(self, test_loader, size=8):
        self.eval()
        imgs_sampled, _ = next(iter(test_loader))
        if self.network_type == 'fc':
            height = int(imgs_sampled.size()[1] ** 0.5)
            width = int(imgs_sampled.size()[1] ** 0.5)
        elif self.network_type == 'cnn':
            height = 64
            width = 64
        elif self.network_type == 'cnn128':
            height = 128
            width = 128
        imgs_sampled = imgs_sampled.to(device)

        recons = self(imgs_sampled)[0]
        recons = torch.sigmoid(recons)

        imgs_sampled = imgs_sampled.cpu().contiguous().detach().view([-1, self.n_channels, height, width])
        imgs_sampled = imgs_sampled[:size]
        recons = recons.cpu().contiguous().detach().view([-1, self.n_channels, height, width])
        recons = recons[:size]
        truth_vs_recons = torch.cat((imgs_sampled, recons))

        if 'recon' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['recon']):
            self.visdom.images(truth_vs_recons, nrow=size, win=self.visdom_wins['recon'],
                               opts={'title': 'x vs x_tilde',
                                     'jpgquality': 100,
                                     'height': 2 * 2 * height,
                                     'width': 2 * size * width})
        else:
            self.visdom_wins['recon'] = self.visdom.images(truth_vs_recons, nrow=size,
                                                           opts={'title': 'x vs x_tilde',
                                                                 'jpgquality': 100,
                                                                 'height': 2 * 2 * height,
                                                                 'width': 2 * size * width})

    def plot_reconstruction_vs_latent(self, sorted_latents, img, n_sigma=10):
        self.eval()
        indices = sorted_latents
        #         img = next(iter(data_loader))[:1]

        #         a = MovingObjectRenderer(height=70, width=70, rgb=False, object_size=0.3, object_type='square')
        #         a.reset()
        #         a.act(observation=np.array([0., 0., 0., 0., 0., 0., 0., 0, 0]))
        #         img = torch.from_numpy(a.rendering.ravel()).float().unsqueeze(0)
        #         a.terminate()

        if self.network_type == 'fc':
            img = Variable(torch.from_numpy(img).float()).unsqueeze(0)
            height = int(img.size()[1] ** 0.5)
            width = int(img.size()[1] ** 0.5)
        elif self.network_type == 'cnn':
            # img = Variable(torch.from_numpy(img).float()).contiguous().view(1, self.n_channels, 64, 64)
            img = Variable(torch.from_numpy(img).float()).unsqueeze(0) # .permute(0, 2, 3, 1)
            height = 64
            width = 64
        elif self.network_type == 'cnn128':
            # img = Variable(torch.from_numpy(img).float()).contiguous().view(1, self.n_channels, 128, 128)
            img = Variable(torch.from_numpy(img).float()).unsqueeze(0) # .permute(0, 2, 3, 1)
            height = 128
            width = 128

        img = img.to(device)
        sampled_imgs = []

        for indice in indices:
            # Recover model latent variables for img
            latents = self.encode(img)[0].cpu()
            latents.data[0, indice] = 0
            new_lat = torch.zeros_like(latents)
            new_lat.data[0, indice] = 1
            for sigma in np.linspace(-3, 3, n_sigma):
                latents_sample = latents + float(sigma) * new_lat
                latents_sample = latents_sample.to(device)
                sample = torch.sigmoid(self.decode(latents_sample))
                sampled_imgs.append(sample.cpu().detach().numpy().reshape(self.n_channels, height, width))

        if 'img_vs_latent' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['img_vs_latent']):
            self.visdom.images(sampled_imgs, nrow=n_sigma, win=self.visdom_wins['img_vs_latent'],
                               opts={'title': 'Reconstruction against latent value',
                                     'jpgquality': 100,
                                     'height': self.n_latents * 80,
                                     'width': n_sigma * 80})
        else:
            self.visdom_wins['img_vs_latent'] = self.visdom.images(sampled_imgs, nrow=n_sigma,
                                                                   opts={
                                                                       'title': 'Reconstruction against latent value',
                                                                       'jpgquality': 100,
                                                                       'height': self.n_latents * 80,
                                                                       'width': n_sigma * 80})

    def plot_latents_kld_values(self, kld_latents, epoch):
        KLD = kld_latents
        if 'latents_kld' in self.visdom_wins and self.visdom.win_exists(self.visdom_wins['latents_kld']):
            self.visdom.line(X=torch.ones(1) * epoch, Y=np.expand_dims(KLD, 0),
                             win=self.visdom_wins['latents_kld'], update='append')
        else:
            self.visdom_wins['latents_kld'] = self.visdom.line(X=torch.zeros(1), Y=np.expand_dims(KLD, 0),
                                                               opts={'title': 'KLD for each latent'})

    def calc_encoding_capacity(self):
        if self.n_iters > self.capacity_change_duration:
            return self.capacity
        else:
            return self.capacity * self.n_iters / self.capacity_change_duration


class PytorchBetaVAERepresentation(AbstractActor, IRepresentation, ITrainable, IParameterized):

    def __init__(self, n_latents, initial_epochs, beta, *args, network_type='cnn', n_channels=3, height=70, width=70,
                 batch_size=128, learning_rate=1e-2, Ta=1, capacity=0, capacity_change_duration=1,
                 visdom_record=False, visdom_env="main", log_interval=40, store_loader_gpu=True, logdir='',
                 **kwargs):

        if logdir != '':
            logdir += datetime.datetime.now().strftime("/%Y%m%d-%H%M%S")

        self._beta = beta
        self._n_latents = n_latents

        self._model = None
        self._network_type = network_type
        self._n_channels = n_channels
        self._height = height
        self._width = width
        self._data_size = height * width
        self._batch_size = batch_size
        self._initial_epochs = initial_epochs
        self._optimizer = None
        self._learning_rate = learning_rate
        self._net_epochs = 0
        self._Ta = Ta
        self._capacity = capacity
        self._capacity_change_duration = capacity_change_duration
        self._store_loader_gpu = store_loader_gpu

        self._typical_img = None

        self._prediction = None
        self._performance = None
        self._representation = None
        self._data_mean = None
        self._data_std = None
        self._kld_latents = None
        self._sorted_latents = None

        self._visdom_record = visdom_record
        self._visdom_env = visdom_env
        self._log_interval = log_interval

    def reset(self, X_train, y_train, typical_img=None):

        if self._network_type == 'fc':
            X_train = X_train.reshape(-1, self._height * self._width)
            y_train = y_train.reshape(-1, self._height * self._width)
            if typical_img is not None:
                self._typical_img = typical_img.ravel()
        elif self._n_channels == 1:
            X_train = X_train.reshape(-1, 1, self._height, self._width)
            y_train = y_train.reshape(-1, 1, self._height, self._width)
            if typical_img is not None:
                self._typical_img = typical_img.reshape(1, self._height, self._width)
        elif self._n_channels == 3:
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=0)
                y_train = np.expand_dims(y_train, axis=0)
            X_train = X_train.transpose(0, 3, 1, 2)
            y_train = y_train.transpose(0, 3, 1, 2)
            if typical_img is not None:
                self._typical_img = typical_img.transpose(2, 0, 1)

        self._data_mean = X_train.mean(axis=0)
        self._data_std = X_train.std(axis=0)

        self._model = BetaVAE(self._data_size, self._n_latents, self._Ta, self._capacity,
                              self._capacity_change_duration, self._network_type, self._n_channels,
                              self._visdom_env, self._visdom_record)
        self._model.to(device)
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)

        #         X_train -= self._data_mean
        #         X_train /= self._data_std

        if CUDA and self._store_loader_gpu:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                                     torch.from_numpy(y_train).float().to(device))
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                     torch.from_numpy(y_train).float())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        for i in range(self._initial_epochs):
            self._model._perform_epoch(data_loader, self._optimizer, self._beta)
            if self._visdom_record and self._net_epochs % self._log_interval == 0:
                self._update_latents_order(data_loader)
                self._model.plot_reconstruction(data_loader)
                self._model.plot_reconstruction_vs_latent(self._sorted_latents, self._typical_img)
                self._model.plot_latent_space(data_loader, self._sorted_latents)
            self._net_epochs += 1

        self._update_prediction(data_loader)
        self._update_performance(data_loader)
        self._update_representation(data_loader)
        self._update_latents_order(data_loader)

    def train_representation(self, X_train, y_train, epochs):

        if self._network_type == 'fc':
            X_train = X_train.reshape(-1, self._height * self._width)
            y_train = y_train.reshape(-1, self._height * self._width)
        elif self._n_channels == 1:
            X_train = X_train.reshape(-1, 1, self._height, self._width)
            y_train = y_train.reshape(-1, 1, self._height, self._width)
        elif self._n_channels == 3:
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=0)
                y_train = np.expand_dims(y_train, axis=0)
            X_train = X_train.transpose(0, 3, 1, 2)
            y_train = y_train.transpose(0, 3, 1, 2)

        self._data_mean = X_train.mean(axis=0)
        self._data_std = X_train.std(axis=0)

        #         X_train -= self._data_mean
        #         X_train /= self._data_std

        if CUDA and self._store_loader_gpu:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                                     torch.from_numpy(y_train).float().to(device))
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                     torch.from_numpy(y_train).float())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        for i in range(epochs):
            self._model._perform_epoch(data_loader, self._optimizer, self._beta)
            if self._visdom_record and self._net_epochs % self._log_interval == 0:
                self._update_latents_order(data_loader)
                self._model.plot_reconstruction(data_loader)
                self._model.plot_reconstruction_vs_latent(self._sorted_latents, self._typical_img)
                self._model.plot_latent_space(data_loader, self._sorted_latents)
            self._net_epochs += 1

        self._update_latents_order(data_loader)
        self._update_prediction(data_loader)
        self._update_performance(data_loader)
        self._update_representation(data_loader)

    def train_representation_n_iters(self, X_train, y_train, n_iters):

        if self._network_type == 'fc':
            X_train = X_train.reshape(-1, self._height * self._width)
            y_train = y_train.reshape(-1, self._height * self._width)
        elif self._n_channels == 1:
            X_train = X_train.reshape(-1, 1, self._height, self._width)
            y_train = y_train.reshape(-1, 1, self._height, self._width)
        elif self._n_channels == 3:
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=0)
                y_train = np.expand_dims(y_train, axis=0)
            X_train = X_train.transpose(0, 3, 1, 2)
            y_train = y_train.transpose(0, 3, 1, 2)

        self._data_mean = X_train.mean(axis=0)
        self._data_std = X_train.std(axis=0)

        #         X_train -= self._data_mean
        #         X_train /= self._data_std

        if CUDA and self._store_loader_gpu:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                                     torch.from_numpy(y_train).float().to(device))
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                     torch.from_numpy(y_train).float())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        epochs = n_iters * self._batch_size // len(X_train)
        for i in range(epochs):
            self._model._perform_epoch(data_loader, self._optimizer, self._beta)
            if self._visdom_record and self._net_epochs % self._log_interval == 0:
                self._update_latents_order(data_loader)
                self._model.plot_reconstruction(data_loader)
                self._model.plot_reconstruction_vs_latent(self._sorted_latents, self._typical_img)
                self._model.plot_latent_space(data_loader, self._sorted_latents)
            self._net_epochs += 1

        self._update_latents_order(data_loader)

    def estimate_kld(self, X_train, y_train):

        if self._network_type == 'fc':
            X_train = X_train.reshape(-1, self._height * self._width)
            y_train = y_train.reshape(-1, self._height * self._width)
        elif self._n_channels == 1:
            X_train = X_train.reshape(-1, 1, self._height, self._width)
            y_train = y_train.reshape(-1, 1, self._height, self._width)
        elif self._n_channels == 3:
            if len(X_train.shape) == 3:
                X_train = np.expand_dims(X_train, axis=0)
                y_train = np.expand_dims(y_train, axis=0)
            X_train = X_train.transpose(0, 3, 1, 2)
            y_train = y_train.transpose(0, 3, 1, 2)

        #         X_train -= self._data_mean
        #         X_train /= self._data_std

        if CUDA and self._store_loader_gpu:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                                     torch.from_numpy(y_train).float().to(device))
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                     torch.from_numpy(y_train).float())
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        self._update_latents_order(data_loader)

    def load_model(self, filepath, typical_img=None):
        if typical_img is not None:
            if self._network_type == 'fc':
                self._typical_img = typical_img.ravel()
            elif self._n_channels == 1:
                self._typical_img = typical_img.reshape(1, self._height, self._width)
            elif self._n_channels == 3:
                self._typical_img = typical_img.transpose(2, 0, 1)

        self._model = BetaVAE(self._data_size, self._n_latents, self._Ta, self._capacity,
                              self._capacity_change_duration, self._network_type, self._n_channels,
                              self._visdom_env, self._visdom_record)
        if CUDA:
            self._model.to(device)
            self._model.load_state_dict(torch.load(filepath))
        else:
            self._model.load_state_dict(torch.load(filepath, map_location=lambda storage, loc: storage))

        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)

        self._model.eval()

    def act(self, *args, X_pred=None, X_train=None, y_train=None, X_test=None, y_test=None):

        if X_train is not None and X_pred is not None and X_test is not None:
            raise Exception("Calling multiple modes at once is not possible.")

        if X_train is not None:

            if self._network_type == 'fc':
                X_train = X_train.reshape(-1, self._height * self._width)
                y_train = y_train.reshape(-1, self._height * self._width)
            elif self._n_channels == 1:
                X_train = X_train.reshape(-1, 1, self._height, self._width)
                y_train = y_train.reshape(-1, 1, self._height, self._width)
            elif self._n_channels == 3:
                if len(X_train.shape) == 3:
                    X_train = np.expand_dims(X_train, axis=0)
                    y_train = np.expand_dims(y_train, axis=0)
                X_train = X_train.transpose(0, 3, 1, 2)
                y_train = y_train.transpose(0, 3, 1, 2)

            #             X_train -= self._data_mean
            #             X_train /= self._data_std

            if CUDA and self._store_loader_gpu:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float().to(device),
                                                         torch.from_numpy(y_train).float().to(device))
            else:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train).float(),
                                                         torch.from_numpy(y_train).float())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=True)
            self._model._perform_epoch(data_loader, self._optimizer, self._beta)

        elif X_test is not None:

            if self._network_type == 'fc':
                X_test = X_test.reshape(-1, self._height * self._width)
                y_test = y_test.reshape(-1, self._height * self._width)
            elif self._n_channels == 1:
                X_test = X_test.reshape(-1, 1, self._height, self._width)
                y_test = y_test.reshape(-1, 1, self._height, self._width)
            elif self._n_channels == 3:
                if len(X_test.shape) == 3:
                    X_test = np.expand_dims(X_test, axis=0)
                    y_test = np.expand_dims(y_test, axis=0)
                X_test = X_test.transpose(0, 3, 1, 2)
                y_test = y_test.transpose(0, 3, 1, 2)

            #             X_test -= self._data_mean
            #             X_test /= self._data_std

            if CUDA and self._store_loader_gpu:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float().to(device),
                                                         torch.from_numpy(y_test).float().to(device))
            else:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test).float(),
                                                         torch.from_numpy(y_test).float())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
            self._update_performance(data_loader)

        elif X_pred is not None:

            if self._network_type == 'fc':
                X_pred = X_pred.reshape(-1, self._height * self._width)
            elif self._n_channels == 1:
                X_pred = X_pred.reshape(-1, 1, self._height, self._width)
            elif self._n_channels == 3:
                if len(X_pred.shape) == 3:
                    X_pred = np.expand_dims(X_pred, axis=0)
                X_pred = X_pred.transpose(0, 3, 1, 2)

            #             X_pred -= self._data_mean
            #             X_pred /= self._data_std

            if CUDA and self._store_loader_gpu:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_pred).float().to(device),
                                                         torch.from_numpy(X_pred).float().to(device))
            else:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_pred).float(),
                                                         torch.from_numpy(X_pred).float())
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=self._batch_size, shuffle=False)
            self._update_prediction(data_loader)
            self._update_representation(data_loader)

    def terminate(self):

        self._batch_size = None
        self._initial_epochs = None
        self._net_epochs = None
        self._prediction = None
        self._performance = None
        self._representation = None
        self._kld_latents = None
        self._sorted_latents = None
        torch.cuda.empty_cache()

    @property
    def prediction(self):

        return self._prediction

    @property
    def performance(self):

        return self._performance

    @property
    def dataset(self):

        return self._X, self._y

    @property
    def representation(self):

        return self._representation

    @property
    def sorted_latents(self):

        return self._sorted_latents

    @property
    def kld_latents(self):

        return self._kld_latents

    @property
    def parameters(self):

        return [param for param in self._model.parameters()]

    @classmethod
    def test(cls):

        pass

    def _update_prediction(self, data_loader):

        self._model.eval()
        pred = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                X_train = data.to(device)
                pred.append(self._model(X_train)[0].cpu().detach().numpy())

        self._prediction = np.concatenate(pred)

    def _update_performance(self, data_loader):

        capacity = self._model.calc_encoding_capacity()
        self._performance = 0

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                X_train = data.to(device)
                self._performance += self._model.test(X_train, self._beta, capacity)

        self._performance /= len(data_loader)

    def _update_representation(self, data_loader):

        self._model.eval()
        rep = []

        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(data_loader):
                X_train = data.to(device)
                rep.append(self._model.encode(X_train)[0].cpu().detach().numpy())

        self._representation = np.concatenate(rep)

    def _update_latents_order(self, data_loader):
        KLD = self._model.KLD(data_loader)
        values, indices = KLD.sort(descending=True)
        self._kld_latents = values.cpu().detach().numpy()
        self._sorted_latents = indices.cpu().detach().numpy()
