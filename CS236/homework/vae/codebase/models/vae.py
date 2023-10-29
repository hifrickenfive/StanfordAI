# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class VAE(nn.Module):
    def __init__(self, nn="v1", name="vae", z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Notes
        # x.shape = 97, 784. Minibatch 97 imgs of 784 bits (28 x 28 pixels)
        # z_dim = 10. There's 10 latent variables per img, x
        # P(z)~N(z_prior_m, z_prior_v)

        # Sample latent variables
        mean_post, variance_post = self.enc(x)
        z = ut.sample_gaussian(mean_post, variance_post) # shape 97, 10

        # Reconstruction loss
        logits = self.dec(z)
        rec = -ut.log_bernoulli_with_logits(x, logits)

        # Regularization penality via KL(posterior|prior)
        mean_prior, variance_prior = self.z_prior
        kl = ut.kl_normal(mean_post, variance_post, mean_prior, variance_prior)
   
        # Negative ELBO
        # "Make sure to compute the average ELBO over the mini batch" PDF pg. 2/7 HW2, 2023
        nelbo = kl + rec
        nelbo , kl , rec = nelbo.mean(), kl.mean(), rec.mean()
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be scalar
        ################################################################################
        # Sample latent variables via IWAE
        mean_post, variance_post = self.enc(x)
        
        # Duplicate
        mean_post = ut.duplicate(mean_post, iw)
        variance_post = ut.duplicate(variance_post, iw)
        x = ut.duplicate(x, iw)

        # Reconstruct x
        z = ut.sample_gaussian(mean_post, variance_post)
        logits = self.dec(z)
        
        # Reconstruction loss
        rec = -ut.log_bernoulli_with_logits(x, logits)

        # Regularization penalty
        mean_prior, variance_prior = self.z_prior
        kl = ut.log_normal(z, mean_post, variance_post) - ut.log_normal(z, mean_prior, variance_prior)
 
        # NELBO
        # Don't forget to find the mean over the batch!
        nelbo = kl + rec
        niwae = -ut.log_mean_exp(-nelbo.reshape(iw, -1), dim=0)
        niwae, kl, rec = niwae.mean(), kl.mean(), rec.mean()
        ################################################################################
        # End of code modification
        ################################################################################
        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict(
            (
                ("train/loss", nelbo),
                ("gen/elbo", -nelbo),
                ("gen/kl_z", kl),
                ("gen/rec", rec),
            )
        )

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim),
        )

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
