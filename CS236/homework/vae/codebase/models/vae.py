# Copyright (c) 2021 Rui Shu

import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
        # z_dim = 10. This means there's 10 latent variables
        # We know z's distribution z_prior_m, z_prior_v
        # If sample 10 latent variables, then decode them to get a reconstructed x
        batch_size = x.shape[0]

        # Reconstruct x
        z_prior_m_batched = torch.full((batch_size, self.z_dim), self.z_prior_m.item()) 
        z_prior_v_batched = torch.full((batch_size, self.z_dim), self.z_prior_v.item())
        z = ut.sample_gaussian(z_prior_m_batched, z_prior_v_batched) # 91, 10
        reconstructed_x = self.dec(z) # 97, 784. Values are negative and positive
        reconstructed_x_rescaled = torch.sigmoid(reconstructed_x) # Rescale values to (0,1)

        # Reconstruction loss
        rec = F.binary_cross_entropy(reconstructed_x_rescaled, x, reduction="mean")

        kl = ut.kl_normal(qm, qv, pm, pv) # KL(q||p) betwen two normal dist. mean, variance
        nelbo = rec + rec
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
