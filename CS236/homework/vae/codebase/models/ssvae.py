# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import torch.utils.data
from codebase import utils as ut
from codebase.models import nns
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim)
        self.dec = nn.Decoder(self.z_dim, self.y_dim)
        self.cls = nn.Classifier(self.y_dim)

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
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be scalar
        ################################################################################
        # Get predicted y labels
        y_logits = self.cls(x)
        y_log_prob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_log_prob, dim=1) # (batch, y_dim)

        # Modify x and z tensors
        # Treat each possible label for each datapoint as a separate instance i.e. x.size x num y labels
        # Then weight each instance with the model's predicted probability
        # Expectation[ELBO(x,y)] 
        num_y_labels = self.y_dim # 10
        batch_size = x.shape[0] # 97
        y = torch.arange(num_y_labels).repeat_interleave(batch_size)
        y = F.one_hot(y, num_y_labels).to(dtype=x.dtype, device=x.device)
        x = ut.duplicate(x, num_y_labels)

        # Get reconstructed x
        mean_post, variance_post = self.enc(x, y)
        z = ut.sample_gaussian(mean_post, variance_post)
        x_logits = self.dec(z, y)

        # Evaluate kl_y, kl_z, rec
        mean_prior, variance_prior = self.z_prior
        uniform_dist_over_classes = np.log(1.0 / num_y_labels)
        kl_y = ut.kl_cat(y_prob, y_log_prob, uniform_dist_over_classes)
        kl_z = ut.kl_normal(mean_post, variance_post, mean_prior, variance_prior)
        rec = -ut.log_bernoulli_with_logits(x, x_logits)

        # Reshape the reconstruction loss and KL divergence to match y_prob dimensions
        rec_reshaped = rec.reshape(self.y_dim, -1)
        kl_z_reshaped = kl_z.reshape(self.y_dim, -1)

        # Weight the losses by the predicted class probabilities
        weighted_rec = y_prob.t() * rec_reshaped
        weighted_kl_z = y_prob.t() * kl_z_reshaped

        # Sum the weighted losses to get the expected values
        expected_rec = weighted_rec.sum(0)
        expected_kl_z = weighted_kl_z.sum(0)

        kl_y, kl_z , rec = kl_y.mean(), expected_kl_z.mean(), expected_rec.mean()
        nelbo = rec + kl_z + kl_y
        ################################################################################
        # End of code modification
        ################################################################################
        return nelbo, kl_z, kl_y, rec

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
