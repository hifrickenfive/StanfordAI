import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedLinear(nn.Linear):
    """Masked linear layer for MADE: takes in mask as input and masks out connections in the linear layers."""

    def __init__(self, input_size, output_size, mask):
        super().__init__(input_size, output_size)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


class PermuteLayer(nn.Module):
    """Layer to permute the ordering of inputs.

    Because our data is 2-D, forward() and inverse() will reorder the data in the same way.
    """

    def __init__(self, num_inputs):
        super(PermuteLayer, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])

    def forward(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )

    def inverse(self, inputs):
        return inputs[:, self.perm], torch.zeros(
            inputs.size(0), 1, device=inputs.device
        )


class MADE(nn.Module):
    """Masked Autoencoder for Distribution Estimation.
    https://arxiv.org/abs/1502.03509

    Uses sequential ordering as in the MAF paper.
    Gaussian MADE to work with real-valued inputs"""

    def __init__(self, input_size, hidden_size, n_hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden

        masks = self.create_masks()

        # construct layers: inner, hidden(s), output
        self.net = [MaskedLinear(self.input_size, self.hidden_size, masks[0])]
        self.net += [nn.ReLU(inplace=True)]
        # iterate over number of hidden layers
        for i in range(self.n_hidden):
            self.net += [MaskedLinear(self.hidden_size, self.hidden_size, masks[i + 1])]
            self.net += [nn.ReLU(inplace=True)]
        # last layer doesn't have nonlinear activation
        self.net += [
            MaskedLinear(self.hidden_size, self.input_size * 2, masks[-1].repeat(2, 1))
        ]
        self.net = nn.Sequential(*self.net)

    def create_masks(self):
        """
        Creates masks for sequential (natural) ordering.
        """
        masks = []
        input_degrees = torch.arange(self.input_size)
        degrees = [input_degrees]  # corresponds to m(k) in paper

        # iterate through every hidden layer
        for n_h in range(self.n_hidden + 1):
            degrees += [torch.arange(self.hidden_size) % (self.input_size - 1)]
        degrees += [input_degrees % self.input_size - 1]
        self.m = degrees

        # output layer mask
        for (d0, d1) in zip(degrees[:-1], degrees[1:]):
            masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

        return masks

    def forward(self, z):
        """
        Run the forward mapping (z -> x) for MAF through one MADE block.
        :param z: Input noise of size (batch_size, self.input_size)
        :return: (x, log_det). log_det should be 1-D (batch_dim,)
        """
        # YOUR CODE STARTS HERE

        # My baseline, which autograder declared as incorrect
        # network_output = self.net(z)
        # mu, log_sigma = network_output.chunk(2, dim=1)
        # x = mu + torch.exp(log_sigma) * z
        # log_det = - torch.sum(log_sigma, dim=1)

        # Attempt 2
        x = z
        for i in range(x.shape[1]):
            network_output = self.net(x)
            mu, log_sigma = network_output.chunk(2, dim=1)
            x = mu + torch.exp(log_sigma) * z

        log_det = - torch.sum(log_sigma, dim=1)

        return x, log_det

    def inverse(self, x):
        """
        Run one inverse mapping (x -> z) for MAF through one MADE block.
        :param x: Input data of size (batch_size, self.input_size)
        :return: (z, log_det). log_det should be 1-D (batch_dim,)
        """
        # YOUR CODE STARTS HERE
        network_output = self.net(x)
        mu, log_sigma = network_output.chunk(2, dim=1)
        z = (x - mu) / torch.exp(log_sigma)
        log_det = -torch.sum(log_sigma, dim=1)
        # YOUR CODE ENDS HERE
        return z, log_det


class MAF(nn.Module):
    """
    Masked Autoregressive Flow, using MADE layers.
    https://arxiv.org/abs/1705.07057
    """

    def __init__(self, input_size, hidden_size, n_hidden, n_flows):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.n_flows = n_flows
        self.base_dist = torch.distributions.normal.Normal(0, 1)

        # need to flip ordering of inputs for every layer
        nf_blocks = []
        for i in range(self.n_flows):
            nf_blocks.append(MADE(self.input_size, self.hidden_size, self.n_hidden))
            nf_blocks.append(PermuteLayer(self.input_size))  # permute dims
        self.nf = nn.Sequential(*nf_blocks)

    def log_probs(self, x):
        """
        Obtain log-likelihood p(x) through one pass of MADE
        :param x: Input data of size (batch_size, self.input_size)
        :return: log_prob. This should be a Python scalar.
        """
        # YOUR CODE STARTS HERE
        z_prev = x
        log_det_jacobian = 0
        for flow in self.nf:
            z, log_det = flow.inverse(z_prev)
            # print(self.base_dist.log_prob(z).sum(dim=1))

            if torch.isnan(z).any():
                print('Nan detected in z')
                z, __ = flow.inverse(z_prev)

            if type(flow) != PermuteLayer: # log_det's shape after a permute layer is [100,1] and all zeros so skip
                log_det_jacobian += log_det

            z_prev = z

        # Evaluate the log probability of the base distribution
        log_prob_base_dist = self.base_dist.log_prob(z).sum(dim=1)

        # Combine the log probability of the base distribution with the log determinant
        # of the Jacobian to get the final log probability
        log_prob = (log_prob_base_dist + log_det_jacobian).sum() / x.size(0)
        # YOUR CODE ENDS HERE

        return log_prob

    def loss(self, x):
        """
        Compute the loss.
        :param x: Input data of size (batch_size, self.input_size)
        :return: loss. This should be a Python scalar.
        """
        return -self.log_probs(x)

    def sample(self, device, n):
        """
        Draw <n> number of samples from the model.
        :param device: [cpu,cuda]
        :param n: Number of samples to be drawn.
        :return: x_sample. This should be a numpy array of size (n, self.input_size)
        """
        with torch.no_grad():
            x_sample = torch.randn(n, self.input_size).to(device)
            for flow in self.nf[::-1]:
                x_sample, log_det = flow.forward(x_sample)
            x_sample = x_sample.view(n, self.input_size)
            x_sample = x_sample.cpu().data.numpy()

        return x_sample
