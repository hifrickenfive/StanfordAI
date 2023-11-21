import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR DISCRIMINATOR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    x_generated = g(z)
    d_loss = -F.logsigmoid(d(x_real)).mean() - F.logsigmoid(1-d(x_generated)).mean()
    # YOUR CODE ENDS HERE

    return d_loss

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR GENERATOR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    x_generated = g(z)
    d_fake = d(x_generated)
    g_loss = -F.logsigmoid(d_fake).mean()
    # YOUR CODE ENDS HERE

    return g_loss


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR DISCRIMINATOR STARTS HERE
    x_generated = g(z, y_real)
    d_loss = -F.logsigmoid(d(x_real, y_real)).mean() - F.logsigmoid(1-d(x_generated, y_real)).mean()
    # YOUR CODE ENDS HERE

    return d_loss


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_fake = y_real  # use the real labels as the fake labels as well

    # YOUR DISCRIMINATOR STARTS HERE
    x_generated = g(z, y_real)
    g_loss = -F.logsigmoid(d(x_generated, y_real)).mean()
    # YOUR CODE ENDS HERE

    return g_loss


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    penalty = 10 # lambda = 10 suggested
    x_generated = g(z)

    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    x_sampled_from_r_theta = alpha * x_generated + (1 - alpha) * x_real

    grad = torch.autograd.grad(
        d(x_sampled_from_r_theta).sum(),
        x_sampled_from_r_theta,
        create_graph=True
    )
    
    grad_norm = grad[0].reshape(batch_size, -1).norm(dim=1)

    d_loss = (
      d(x_generated).mean() -
      d(x_real).mean() + 
      penalty*((grad_norm - 1)**2).mean()
    )
    # YOUR CODE ENDS HERE

    return d_loss


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)

    # YOUR CODE STARTS HERE
    x_generated = g(z)
    g_loss = -d(x_generated).mean()
    # YOUR CODE ENDS HERE

    return g_loss
