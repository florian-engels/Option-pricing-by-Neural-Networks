import torch
import math
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


# Euler-Maruyama Simulation
def euler_bs(x, t, gamma_sigma, batch_size):
    """
    Euler-Maruyama Scheme to Simulate Path of SDE
    :param x: initial value tensor
    :param t: end time tensor
    :param M: #steps
    :param gamma_sigma: volatility tensor
    :param batch_size: tensor size
    :return: value of SDE at time t
    """
    x = x.squeeze(dim=1)
    t = t.squeeze(dim=1)
    gamma_sigma = gamma_sigma.squeeze(dim=1)
    S = x * torch.exp(-0.5 * t * gamma_sigma**2 + gamma_sigma *
                      torch.sqrt(t)*torch.normal(mean=torch.zeros(batch_size), std=torch.ones(batch_size)))
    return S


def true_euler_for_check(x, t, gamma_sigma, batch_size, steps):
    S = torch.zeros(batch_size, steps+1)
    S[:, 0] = x
    for i in range(steps):
        S[:, i + 1] = S[:, i] + S[:, i] * gamma_sigma * torch.normal(mean=torch.zeros(batch_size), std=torch.sqrt(t/steps))
    return S[:, -1]


def bs_closed_payoff(x, t, gamma_sigma, gamma_phi):
    """
    function to compute closed BS price for european put
    :param x: initial value
    :param t: execution time
    :param gamma_sigma: volatility
    :param gamma_phi: strike price
    :return: fair price of put option
    """
    sigma_sqrtt = gamma_sigma * torch.sqrt(t)
    h_gamma = -(1/sigma_sqrtt)*(torch.log(x / gamma_phi) + (t * (gamma_sigma ** 2))/2)
    value = gamma_phi * n_dist(h_gamma + sigma_sqrtt) - x * n_dist(h_gamma)

    return value


def n_dist(x):
    """
    Cumulative distribution function of the standard normal distribution.
    """
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def euler_multidim(x, t, batch_size, M, gamma_sigma, cor, gamma_mu, d, one_step_euler):
    # EULER für Basket Option
    noise_mat = torch.linalg.cholesky(cor)

    # using only one step to approximate solution
    if one_step_euler:
        t = t.unsqueeze(1)
        corr_bm = torch.einsum("ik,ijk -> ij", torch.normal(mean=torch.zeros(batch_size, 3), std=torch.ones(batch_size, 3)), noise_mat)
        S = x * torch.exp((gamma_mu - 0.5 * gamma_sigma ** 2) * t + gamma_sigma * torch.sqrt(t) * corr_bm)
    # using M steps to calculate solution
    else:
        # time steps in richtige Form bringen um korrelierte ZV erstellen zu können
        #TODO: mit repeat variabel in der Dimension
        stacked_time = torch.stack((torch.sqrt(t / M), torch.sqrt(t / M), torch.sqrt(t / M)))
        S = torch.zeros(batch_size, d, M+1)
        S[:, :, 0] = x
        for m in range(M):
            corr_bm = torch.matmul(noise_mat, torch.normal(mean=torch.zeros(d, batch_size), std=stacked_time)).squeeze(0)
            for j in range(d):
                S[:, j, m+1] = S[:, j, m] + S[:, j, m] * gamma_mu[:, j] * (t/M) +\
                        S[:, j, m] * gamma_sigma[:, j] * corr_bm[j]
        S = S[:, :, M]

    return S

# method of code from paper
def testing(batch_size, d, steps):
    x = torch.FloatTensor(batch_size, d).uniform_(9, 10)
    time = torch.FloatTensor(batch_size).uniform_(0, 1)
    gamma_sigma = torch.FloatTensor(batch_size, d, d, d + 1).uniform_(0.1, 0.6)
    gamma_phi = torch.FloatTensor(batch_size).uniform_(10, 12)
    gamma_mu = torch.FloatTensor(batch_size, d, d + 1).uniform_(0.1, 0.6)

    x_input = torch.cat((x, time.unsqueeze(1), gamma_sigma.flatten(start_dim=1), gamma_mu.flatten(start_dim=1), gamma_phi.unsqueeze(1)), dim=1)

    steplen = (time / steps).flatten()
    std = torch.sqrt(steplen)
    outputs = x.clone()
    for _ in range(steps):
        dw = (
                torch.randn(
                    d, batch_size, dtype=x.dtype, device=x.device
                )
                * std
        )
        sigma_x = (
                torch.einsum("iklj, il -> ikj", gamma_sigma[:, :, :, :d], outputs)
                + gamma_sigma[:, :, :, d]
        )
        mu_x = (
                torch.einsum("ikj, ij -> ik", gamma_mu[:, :, :d], outputs)
                + gamma_mu[:, :, d]
        )
        outputs += torch.einsum("ij, i -> ij", mu_x, steplen) + torch.einsum(
            "ijk, ki -> ij", sigma_x, dw
        )

    return x_input, torch.nn.ReLU()(gamma_phi - outputs.mean(dim=1))
