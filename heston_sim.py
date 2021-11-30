import torch
import math
import matplotlib.pyplot as pt


def fte_euler(batch_size, s0, v0, mu, kappa, theta, sigma, steps, t, rho, K, r):

    s = torch.zeros(batch_size, steps+1)
    v = torch.zeros(batch_size, steps+1)
    v_tilde = torch.zeros(batch_size, steps+1)
    s[:, 0] = torch.log(s0)
    v[:, 0] = v0
    v_tilde[:, 0] = v[:, 0]

    # create correlation matrices
    cor_temp1 = torch.cat((torch.ones(batch_size).unsqueeze(1), rho), dim=1).unsqueeze(1)
    cor_temp2 = torch.cat((rho, torch.ones(batch_size).unsqueeze(1)), dim=1).unsqueeze(1)
    cor = torch.cat((cor_temp1, cor_temp2), dim=1)
    noise_mat = torch.linalg.cholesky(cor)

    for i in range(steps):
        corr_bm = torch.einsum("ik,ijk -> ij", torch.normal(mean=torch.zeros(batch_size, 2), std=torch.ones(batch_size, 2)), noise_mat)
        s[:, i + 1] = s[:, i] + \
                    (mu - 0.5 * v[:, i]) * t/steps + \
                    torch.sqrt(v[:, i]) * corr_bm[:, 0] * torch.sqrt(t/steps)
        v_tilde[:, i + 1] = v_tilde[:, i] - \
                            kappa * (t / steps) * (torch.nn.ReLU()(v_tilde[:, i]) - theta) + \
                            sigma * torch.sqrt(torch.nn.ReLU()(v_tilde[:, i])) * corr_bm[:, 1] * torch.sqrt(t / steps)
        v[:, i + 1] = torch.nn.ReLU()(v_tilde[:, i + 1])

    asset_end = torch.exp(s[:, -1])
    return asset_end



def multidim_fte_euler(d, batch_size, s0, v0, mu, kappa, theta, sigma, steps, t, rho):

    s = torch.zeros(batch_size, d, steps+1)
    v = torch.zeros(batch_size, d, steps+1)
    v_tilde = torch.zeros(batch_size, d, steps+1)
    s[:, :, 0] = torch.log(s0)
    v[:, :, 0] = v0
    v_tilde[:, :, 0] = v[:, :, 0]
    t = t.unsqueeze(1).repeat(1, d)

    cor_lower = torch.zeros(batch_size, 2 * d, 2 * d)
    count = 0
    for i in range(2*d):
        for j in range(i):
            cor_lower[:, i, j] = rho[:, count]
            count = count + 1
    cor_upper = torch.transpose(cor_lower, 1, 2)
    cor = cor_upper + cor_lower
    for i in range(2*d):
        cor[:, i, i] = 1

    # TODO: wird leider machmal singuläre Matrix, darum noch kümmern
    noise_mat = torch.linalg.cholesky(cor)

    for i in range(steps):
        corr_bm = torch.einsum("ik,ijk -> ij", torch.normal(mean=torch.zeros(batch_size, 2*d), std=torch.ones(batch_size, 2*d)), noise_mat)
        corr_bm_price = corr_bm.split(d, dim=1)[0]
        corr_bm_vola = corr_bm.split(d, dim=1)[1]
        s[:, :, i + 1] = s[:, :, i] + \
                    (mu - 0.5 * v[:, :, i]) * t/steps + \
                    torch.sqrt(v[:, :, i]) * corr_bm_price * torch.sqrt(t/steps)
        v_tilde[:, :, i + 1] = v_tilde[:, :, i] - \
                            kappa * (t / steps) * (torch.nn.ReLU()(v_tilde[:, :, i]) - theta) + \
                            sigma * torch.sqrt(torch.nn.ReLU()(v_tilde[:, :, i])) * corr_bm_vola * torch.sqrt(t / steps)
        v[:, :, i + 1] = torch.nn.ReLU()(v_tilde[:, :, i + 1])

    asset_end = torch.exp(s[:, :, -1])
    return asset_end


# "exact Heston option-pricing"
def heston_prob(v0, s0, K, kappa, theta, rho, sigma, phi, b, u, tau, r, act_hestontrap):
    x = torch.log(s0)
    a = kappa * theta
    d = torch.sqrt(torch.pow(sigma * rho * phi * 1j - b, 2) - sigma**2 * (2 * u * 1j * phi - torch.pow(phi, 2)))
    g = (b - rho * sigma * 1j * phi + d) / (b - rho * sigma * 1j * phi - d)

    if act_hestontrap:
        # with "little Heston trap fix by Albrecher"
        G = (1 - 1 / g * torch.exp(-d * tau)) / (1 - 1 / g)
        C = r * 1j * phi * tau + a / (sigma ** 2) * ((b - rho * sigma * 1j * phi - d) * tau - 2 * torch.log(G))
        D = (b - rho * sigma * 1j * phi - d) / (sigma ** 2) * (
                    (1 - torch.exp(-d * tau)) / (1 - 1 / g * torch.exp(-d * tau)))
    else:
        # without "little Heston trap fix by Albrecher"
        G = (1 - g * torch.exp(d * tau)) / (1 - g)
        C = r * 1j * phi * tau + a / (sigma**2) * ((b - rho * sigma * 1j * phi + d) * tau - 2*torch.log(G))
        D = (b - rho * sigma * 1j * phi + d) / (sigma**2) * ((1 - torch.exp(d * tau)) / (1 - g * torch.exp(d * tau)))#


    f = torch.exp(C + D * v0 + 1j * phi * x)
    y = torch.real(torch.exp(-1j * phi * torch.log(K)) * f / (1j * phi))
    return y


def heston_price(v0, s0, t, K, kappa, theta, rho, sigma, lambd, r, act_hestontrap):
    #integration grid
    int_grid_len = int(1e4)
    phi = torch.linspace(0.00000000001, 100, int_grid_len).unsqueeze(0).repeat(len(s0), 1)

    # bring input parameters in right dimension
    v0 = v0.repeat(1, int_grid_len)
    s0 = s0.repeat(1, int_grid_len)
    t = t.repeat(1, int_grid_len)
    K = K.repeat(1, int_grid_len)
    kappa = kappa.repeat(1, int_grid_len)
    theta = theta.repeat(1, int_grid_len)
    rho = rho.repeat(1, int_grid_len)
    sigma = sigma.repeat(1, int_grid_len)
    r = r.repeat(1, int_grid_len)

    b1 = kappa + lambd - rho*sigma
    b2 = kappa + lambd
    u1 = 0.5
    u2 = -0.5
    tau = t
    integrand1 = heston_prob(v0=v0, s0=s0, K=K, kappa=kappa,
                             theta=theta, rho=rho, sigma=sigma,
                             phi=phi, b=b1, u=u1, tau=tau, r=r, act_hestontrap=act_hestontrap)
    integrand2 = heston_prob(v0=v0, s0=s0, K=K, kappa=kappa,
                             theta=theta, rho=rho, sigma=sigma,
                             phi=phi, b=b2, u=u2, tau=tau, r=r, act_hestontrap=act_hestontrap)
    #integrand1 = integrand1[~torch.isnan(integrand1) & ~torch.isnan(integrand2)]
    #integrand2 = integrand2[0:integrand1.size()[0]]
    #pt.plot(integrand1[0].tolist())
    #pt.plot(integrand2.tolist())
    #pt.show()
    #phi = phi[0:integrand1.size()[0]]
    p1 = 0.5 + 1/math.pi * torch.trapz(integrand1, phi)
    p2 = 0.5 + 1/math.pi * torch.trapz(integrand2, phi)
    c_price = s0[:, 0] * p1 - K[:, 0] * torch.exp(-r[:, 0] * tau[:, 0]) * p2

    return c_price


import numpy as np
from scipy.integrate import quad

# fourier methode aus dem Netz
def cf_Heston_good(u, t, v0, mu, kappa, theta, sigma, rho):
    """
    Heston characteristic function as proposed by Schoutens (2004)
    """
    xi = kappa - sigma*rho*u*1j
    d = np.sqrt(xi**2 + sigma**2 * (u**2 + 1j*u))
    g1 = (xi+d)/(xi-d)
    g2 = 1/g1
    cf = np.exp( 1j*u*mu*t + (kappa*theta)/(sigma**2) * ( (xi-d)*t - 2*np.log( (1-g2*np.exp(-d*t))/(1-g2) ))\
              + (v0/sigma**2)*(xi-d) * (1-np.exp(-d*t))/(1-g2*np.exp(-d*t)) )
    return cf


def Q1(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the stock numeraire.
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( (np.exp(-u*k*1j) / (u*1j)) *
                                  cf(u-1j) / cf(-1.0000000000001j) )
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=2000 )[0]


def Q2(k, cf, right_lim):
    """
    P(X<k) - Probability to be in the money under the money market numeraire
    cf: characteristic function
    right_lim: right limit of integration
    """
    integrand = lambda u: np.real( np.exp(-u*k*1j) /(u*1j) * cf(u) )
    return 1/2 + 1/np.pi * quad(integrand, 1e-15, right_lim, limit=2000 )[0]