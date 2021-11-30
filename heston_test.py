import heston_sim
import math
import torch

def examples(beispiel: str, batch_size):

    if beispiel=="compfin1":
        v0 = torch.tensor([0.010201]).repeat(batch_size)  # initial variance
        s0 = torch.tensor([100]).repeat(batch_size)  # initial stock price
        K = torch.tensor([100]).repeat(batch_size)  # strike price
        t = torch.tensor([1]).repeat(batch_size)  # time end point
        mu = torch.tensor([0.0319]).repeat(batch_size)  # drift
        r = mu  # because of risk neutral measure..
        kappa = torch.tensor([6.21]).repeat(batch_size)  # mean reversion speed of variance
        sigma = torch.tensor([0.61]).repeat(batch_size)  # volatility of variance
        rho = torch.tensor([-0.7]).repeat(batch_size) # correlation of the brownian motions
        theta = torch.tensor([0.019]).repeat(batch_size)  # mean reversion level of variance
        lambd = 0  # only in exakt price: price of volatility risk
        reference_price = 6.8061

    elif beispiel=="compfin2":
        v0 = torch.tensor([0.09]).repeat(batch_size)  # initial variance
        s0 = torch.tensor([100]).repeat(batch_size)  # initial stock price
        K = torch.tensor([100]).repeat(batch_size)  # strike price
        t = torch.tensor([5]).repeat(batch_size)  # time end point
        mu = torch.tensor([0.05]).repeat(batch_size)  # drift
        r = mu  # because of risk neutral measure..
        kappa = torch.tensor([2]).repeat(batch_size)  # mean reversion speed of variance
        sigma = torch.tensor([1]).repeat(batch_size)  # volatility of variance
        rho = torch.tensor([-0.3]).repeat(batch_size)  # correlation of the brownian motions
        theta = torch.tensor([0.09]).repeat(batch_size)  # mean reversion level of variance
        lambd = 0  # only in exakt price: price of volatility risk
        reference_price = 34.9998

    elif beispiel=="roah1":
        v0 = torch.tensor([0.05]).repeat(batch_size)  # initial variance
        s0 = torch.tensor([100]).repeat(batch_size)  # initial stock price
        K = torch.tensor([100]).repeat(batch_size)  # strike price
        t = torch.tensor([0.5]).repeat(batch_size)  # time end point
        mu = torch.tensor([0.03]).repeat(batch_size)  # drift
        r = mu  # because of risk neutral measure..
        kappa = torch.tensor([5]).repeat(batch_size)  # mean reversion speed of variance
        sigma = torch.tensor([0.5]).repeat(batch_size)  # volatility of variance
        rho = torch.tensor([-0.8]).repeat(batch_size)  # correlation of the brownian motions
        theta = torch.tensor([0.05]).repeat(batch_size)  # mean reversion level of variance
        lambd = 0  # only in exakt price: price of volatility risk
        reference_price = 6.8678

    elif beispiel == "hestontrap":
        v0 = torch.tensor([0.0175]).repeat(batch_size)  # initial variance
        s0 = torch.tensor([100]).repeat(batch_size)  # initial stock price
        K = torch.tensor([100]).repeat(batch_size)  # strike price
        t = torch.tensor([5]).repeat(batch_size)  # time end point
        mu = torch.tensor([0.00]).repeat(batch_size)  # drift
        r = mu  # because of risk neutral measure..
        kappa = torch.tensor([1.5768]).repeat(batch_size)  # mean reversion speed of variance
        sigma = torch.tensor([0.5751]).repeat(batch_size)  # volatility of variance
        rho = torch.tensor([-0.5711]).repeat(batch_size)  # correlation of the brownian motions
        theta = torch.tensor([0.0398]).repeat(batch_size)  # mean reversion level of variance
        lambd = 0  # only in exakt price: price of volatility risk
        reference_price = 9999999

    return v0, s0, K, t, r, mu, kappa, sigma, rho, theta, lambd, reference_price


if __name__ == '__main__':

    act_hestontrap = True
    batch_size = 1
    example = "compfin2"
    v0, s0, K, t, r, mu, kappa, sigma, rho, theta, lambd, reference_price = examples(example, batch_size=batch_size)
    print("Reference_price: {}".format(reference_price))

    c_price = heston_sim.heston_price(v0=v0.unsqueeze(1), s0=s0.unsqueeze(1), r=r.unsqueeze(1), kappa=kappa.unsqueeze(1),
                                      sigma=sigma.unsqueeze(1), rho=rho.unsqueeze(1), theta=theta.unsqueeze(1),
                                      lambd=lambd, K=K.unsqueeze(1), t=t.unsqueeze(1), act_hestontrap=act_hestontrap)
    print("Exact_Sol: {}".format(c_price.min().item()))

    batch_size = 1000000
    v0, s0, K, t, r, mu, kappa, sigma, rho, theta, lambd, reference_price = examples(example, batch_size=batch_size)
    asset_end = heston_sim.fte_euler(batch_size=batch_size, s0=s0, v0=v0, mu=mu,
                                    kappa=kappa, theta=theta, sigma=sigma,
                                    steps=100, t=t, rho=rho.unsqueeze(1), K=K, r=r)
    y_target = torch.mean(torch.exp(-r * t) * torch.nn.ReLU()(asset_end - K))
    print("FTE_Sol: {}".format(y_target.item()))

    limit_max = 1000
    k = math.log(K[0].item() / s0[0].item())
    from functools import partial
    cf_H_b_good = partial(heston_sim.cf_Heston_good, t=t[0].item(), v0=v0[0].item(), mu=mu[0].item(), theta=theta[0].item(), sigma=sigma[0].item(), kappa=kappa[0].item(), rho=rho[0].item())
    call = s0[0].item() * heston_sim.Q1(k, cf_H_b_good, limit_max) - K[0].item() * math.exp(-r[0].item() * t[0].item()) * heston_sim.Q2(k, cf_H_b_good, limit_max)
    print("Fourier Internet Sol: {}".format(call))
