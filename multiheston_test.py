import torch
import heston_sim

# d = 2
# batch_size = 2**14
# steps = 10000
#
# v0 = torch.FloatTensor([0.05, 0.05]).expand(batch_size, d)  # initial variance
# s0 = torch.FloatTensor([1, 1]).expand(batch_size, d)   # initial stock price
# K = torch.FloatTensor([1]).expand(batch_size)   # strike price
# t = torch.FloatTensor([1]).expand(batch_size)   # time to maturity
# mu = torch.FloatTensor([0.05, 0.05]).expand(batch_size, d)  # drift term
# r = mu  # because of risk neutral measure..
# kappa = torch.FloatTensor([2.0, 2.0]).expand(batch_size, d)  # mean reversion speed of variance
# sigma = torch.FloatTensor([0.3, 0.3]).expand(batch_size, d)  # volatility of variance
# rho = torch.FloatTensor([0.2,
#                          -0.05, 0,
#                          0, -0.05, 0]).expand(batch_size, int((2*d-1)*(2*d)/2))  # correlation of brownian motions
# theta = torch.FloatTensor([0.06, 0.06]).expand(batch_size, d)  # mean reversion level of variance


d = 3
batch_size = 2**14
steps = 10000

v0 = torch.FloatTensor([0.05, 0.05, 0.05]).expand(batch_size, d)  # initial variance
s0 = torch.FloatTensor([1, 1, 1]).expand(batch_size, d)   # initial stock price
K = torch.FloatTensor([1]).expand(batch_size)   # strike price
t = torch.FloatTensor([1]).expand(batch_size)   # time to maturity
mu = torch.FloatTensor([0.05, 0.05, 0.05]).expand(batch_size, d)  # drift term
r = mu  # because of risk neutral measure..
kappa = torch.FloatTensor([2.0, 2.0, 2.0]).expand(batch_size, d)  # mean reversion speed of variance
sigma = torch.FloatTensor([0.3, 0.3, 0.3]).expand(batch_size, d)  # volatility of variance
rho = torch.FloatTensor([0.2,
                         0.2, 0.2,
                         -0.05, 0, 0,
                         0, -0.05, 0, 0,
                         0, 0, -0.05, 0, 0]).expand(batch_size, int((2*d-1)*(2*d)/2))  # correlation of brownian motions
theta = torch.FloatTensor([0.06, 0.06, 0.06]).expand(batch_size, d)  # mean reversion level of variance


# Reihenfolge wichtig!!!
x_input = torch.cat((v0, s0, K.unsqueeze(1), t.unsqueeze(1), mu, kappa, sigma, rho, theta), dim=1)
# use of Full Truncation Euler
y_target = heston_sim.multidim_fte_euler(d=d, batch_size=batch_size, steps=steps, v0=v0, s0=s0,
                                         t=t, mu=mu, kappa=kappa, sigma=sigma, rho=rho, theta=theta)
# payoffs, then discounted
y_target = torch.mean(torch.exp(-t[0] * r[0, 0]) * torch.nn.ReLU()(torch.mean(y_target, dim=1) - K))
print(y_target)