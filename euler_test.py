import torch
import black_scholes_sim
import math
import matplotlib.pyplot as plt

def teste_eindim_euler():

    batch_size = 2**16

    gamma_sigma = (torch.ones(batch_size) * 0.1).unsqueeze(1)
    gamma_phi = (torch.ones(batch_size) * 10.0).unsqueeze(1)
    time = (torch.ones(batch_size) * 1).unsqueeze(1)
    x0 = (torch.ones(batch_size) * 10.0).unsqueeze(1)

    x_input = torch.cat((x0, time, gamma_sigma, gamma_phi), dim=1)

    sol = 0
    times = 1000
    for i in range(times):
        # Euler mit gesamtem Pfad als Output
        y_target = black_scholes_sim.euler_bs(x=x0, t=time, gamma_sigma=gamma_sigma, batch_size=batch_size)
        y_target = gamma_phi.squeeze(dim=1) - y_target
        y_target[y_target < 0] = 0
        sol = sol + torch.mean(y_target)

    print(sol/times)

    x0_test = torch.tensor(10.0)
    time_test = torch.tensor(1)
    gamma_sigma_test = torch.tensor(0.1)
    gamma_phi_test = torch.tensor(10.0)
    closed_price = black_scholes_sim.bs_closed_payoff(x=x0_test, t=time_test, gamma_sigma=gamma_sigma_test, gamma_phi=gamma_phi_test)
    print(closed_price)

    print("error: {}".format(sol/times-closed_price))
    return


def teste_true_euler():
    batch_size = 2 ** 18
    steps = 100

    x0 = torch.tensor([10.0]).repeat(batch_size)
    time = torch.tensor([1]).repeat(batch_size)
    gamma_sigma = torch.tensor([0.1]).repeat(batch_size)
    gamma_phi = torch.tensor([10.0]).repeat(batch_size)

    values = black_scholes_sim.true_euler_for_check(x=x0, t=time, gamma_sigma=gamma_sigma, batch_size=batch_size, steps=steps)
    price = torch.nn.ReLU()(gamma_phi - values)
    price = torch.mean(price)
    print("Euler Price: {}".format(price))

    x0_test = torch.tensor(10.0)
    time_test = torch.tensor(1)
    gamma_sigma_test = torch.tensor(0.1)
    gamma_phi_test = torch.tensor(10.0)
    closed_price = black_scholes_sim.bs_closed_payoff(x=x0_test, t=time_test, gamma_sigma=gamma_sigma_test,
                                                      gamma_phi=gamma_phi_test)
    print("Closed Price: {}".format(closed_price))
    return


def teste_multidim_euler():

    batch_size = 2**18
    M = 25
    d = 3
    one_step_euler = True

    #x0 = torch.FloatTensor(batch_size, d).uniform_(9, 10)
    #time = torch.FloatTensor(batch_size).uniform_(0, 1)
    #gamma_sigma = torch.FloatTensor(batch_size, d).uniform_(0.1, 0.6)
    #gamma_phi = torch.FloatTensor(batch_size).uniform_(10, 12)
    #gamma_mu = torch.FloatTensor(batch_size, d).uniform_(0.1, 0.6)

    x0 = torch.tensor([9.5]).repeat(batch_size, d)
    time = torch.tensor([0.5]).repeat(batch_size)
    gamma_sigma = torch.tensor([0.3]).repeat(batch_size, d)
    gamma_phi = torch.tensor([10.5]).repeat(batch_size)
    gamma_mu = torch.tensor([0.5]).repeat(batch_size, d)

    cor = torch.tensor([[[1, 0.5, 0.5],
                          [0.5, 1, 0.5],
                          [0.5, 0.5, 1]]])


    y_target = black_scholes_sim.euler_multidim(x=x0,
                                                t=time,
                                                gamma_sigma=gamma_sigma,
                                                gamma_mu=gamma_mu,
                                                cor=cor,
                                                batch_size=batch_size,
                                                d=d,
                                                M=M,
                                                one_step_euler=one_step_euler)

    #batch = 0
    #plt.plot(y_target[batch, 0, :])
    #plt.plot(y_target[batch, 1, :])
    #plt.plot(y_target[batch, 2, :])
    #plt.show()

    price = torch.nn.ReLU()(gamma_phi - torch.mean(y_target, dim=1))
    price = torch.mean(price)

    print("fair price: {}".format(torch.mean(price)))

    return


if __name__ == '__main__':

    #teste_eindim_euler()

    #teste_multidim_euler()

    teste_true_euler()
