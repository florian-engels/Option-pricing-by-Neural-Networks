import torch
import BlackScholes.black_scholes_sim as black_scholes_sim
import network_class
import network_class_skip_connections
import matplotlib.pyplot as plt
import time


def simulate_data(batch_size):
    # drawing random values
    x0 = torch.FloatTensor(batch_size).uniform_(9, 10).unsqueeze(1)
    time = torch.FloatTensor(batch_size).uniform_(0, 1).unsqueeze(1)
    gamma_sigma = torch.FloatTensor(batch_size).uniform_(0.1, 0.6).unsqueeze(1)
    gamma_phi = torch.FloatTensor(batch_size).uniform_(10, 12).unsqueeze(1)

    x_input = torch.cat((x0, time, gamma_sigma, gamma_phi), dim=1)

    # Euler mit gesamtem Pfad als Output
    y_target = black_scholes_sim.euler_bs(x=x0, t=time, gamma_sigma=gamma_sigma, batch_size=batch_size)
    # Payoff betrachtet nur Endzeitpunkt
    y_target = gamma_phi.squeeze(dim=1) - y_target
    y_target[y_target < 0] = 0

    return x_input, y_target


def load_model(skip, path):

    if skip:
        network = network_class_skip_connections.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    else:
        network = network_class.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)

    network.load_state_dict(torch.load(path))

    return network


if __name__ == '__main__':

    # Enable skip connections
    skip_enabled = False
    # Chargengroeße
    batch_size = 2 ** 16
    # amplyfying factor
    q = 5
    # overall multilevel architecture levels
    overall_levels = 4

    x_input, y_target = simulate_data(batch_size=batch_size)

    # loading model
    net = load_model(skip=skip_enabled, path="models/model_24000epochs_bs_skipdisabled_2hoch16batch")

    ### Validation ###
    # test Model against BS Price of closed form, but only change one variable
    x0 = torch.tensor([10.0]).repeat(batch_size).unsqueeze(1)
    #x0 = torch.arange(start=9.5, end=10.5, step=(10.5-9.5) / batch_size).unsqueeze(1)
    time = torch.tensor([1]).repeat(batch_size).unsqueeze(1)
    gamma_sigma = torch.arange(start=0.1, end=0.6, step=(0.6-0.1)/batch_size).unsqueeze(1)
    gamma_phi = torch.tensor([10.5]).repeat(batch_size).unsqueeze(1)

    x_input_test = torch.cat((x0, time, gamma_sigma, gamma_phi), dim=1)

    net.eval()
    nn_price = net.predict(x_input_test)
    validation_price = black_scholes_sim.bs_closed_payoff(x=x0, t=time,
                                                          gamma_sigma=gamma_sigma,
                                                          gamma_phi=gamma_phi)
    l1_error = (torch.abs(nn_price - validation_price) / (1 + torch.abs(validation_price)))
    nn_prices_for_plot = nn_price
    validation_prices_for_plot = validation_price

    # evaluation_batches = 1
    # for i in range(1, evaluation_batches):
    #     x0_test = torch.FloatTensor(batch_size).uniform_(9, 10).unsqueeze(1)
    #     time_test = torch.FloatTensor(batch_size).uniform_(0, 1).unsqueeze(1)
    #     gamma_sigma_test = torch.FloatTensor(batch_size).uniform_(0.1, 0.6).unsqueeze(1)
    #     gamma_phi_test = torch.FloatTensor(batch_size).uniform_(10, 12).unsqueeze(1)
    #     x_input_test = torch.cat((x0_test, time_test, gamma_sigma_test, gamma_phi_test), dim=1)
    #
    #     net.eval()
    #     nn_price = net.predict(x_input_test)
    #     validation_price = black_scholes_sim.bs_closed_payoff(x=x0_test, t=time_test,
    #                                                           gamma_sigma=gamma_sigma_test,
    #                                                           gamma_phi=gamma_phi_test)
    #     l1_error = torch.cat((l1_error, torch.abs(nn_price - validation_price) / (1 + torch.abs(validation_price))))
    #     nn_prices_for_plot = torch.cat((nn_prices_for_plot, nn_price))
    #     validation_prices_for_plot = torch.cat((validation_prices_for_plot, validation_price))

    l1_error_avg = torch.mean(l1_error)
    std = torch.sqrt(torch.var(l1_error))
    print("L1-Error: {}".format(l1_error_avg))
    print("Std: {}".format(std))

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # surf = ax.plot_surface(x0, gamma_sigma, nn_prices_for_plot.squeeze(1).tolist(), cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()

    plt.plot(gamma_sigma, nn_prices_for_plot.squeeze(1).tolist(), marker=".", markersize=0.7)
    plt.plot(gamma_sigma, validation_prices_for_plot.squeeze(1).tolist())
    plt.ylabel("option price")
    plt.xlabel("sigma")
    plt.show()

    plt.plot(gamma_sigma, l1_error.squeeze(1).tolist())
    plt.ylabel("l1-error of predicted option price")
    plt.xlabel("sigma")
    plt.show()