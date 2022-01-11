import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

import Heston.heston_sim as heston_sim
import network_class_skip_connections
import network_class
import Heston.surface_plot_heston as surface_plot_heston


def simulate_data(batch_size, steps):

    # draw random variables
    v0 = torch.FloatTensor(batch_size).uniform_(0.03, 0.05)  # initial variance
    s0 = torch.FloatTensor(batch_size).uniform_(100, 102)  # initial stock price
    K = torch.FloatTensor(batch_size).uniform_(102, 104)  # strike price
    t = torch.FloatTensor(batch_size).uniform_(0.5, 1)  # time to maturity
    mu = torch.FloatTensor(batch_size).uniform_(0.02, 0.06)  # drift term
    r = mu  # because of risk neutral measure..
    kappa = torch.FloatTensor(batch_size).uniform_(4, 6)  # mean reversion speed of variance
    sigma = torch.FloatTensor(batch_size).uniform_(0.5, 0.8)  # volatility of variance
    rho = torch.FloatTensor(batch_size).uniform_(-0.8, -0.5).unsqueeze(1)  # correlation of brownian motions
    theta = torch.FloatTensor(batch_size).uniform_(0.01, 0.04)  # mean reversion level of variance

    # Reihenfolge wichtig!!!
    x_input = torch.cat((v0.unsqueeze(1), s0.unsqueeze(1), K.unsqueeze(1),
                         t.unsqueeze(1), mu.unsqueeze(1), kappa.unsqueeze(1),
                         sigma.unsqueeze(1), rho, theta.unsqueeze(1)), dim=1)

    # use of Full Truncation Euler
    y_target = heston_sim.fte_euler(batch_size=batch_size, steps=steps, v0=v0, s0=s0, K=K, t=t, mu=mu, r=r,
                                    kappa=kappa, sigma=sigma, rho=rho, theta=theta)
    # payoffs, then discounted
    y_target = torch.exp(-r * t) * torch.nn.ReLU()(y_target - K)

    return x_input, y_target


def validation_mc(batch_size):

    # draw random variables
    v0 = torch.FloatTensor(batch_size).uniform_(0.03, 0.05).unsqueeze(1)  # initial variance
    s0 = torch.FloatTensor(batch_size).uniform_(100, 102).unsqueeze(1)  # initial stock price
    K = torch.FloatTensor(batch_size).uniform_(102, 104).unsqueeze(1)  # strike price
    t = torch.FloatTensor(batch_size).uniform_(0.5, 1).unsqueeze(1)  # time to maturity
    mu = torch.FloatTensor(batch_size).uniform_(-0.06, -0.02).unsqueeze(1)  # drift term
    r = mu  # because of risk neutral measure..
    kappa = torch.FloatTensor(batch_size).uniform_(4, 6).unsqueeze(1)  # mean reversion speed of variance
    sigma = torch.FloatTensor(batch_size).uniform_(0.05, 0.08).unsqueeze(1)  # volatility of variance
    rho = torch.FloatTensor(batch_size).uniform_(-0.8, -0.5).unsqueeze(1)  # correlation of brownian motions
    theta = torch.FloatTensor(batch_size).uniform_(0.06, 0.09).unsqueeze(1)  # mean reversion level of variance

    # Reihenfolge wichtig!!!
    x_input = torch.cat((v0, s0, K, t, mu, kappa, sigma, rho, theta), dim=1)

    y_target = heston_sim.heston_price(v0=v0, s0=s0, K=K, kappa=kappa, theta=theta, rho=rho,
                                       sigma=sigma, r=r, t=t, lambd=0, act_hestontrap=True)

    return x_input, y_target


def load_model(skip, path):

    if skip:
        network = network_class_skip_connections.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    else:
        network = network_class.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)

    network.load_state_dict(torch.load(path))

    return network


if __name__ == '__main__':
    start_time = time.time()

    # Enable skip connections
    skip_enabled = False

    # Chargengroeße
    batch_size = 2 ** 17
    # number of steps by ft-euler simulation
    steps = 100
    # amplyfying factor
    q = 5
    # overall multilevel architecture levels
    overall_levels = 4

    x_input, y_target = simulate_data(batch_size=batch_size, steps=steps)

    # create instance of Net Class
    if skip_enabled:
        net = network_class_skip_connections.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    else:
        net = network_class.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    # create your optimizer and scheduler
    optimizer = optim.AdamW(params=net.parameters(), lr=0.01, weight_decay=0.01)
    # create loss function
    criterion = torch.nn.MSELoss()

    epochs = 30000
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                               milestones=[4000, 8000, 12000, 16000, 20000, 24000],
                                               gamma=0.25)
    loss_graph = [0] * epochs
    for i in range(epochs):
        optimizer.zero_grad()
        out = net(x_input)
        loss = criterion(out, y_target.unsqueeze(dim=1))
        loss_graph[i] = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        x_input, y_target = simulate_data(batch_size=batch_size, steps=steps)
        if (i % 100) == 0:
            print("Epoche: {}".format(i))
        if ((i+1) % 4000) == 0:
            torch.save(net.state_dict(), "models/model_new_{}epochs_heston_skipdisabled_100stepsFT_2hoch17batch".format(i+1))
            print("--- %s seconds ---" % (time.time() - start_time))

    print("Last Learning Rate {}".format(scheduler.get_last_lr()[0]))

    # saving model
    torch.save(net.state_dict(), "models/model_new_24000epochs_heston_skipdisabled_100stepsFT_2hoch17batch")
    # loading model
    net = load_model(skip=skip_enabled, path="Heston/models/model_new_30000epochs_heston_skipdisabled_100stepsFT_2hoch17batch")

    ### Validation ###
    # MC validation of Model
    # 2hoch10 fast maximum, mehr killt den laptop wegen zwischenspeicher
    batch_size_validation = 2 ** 11
    x_input_test, validation_price = validation_mc(batch_size=batch_size_validation)
    #  x_input, evaluation_price = black_scholes_sim.testing(batch_size=batch_size, d=3, steps=2**15)

    net.eval()
    nn_price = net.predict(x_input_test).squeeze(1)
    l1_error = torch.abs(nn_price - validation_price) / (1 + torch.abs(validation_price))
    l1_error = torch.sum(l1_error) / batch_size_validation
    varianz = torch.var(torch.abs(nn_price - validation_price) / (1 + torch.abs(validation_price)))

    print("--- %s seconds ---" % (time.time() - start_time))
    print("L1-Error: {}".format(l1_error))
    print("Std: {}".format(torch.sqrt(varianz)))

    plt.plot(nn_price[torch.sort(validation_price)[1]].tolist(), ".", markersize=0.7)
    plt.plot(torch.sort(validation_price)[0].tolist())
    plt.show()
    fig = plt.plot(loss_graph)
    plt.show()

    surface_plot_heston.plot_surface(net)

