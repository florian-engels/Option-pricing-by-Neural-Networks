import time
import torch
import Heston.heston_sim as heston_sim
import network_class_skip_connections, network_class
import torch.optim as optim
import matplotlib.pyplot as plt


def simulate_data(d, batch_size, steps):

    # draw random variables
    v0 = torch.FloatTensor(batch_size, d).uniform_(0.03, 0.05)  # initial variance
    s0 = torch.FloatTensor(batch_size, d).uniform_(100, 102)  # initial stock price
    K = torch.FloatTensor(batch_size).uniform_(102, 104)  # strike price
    t = torch.FloatTensor(batch_size).uniform_(0.5, 1)  # time to maturity
    mu = torch.FloatTensor(batch_size, d).uniform_(-0.06, -0.02)  # drift term
    r = mu  # because of risk neutral measure..
    kappa = torch.FloatTensor(batch_size, d).uniform_(4, 6)  # mean reversion speed of variance
    sigma = torch.FloatTensor(batch_size, d).uniform_(0.05, 0.08)  # volatility of variance
    rho = torch.FloatTensor(batch_size, int((2*d-1)*(2*d)/2)).uniform_(-0.9, 0.9)  # correlation of brownian motions
    theta = torch.FloatTensor(batch_size, d).uniform_(0.06, 0.09)  # mean reversion level of variance

    # draw random correlation matrix -> besserer approach, aber dauert leider lange
    # rng = np.random.default_rng()
    # eigen = -1.5 * np.random.rand(1, 2*d) + 2
    # eigen = eigen / eigen.sum() * 2*d
    # eigen = tuple(map(tuple, eigen))[0]
    # cor_np = [scipy.stats.random_correlation.rvs(eigen, random_state=rng)]
    # for i in range(batch_size-1):
    #     cor_np = np.append(cor_np, [scipy.stats.random_correlation.rvs(eigen, random_state=rng)], axis=0)
    # corr = torch.from_numpy(cor_np)

    # Reihenfolge wichtig!!!
    x_input = torch.cat((v0, s0, K.unsqueeze(1), t.unsqueeze(1), mu, kappa, sigma, rho, theta), dim=1)

    # use of Full Truncation Euler
    y_target = heston_sim.multidim_fte_euler(d=d, batch_size=batch_size, steps=steps, v0=v0, s0=s0,
                                             t=t, mu=mu, kappa=kappa, sigma=sigma, rho=rho, theta=theta)
    # payoffs, then discounted
    # TODO: hier muss noch mit passendem mu dann abgezinst werden
    y_target = torch.nn.ReLU()(K - torch.mean(y_target, dim=1))  # torch.exp(-r * t) *

    return x_input, y_target



def validation_mc(d, M, batch_size, mc_samples):

    # draw random variables
    v0 = torch.FloatTensor(batch_size, d).uniform_(0.03, 0.05)  # initial variance
    s0 = torch.FloatTensor(batch_size, d).uniform_(100, 102)  # initial stock price
    K = torch.FloatTensor(batch_size).uniform_(102, 104)  # strike price
    t = torch.FloatTensor(batch_size).uniform_(0.5, 1)  # time to maturity
    mu = torch.FloatTensor(batch_size, d).uniform_(-0.06, -0.02)  # drift term
    r = mu  # because of risk neutral measure..
    kappa = torch.FloatTensor(batch_size, d).uniform_(4, 6)  # mean reversion speed of variance
    sigma = torch.FloatTensor(batch_size, d).uniform_(0.05, 0.08)  # volatility of variance
    rho = torch.FloatTensor(batch_size, int((2*d-1)*(2*d)/2)).uniform_(-0.1, 0.1)  # correlation of brownian motions
    theta = torch.FloatTensor(batch_size, d).uniform_(0.06, 0.09)  # mean reversion level of variance

    # Reihenfolge wichtig!!!
    x_input = torch.cat((v0, s0, K.unsqueeze(1), t.unsqueeze(1), mu, kappa, sigma, rho, theta), dim=1)

    # erweitere jede einzelne Batch um "mc_samples" um mittels MC den fairen Preis zu berechnen
    y_target = []
    for i in range(batch_size):
        if i % 50 == 0:
            print("Validation at batch: {}".format(i))
        s0_batch = s0[i, :].unsqueeze(0).expand(mc_samples, -1)
        v0_batch = v0[i, :].unsqueeze(0).expand(mc_samples, -1)
        K_batch = K[i].expand(mc_samples)
        t_batch = t[i].expand(mc_samples)
        mu_batch = mu[i, :].unsqueeze(0).expand(mc_samples, -1)
        kappa_batch = kappa[i, :].unsqueeze(0).expand(mc_samples, -1)
        sigma_batch = sigma[i, :].unsqueeze(0).expand(mc_samples, -1)
        rho_batch = rho[i, :].unsqueeze(0).expand(mc_samples, -1)
        theta_batch = theta[i, :].unsqueeze(0).expand(mc_samples, -1)

        # Simuliere Endzeitpunkte der SDE
        y_target_batch = heston_sim.multidim_fte_euler(d=d, batch_size=mc_samples, steps=M, v0=v0_batch, s0=s0_batch,
                                                       t=t_batch, mu=mu_batch, kappa=kappa_batch,
                                                       sigma=sigma_batch, rho=rho_batch, theta=theta_batch)

        y_target_batch = torch.nn.ReLU()(K_batch - torch.mean(y_target_batch, dim=1))
        y_target.append(torch.mean(y_target_batch))

    y_target = torch.stack(y_target)

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
    skip_enabled = True

    # Chargengroeße
    batch_size = 2 ** 15
    # number of steps by ft-euler simulation
    steps = 100
    # amplyfying factor
    q = 5
    # overall multilevel architecture levels
    overall_levels = 4
    # dimensions (meaning the number of different price processes)
    d = 3

    x_input, y_target = simulate_data(d=d, batch_size=batch_size, steps=steps)

    # create instance of Net Class
    if skip_enabled:
        net = network_class_skip_connections.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    else:
        net = network_class.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    # create your optimizer and scheduler
    optimizer = optim.AdamW(params=net.parameters(), lr=0.001, weight_decay=0.01)
    # create loss function
    criterion = torch.nn.MSELoss()

    epochs = 4000
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                               milestones=[1000, 2000, 3000, 4000, 5000, 6000, 7000], gamma=0.4)
    loss_graph = [0] * epochs
    for i in range(epochs):
        optimizer.zero_grad()
        out = net(x_input)
        loss = criterion(out, y_target.unsqueeze(dim=1))
        loss_graph[i] = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        x_input, y_target = simulate_data(d=d, batch_size=batch_size, steps=steps)
        #x_input, y_target = black_scholes_sim.testing(batch_size=batch_size, d=3, steps=25)
        if (i % 100) == 0:
            print("Epoche: {}".format(i))
            torch.save(net.state_dict(), "models/model_weights_zwischenspeicher")

    print("Last Learning Rate {}".format(scheduler.get_last_lr()[0]))

    # saving model
    torch.save(net.state_dict(), "models/model_weights_hestonmultidim_2dim_4000epochs_100steps")
    # loading model
    net = load_model(skip=skip_enabled, path="models/model_weights_hestonmultidim_2dim_4000epochs_100steps")

    ### Validation ###
    # MC validation of Model
    batch_size_validation = 2**12
    x_input_test, validation_price = validation_mc(d=d, M=100,
                                                   batch_size=batch_size_validation,
                                                   mc_samples=2**14)

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

