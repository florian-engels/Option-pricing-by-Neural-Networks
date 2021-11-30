import torch
import black_scholes_sim
import network_class
import network_class_skip_connections
import torch.optim as optim
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

    start_time = time.time()

    # Enable skip connections
    skip_enabled = False
    # Chargengroeße
    batch_size = 2 ** 16
    # amplyfying factor
    q = 5
    # overall multilevel architecture levels
    overall_levels = 4

    x_input, y_target = simulate_data(batch_size=batch_size)

    # create instance of Net Class
    if skip_enabled:
        net = network_class_skip_connections.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    else:
        net = network_class.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    # create your optimizer and scheduler
    optimizer = optim.AdamW(params=net.parameters(), lr=0.01, weight_decay=0.01)
    # create loss function
    criterion = torch.nn.MSELoss()

    epochs = 24000
    #scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.25)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4000, 8000, 12000, 16000, 20000, 24000], gamma=0.25)
    loss_graph = [0]*epochs
    for i in range(epochs):
        optimizer.zero_grad()
        out = net(x_input)
        loss = criterion(out, y_target.unsqueeze(dim=1))
        loss_graph[i] = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        x_input, y_target = simulate_data(batch_size=batch_size)
        if (i % 100) == 0:
            print("Epoche: {}".format(i))
        if ((i+1) % 4000) == 0:
            torch.save(net.state_dict(), "models/model_{}epochs_bs_skipdisabled_2hoch16batch".format(i+1))
            print("--- %s seconds ---" % (time.time() - start_time))


    print("Last Learning Rate {}".format(scheduler.get_last_lr()[0]))

    # saving model
    torch.save(net.state_dict(), "models/model_24000epochs_bs_skipdisabled_2hoch16batch")
    # loading model
    net = load_model(skip=skip_enabled, path="models/model_20000epochs_bs_skipdisabled_2hoch16batch")


    ### Validation ###
    # test Model against BS Price of closed form
    x0_test = torch.FloatTensor(batch_size).uniform_(9, 10).unsqueeze(1)
    time_test = torch.FloatTensor(batch_size).uniform_(0, 1).unsqueeze(1)
    gamma_sigma_test = torch.FloatTensor(batch_size).uniform_(0.1, 0.6).unsqueeze(1)
    gamma_phi_test = torch.FloatTensor(batch_size).uniform_(10, 12).unsqueeze(1)
    x_input_test = torch.cat((x0_test, time_test, gamma_sigma_test, gamma_phi_test), dim=1)

    net.eval()
    nn_price = net.predict(x_input_test)
    validation_price = black_scholes_sim.bs_closed_payoff(x=x0_test, t=time_test,
                                                          gamma_sigma=gamma_sigma_test,
                                                          gamma_phi=gamma_phi_test)
    l1_error = (torch.abs(nn_price - validation_price) / (1 + torch.abs(validation_price)))
    nn_prices_for_plot = nn_price
    validation_prices_for_plot = validation_price

    evaluation_batches = 100
    for i in range(1, evaluation_batches):
        x0_test = torch.FloatTensor(batch_size).uniform_(9, 10).unsqueeze(1)
        time_test = torch.FloatTensor(batch_size).uniform_(0, 1).unsqueeze(1)
        gamma_sigma_test = torch.FloatTensor(batch_size).uniform_(0.1, 0.6).unsqueeze(1)
        gamma_phi_test = torch.FloatTensor(batch_size).uniform_(10, 12).unsqueeze(1)
        x_input_test = torch.cat((x0_test, time_test, gamma_sigma_test, gamma_phi_test), dim=1)

        net.eval()
        nn_price = net.predict(x_input_test)
        validation_price = black_scholes_sim.bs_closed_payoff(x=x0_test, t=time_test,
                                                              gamma_sigma=gamma_sigma_test,
                                                              gamma_phi=gamma_phi_test)
        l1_error = torch.cat((l1_error, torch.abs(nn_price - validation_price) / (1 + torch.abs(validation_price))))
        nn_prices_for_plot = torch.cat((nn_prices_for_plot, nn_price))
        validation_prices_for_plot = torch.cat((validation_prices_for_plot, validation_price))

    l1_error_avg = torch.mean(l1_error)
    std = torch.sqrt(torch.var(l1_error))
    print("--- %s seconds ---" % (time.time() - start_time))
    print("L1-Error: {}".format(l1_error_avg))
    print("Std: {}".format(std))

    plt.plot(nn_prices_for_plot.squeeze(1)[torch.sort(validation_prices_for_plot.squeeze(1))[1]].tolist(), ".", markersize=0.7)
    plt.plot(torch.sort(validation_prices_for_plot.squeeze(1))[0].tolist())
    plt.show()
    fig = plt.plot(loss_graph)
    plt.show()
