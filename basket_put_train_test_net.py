import torch
import black_scholes_sim
import network_class
import network_class_skip_connections
import torch.optim as optim
import matplotlib.pyplot as plt
import time


def simulate_data(batch_size, d, M, one_step_euler):
    # drawing random values
    x0 = torch.FloatTensor(batch_size, d).uniform_(9, 10)
    time = torch.FloatTensor(batch_size).uniform_(0, 1)
    gamma_sigma = torch.FloatTensor(batch_size, d).uniform_(0.1, 0.6)
    gamma_phi = torch.FloatTensor(batch_size).uniform_(10, 12)
    gamma_mu = torch.FloatTensor(batch_size, d).uniform_(0.1, 0.6)

    # TODO: Korrelationen so häufig sehr niedrig, bzw auch häufig 0 und nie negativ
    # sample correlation-matrix
    a = torch.rand(batch_size, d, d)
    a = torch.matmul(a, torch.transpose(a, 1, 2))
    a = (a - torch.min(a)) / (torch.max(a) - torch.min(a))
    cor = a - torch.triu(torch.tril(a)) + torch.eye(d, d).repeat(batch_size, 1, 1)
    # transform correlations to "input-format" to give it to the NN
    correlations = torch.stack((cor[:, 1, 0], cor[:, 2, 0], cor[:, 2, 1]), dim=1)
    # TODO: um variabel in den Dimensionen zu bleiben einen Ansatz dieser Art waehlen. Aber problematisch manchmal wenn Wert nahe 0.
    #correlations = (cor - torch.triu(cor))[cor - torch.triu(cor) != 0].reshape(batch_size, d)

    x_input = torch.cat((x0, time.unsqueeze(1), gamma_sigma.flatten(start_dim=1),
                         gamma_mu.flatten(start_dim=1), gamma_phi.unsqueeze(1), correlations), dim=1)

    # Simuliere Endzeitpunkte der SDE
    y_target = black_scholes_sim.euler_multidim(x=x0,
                                                t=time,
                                                gamma_sigma=gamma_sigma,
                                                gamma_mu=gamma_mu,
                                                cor=cor,
                                                batch_size=batch_size,
                                                d=d,
                                                M=M,
                                                one_step_euler=one_step_euler)
    # Payoff betrachtet nur Endzeitpunkt

    y_target = torch.nn.ReLU()(gamma_phi - torch.mean(y_target, dim=1))

    return x_input, y_target


def validation_mc(d, M, batch_size, mc_samples, one_step_euler):

    x0 = torch.FloatTensor(batch_size, d).uniform_(9, 10)
    time = torch.FloatTensor(batch_size).uniform_(0, 1)
    gamma_sigma = torch.FloatTensor(batch_size, d).uniform_(0.1, 0.6)
    gamma_phi = torch.FloatTensor(batch_size).uniform_(10, 12)
    gamma_mu = torch.FloatTensor(batch_size, d).uniform_(0.1, 0.6)

    # sample correlation-matrix
    a = torch.rand(batch_size, d, d)
    a_sym = torch.matmul(a, torch.transpose(a, 1, 2))
    a_sym = (a_sym - torch.min(a_sym)) / (torch.max(a_sym) - torch.min(a_sym))
    cor = a_sym - torch.triu(torch.tril(a_sym)) + torch.eye(d, d).repeat(batch_size, 1, 1)
    # transform correlations to "input-format" to give it to the NN
    correlations = (cor - torch.triu(cor))[cor - torch.triu(cor) != 0].reshape(batch_size, d)

    x_input = torch.cat((x0, time.unsqueeze(1), gamma_sigma.flatten(start_dim=1), gamma_mu.flatten(start_dim=1),
                         gamma_phi.unsqueeze(1), correlations), dim=1)

    # erweitere jedes einzelne Batch um "mc_samples" um mittels MC den fairen Preis zu berechnen
    y_target = []
    for i in range(batch_size):
        if i % 500 == 0:
            print("Validation at batch: {}".format(i))
        x0_batch = x0[i, :].unsqueeze(0).expand(mc_samples, -1)
        time_batch = time[i].expand(mc_samples)
        gamma_sigma_batch = gamma_sigma[i, :].unsqueeze(0).expand(mc_samples, -1)
        gamma_mu_batch = gamma_mu[i, :].unsqueeze(0).expand(mc_samples, -1)
        gamma_phi_batch = gamma_phi[i]
        cor_batch = cor[i, :, :].unsqueeze(0).expand(mc_samples, -1, -1)

        # Simuliere Endzeitpunkte der SDE
        y_target_batch = black_scholes_sim.euler_multidim(x=x0_batch,
                                                          t=time_batch,
                                                          gamma_sigma=gamma_sigma_batch,
                                                          gamma_mu=gamma_mu_batch,
                                                          batch_size=mc_samples,
                                                          cor=cor_batch,
                                                          d=d,
                                                          M=M,
                                                          one_step_euler=one_step_euler)

        y_target_batch = torch.nn.ReLU()(gamma_phi_batch - torch.mean(y_target_batch, dim=1))
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
    skip_enabled = False
    # simulate data with one-step solution (True) or via Euler (False)
    one_step_euler = True
    # Chargengroeße
    batch_size = 2 ** 17
    # amplyfying factor
    q = 5
    # overall multilevel architecture levels
    overall_levels = 4

    x_input, y_target = simulate_data(batch_size=batch_size, d=3, M=25, one_step_euler=one_step_euler)

    # create instance of Net Class
    if skip_enabled:
        net = network_class_skip_connections.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    else:
        net = network_class.Multilevel_Net(dim_in=x_input.size()[1], q=q, Level=overall_levels)
    # create your optimizer and scheduler
    optimizer = optim.AdamW(params=net.parameters(), lr=0.001, weight_decay=0.01)
    # create loss function
    criterion = torch.nn.MSELoss()

    epochs = 24000
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                               milestones=[4000, 8000, 12000, 16000, 20000, 24000], gamma=0.4)
    loss_graph = [0] * epochs
    for i in range(epochs):
        optimizer.zero_grad()
        out = net(x_input)
        loss = criterion(out, y_target.unsqueeze(dim=1))
        loss_graph[i] = loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
        x_input, y_target = simulate_data(batch_size=batch_size, M=25, d=3, one_step_euler=one_step_euler)
        #x_input, y_target = black_scholes_sim.testing(batch_size=batch_size, d=3, steps=25)
        if (i % 100) == 0:
            print("Epoche: {}".format(i))
        if ((i+1) % 4000) == 0:
            torch.save(net.state_dict(), "models/model_{}epochs_basket_skipdisabled_2hoch17batch".format(i+1))
            print("--- %s seconds ---" % (time.time() - start_time))

    print("Last Learning Rate {}".format(scheduler.get_last_lr()[0]))

    # saving model
    torch.save(net.state_dict(), "models/model_24000epochs_basket_skipdisabled_2hoch17batch")
    # loading model
    #net = load_model(skip=skip_enabled, path="models/model_weights_3erBasket_Epochen8000_skipenabled_batchsize2hoch17")

    ### Validation ###
    # MC validation of Model
    batch_size_validation = 2**10
    x_input_test, validation_price = validation_mc(d=3, M=25,
                                                   batch_size=batch_size_validation,
                                                   mc_samples=2**10,
                                                   one_step_euler=one_step_euler)
    #x_input, evaluation_price = black_scholes_sim.testing(batch_size=batch_size, d=3, steps=2**15)

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
    #fig = plt.plot(loss_graph)
    #plt.show()

