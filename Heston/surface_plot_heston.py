import matplotlib.pyplot as plt
import numpy as np
import torch



def get_points_for_plot_at_fixed_time(t_min, t_max,
                                      s_min, s_max,
                                      v_fixed,
                                      mu_fixed,
                                      sigma_fixed,
                                      kappa_fixed,
                                      rho_fixed,
                                      theta_fixed,
                                      K_fixed,
                                      n_plot):
    """ Get the spacial and normalised values for surface plots
    at fixed time and parameter, varying both asset prices.
    """
    s_plot = torch.linspace(s_min, s_max, n_plot)#.reshape(-1, 1)
    t_plot = torch.linspace(t_min, t_max, n_plot)#.reshape(-1, 1)
    [s_plot_mesh, t_plot_mesh] = torch.meshgrid(s_plot, t_plot, indexing='ij')

    s_plot_mesh_inp = s_plot_mesh.reshape(-1, 1)

    t_plot_mesh_inp = t_plot_mesh.reshape(-1, 1)

    v_fixed = torch.tensor([v_fixed]).repeat(n_plot ** 2, 1)
    mu_fixed = torch.tensor([mu_fixed]).repeat(n_plot ** 2, 1)
    kappa_fixed = torch.tensor([kappa_fixed]).repeat(n_plot ** 2, 1)
    theta_fixed = torch.tensor([theta_fixed]).repeat(n_plot ** 2, 1)
    sigma_fixed = torch.tensor([sigma_fixed]).repeat(n_plot ** 2, 1)
    rho_fixed = torch.tensor([rho_fixed]).repeat(n_plot ** 2, 1)
    K_fixed = torch.tensor([K_fixed]).repeat(n_plot ** 2, 1)

    x_input = torch.cat((v_fixed, s_plot_mesh_inp, K_fixed,
                         t_plot_mesh_inp, mu_fixed, kappa_fixed,
                         sigma_fixed, rho_fixed, theta_fixed), dim=1)

    #
    return s_plot_mesh, t_plot_mesh, x_input



def plot_surface(net):

    nr_samples_surface_plot = 21

    # fixed parameters and parameter ranges
    v0 = 0.010201  # initial variance
    s_min = 100  # initial stock price
    s_max = 102  # initial stock price
    K = 103  # strike price
    t_min = 0.5  # time end point
    t_max = 1  # time end point
    mu = 0.0319  # drift
    kappa = 6.21  # mean reversion speed of variance
    sigma = 0.61  # volatility of variance
    rho = -0.7  # correlation of the brownian motions
    theta = 0.019  # mean reversion level of variance

    s_plot_mesh, t_plot_mesh, x_input = get_points_for_plot_at_fixed_time(t_min=t_min, t_max=t_max,
                                                                          s_min=s_min, s_max=s_max,
                                                                          v_fixed=v0, mu_fixed=mu,
                                                                          sigma_fixed=sigma, kappa_fixed=kappa,
                                                                          rho_fixed=rho, theta_fixed=theta,
                                                                          K_fixed=K, n_plot=nr_samples_surface_plot)

    net.eval()
    nn_price = net.predict(x_input).squeeze(1).reshape(nr_samples_surface_plot, nr_samples_surface_plot)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(s_plot_mesh.detach().numpy(), t_plot_mesh.detach().numpy(), nn_price.detach().numpy(), cmap='viridis')
    ax.set_title('Net Solution')
    ax.set_xlabel('$s$')
    ax.set_ylabel('$t$')
    plt.show()