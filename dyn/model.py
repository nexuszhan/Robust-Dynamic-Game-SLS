import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class Model:
    def __init__(self):
        self.dt = None
        self.nx = None
        self.nu = None
        self.nw = None
        self.ni = None
        self.ni_f = None
        self.ode = None
        self.E_func = None

    def ddyn(self, x, u, h=0.05):
        ode = self.ode

        k_1 = ode(x, u)
        k_2 = ode(x + 0.5 * h * k_1, u)
        k_3 = ode(x + 0.5 * h * k_2, u)
        k_4 = ode(x + h * k_3, u)
        x_p = x + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
        return x_p

    def remove_constraints(self):
        # todo add check on constraint existence
        m = self

        # remove constraints
        m.G = np.zeros((0, m.nx + m.nu))
        m.g = np.zeros((0, 1))
        m.Gf = np.zeros((0, m.nx))
        m.gf = np.zeros((0, 1))
        m.ni = 0
        m.ni_f = 0

    def replace_constraints(self, x_max, u_max, x_min, u_min, x_max_f, x_min_f):
        self.g = np.hstack((x_max, u_max, -x_min, -u_min))
        self.gf = np.hstack((x_max_f, -x_min_f))

    def plot_nominal_trajectory(self, X, ax=None):
        """
        :param X: nominal trajectory
        :return: plot the nominal trajectory
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        x_max = self.g[0]
        # assume all constraints are the same

        # plot horizontal line
        # ax.axhline(y=-x_max, color='k')
        # ax.axhline(y=x_max, color='k')
        # time vector
        time = np.arange(0, X.shape[1]) * self.dt

        colors = plt.cm.viridis(np.linspace(0, 1, self.nx+2))
        for i in range(self.nx):
            # plot the nominal trajectory
            ax.plot(time, X[i, :], color=colors[i + 1])

        return ax

    def plot_input_nominal_trajectory(self, U, ax=None):
        """
        :param X: nominal trajectory
        :return: plot the nominal trajectory
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        u_max = self.g[4]

        # ax.axhline(y=-u_max, color='k')
        # ax.axhline(y=u_max, color='k')
        # U = U.reshape(-1, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, self.nu+2))
        time = np.arange(0, U.shape[1]) * self.dt
        for i in range(self.nu):
            ax.plot(time, U[i, :], color=colors[i +1])
        return ax

    def plot_tube(self, backoff, center, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        # transpose the matrices if they are not in the right shape: (nx, N+1)
        if not backoff.shape[0] == self.nx:
            backoff = backoff.T

        if not center.shape[0] == self.nx:
            center = center.T

        # at this stage, we don't plot the terminal set: remove the last time step
        center = center[:, :-1]
        time = np.arange(0, center.shape[1]) * self.dt

        # nx = 4
        nx = self.nx
        colors = plt.cm.viridis(np.linspace(0, 1, nx + 2))
        margin = 0.1
        for i in range(nx):
            lower_bound = center[i] - backoff[i] + margin
            upper_bound = center[i] + backoff[i] - margin
            ax.fill_between(time, lower_bound, upper_bound, color=colors[i + 1], alpha=0.5, label='Bounds')

        return ax
        # todo reduce the margin to 0

    def plot_input_tube(self, backoff, center, ax=None):

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        if not backoff.shape[0] == self.nu:
            backoff = backoff.T

        if not center.shape[0] == self.nu:
            center = center.T

        # center = center[:, :-1]
        time = np.arange(0, center.shape[1]) * self.dt

        # at this stage, we don't plot the terminal set: remove the last time step

        colors = plt.cm.viridis(np.linspace(0, 1, self.nu + 2))
        margin = 0.1
        for i in range(self.nu):
            lower_bound = center[i] - backoff[i] + margin
            upper_bound = center[i] + backoff[i] - margin
            ax.fill_between(time, lower_bound, upper_bound, color=colors[i + 1], alpha=0.5, label='Bounds')
        return ax