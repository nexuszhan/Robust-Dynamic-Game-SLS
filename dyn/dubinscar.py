import numpy as np
from dyn.model import Model
import casadi as ca
import matplotlib.pyplot as plt

class DubinsCar(Model):
    def __init__(self, E_func=None):
        self.nx = 3
        self.nu = 2
        self.dt = 0.1

        self.G = ca.vertcat(np.eye(self.nx+self.nu), -np.eye(self.nx+self.nu))
        x_max = np.array([1000, 1000, 2*np.pi]) # x, y, theta
        x_min = np.array([-1000, -1000, -2*np.pi])
        u_max = np.array([np.pi/2, 1.]) # angular_vel, linear acceleration
        u_min = np.array([-np.pi/2, -0.5])

        self.g = np.concatenate((x_max, u_max, -x_min, -u_min))
        self.ni = 2*(self.nx+self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.concatenate((x_max, -x_min))
        self.ni_f = 2*self.nx

        self.nc = 2
        self.Gc_2d = np.eye(self.nx+self.nu)[:self.nc]
        self.Gcf_2d = np.eye(self.nx)[:self.nc]
        self.Gc = np.eye(self.nx+self.nu)[:self.nc]
        self.Gcf = np.eye(self.nx)[:self.nc]
        self.Gheading = np.array([[0,0,1,0,0]])
        self.Gheadingf = np.array([[0,0,1]])
        
        self.E = 0.1 * np.eye(self.nx)
        if E_func:
            self.E_func = E_func
        
        self.nw = self.nx

    def ode(self, X, u):
        x, y , theta = ca.vertsplit(X)
        v = u[1]
        xdot = v * ca.cos(theta)
        ydot = v * ca.sin(theta)
        thetadot = u[0]
        Xd = ca.vertcat(xdot, ydot, thetadot)
        return Xd
    
    def E_func(self, X):
        return self.E

    def plot_nominal_trajectory(self, X, ax=None):
        """
        :param X: nominal trajectory
        :return: plot the nominal trajectory
        """
        ax = super().plot_nominal_trajectory(X, ax)
        ax.legend(['x', 'y', 'theta'])

        return ax

    def plot_input_nominal_trajectory(self, U, ax=None):
        ax = super().plot_input_nominal_trajectory(U, ax)
        ax.legend(['linear vel','angular vel'])
        return ax