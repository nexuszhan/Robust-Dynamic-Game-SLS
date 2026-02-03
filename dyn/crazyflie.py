import numpy as np
from dyn.model import Model
import casadi as ca


class Crazyflie(Model):
    def __init__(self, E_func=None):
        self.nx = 3
        self.nu = 3
        self.dt = 0.1

        self.G = ca.vertcat(np.eye(self.nx+self.nu),-np.eye(self.nx+self.nu))
        x_max = np.array([10., 10., 2.]) # px, py, pz
        x_min = np.array([-10., -10., 0.1])
        u_max = np.array([0.5, 0.5, 0.2]) # vx, vy, vz
        u_min = np.array([-0.5, -0.5, -0.2])
        self.g = np.hstack((x_max, u_max, -x_min, -u_min))
        self.ni = 2*(self.nx+self.nu)
        self.Gf = ca.vertcat(np.eye(self.nx), -np.eye(self.nx))
        self.gf = np.hstack((x_max, -x_min))
        self.ni_f = 2*self.nx
        
        self.nc = 3
        self.Gc = np.eye(self.nx+self.nu)[:self.nc,:]
        self.Gcf = np.eye(self.nx)[:self.nc,:]
        self.Gc_2d = np.eye(self.nx+self.nu)[:2,:]
        self.Gcf_2d = np.eye(self.nx)[:2,:]
        # no LOS implemented for quadcopter
        self.Gheading = np.zeros((1, self.nx+self.nu))
        self.Gheadingf = np.zeros((1, self.nx))
        
        self.nw = self.nx
        self.E = 0.1 * np.eye(self.nw)
        if E_func:
            self.E_func = E_func
    
    def E_func(self, X):
        return self.E

    def ode(self, state, input):
        px, py, pz = ca.vertsplit(state)
        vx, vy, vz = ca.vertsplit(input)

        px_d = vx
        py_d = vy
        pz_d = vz
        return ca.vertcat(px_d, py_d, pz_d)
