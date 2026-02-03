import numpy as np
from dyn.model import Model
import casadi as ca


class Quadrotor(Model):
    def __init__(self, m=1., Ix=0.5, Iy=0.1, Iz=0.3, E_func=None):

        self.nx = 12
        self.nu = 4
        self.dt = 1.

        self.m = m
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz

        self.G = ca.vertcat(np.eye(self.nx+self.nu),-np.eye(self.nx+self.nu))
        x_max = np.array([15.,15.,2., 0.5,0.5,2*np.pi, 1.5,1.5,1., 5.,5.,5.])
        x_min = np.array([-15.,-15.,0.05, -0.5,-0.5,-2*np.pi, -1.5,-1.5,-1., -5.,-5.,-5.])
        u_max = np.array([2*9.81, 0.1, 0.1, 0.1])
        u_min = np.array([0, -0.1, -0.1, -0.1])
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
        self.Gheading = np.eye(self.nx+self.nu)[5,:]
        self.Gheadingf = np.eye(self.nx)[5,:]
        
        self.nw = 12
        self.E = 0.1 * np.eye(self.nw)
        if E_func:
            self.E_func = E_func
    
    def E_func(self, X):
        return self.E

    def ode(self, state, input):
        GRAV = -9.81
        xx, yy, zz = state[0], state[1], state[2]

        psi, theta, phi = state[3], state[4], state[5]

        x_dot, y_dot, z_dot = state[6], state[7], state[8]

        p, q, r = state[9], state[10], state[11]

        u1, u2, u3, u4 = input[0], input[1], input[2], input[3]

        xdot = ca.vertcat(
            x_dot,
            y_dot,
            z_dot,

            q*ca.sin(phi)/ca.cos(theta) + r*ca.cos(phi)/ca.cos(theta),
            q*ca.cos(phi) - r*ca.sin(phi),
            p + q*ca.sin(phi)*ca.tan(theta) + r*ca.cos(phi)*ca.tan(theta),

            u1/self.m * (ca.sin(phi)*ca.sin(psi) + ca.cos(phi)*ca.cos(psi)*ca.sin(theta)),
            u1/self.m * (ca.cos(psi)*ca.sin(phi) - ca.cos(phi)*ca.sin(psi)*ca.sin(theta)),
            GRAV + u1/self.m * (ca.cos(phi)*ca.cos(theta)),

            ((self.Iy - self.Iz) / self.Ix) * q*r + u2 / self.Ix,
            ((self.Iz - self.Ix) / self.Iy) * p*r + u3 / self.Iy,
            ((self.Ix - self.Iy) / self.Iz) * p*q + u4 / self.Iz
        )
        return xdot
