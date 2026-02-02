import numpy as np
from dyn.LTI import LTI
from dyn.LTV import LTV

class OCP:
    def __init__(self, N, Q, R, m, Qf, Q_reg=None, R_reg=None, Q_reg_f=None):
        self.N = N
        self.Q = Q
        self.R = R
        self.m = m
        self.xf = np.zeros((m.nx, 1))
        self.Qf = Qf
        if Q_reg is not None:
            self.Q_reg = Q_reg
        else:
            self.Q_reg = np.eye(Q.shape[0])
        if R_reg is not None:
            self.R_reg = R_reg
        else:
            self.R_reg = np.eye(R.shape[0])
        if Q_reg_f is not None:
            self.Q_reg_f = Q_reg_f
        else:
            self.Q_reg_f = np.eye(Qf.shape[0])

        self.CONV_EPS = 1e-6

        self.A_list = None
        self.B_list = None
        self.C_list = None

        self.E_list = None
        self.F_list = None

        self.g_list = None
        self.gf = None

    def initialize_list_dynamics(self):
        """
        Initialize the linear dynamics. Assume time-invariant cost and constraints
        :return:
        """
        m = self.m
        # check if m is an instance of LTI, where LTI is a class
        if isinstance(m, LTI):
            self.A_list = [m.A for _ in range(self.N)]
            self.B_list = [m.B for _ in range(self.N)]
            self.E_list = [m.E for _ in range(self.N)]
            self.E_list.insert(0, m.E)
            self.g_list = [m.g for _ in range(self.N)]

        elif isinstance(m, LTV):
            self.A_list = m.A_list
            self.B_list = m.B_list
            self.E_list = m.E_list
            self.g_list = m.g_list
            self.gf = m.gf
        else:
            raise ValueError('Model type not supported')
        
    def update_dynamics_list(self, new_list_A, new_list_B, new_list_E=None, new_list_g=None, new_gf=None):
        """
        Update the linear dynamics.
        :return:
        """
        # udpate the dynamics
        self.A_list = new_list_A
        self.B_list = new_list_B
        if new_list_E is not None:
            self.E_list = new_list_E

        if new_list_g is not None:
            self.g_list = new_list_g

        if new_gf is not None:
            self.gf = new_gf

    #todo: not consistent: A, B should only be for linear dynamics

    @staticmethod
    def riccati_step(A, B, Cx, Cu, Sk):
        x = B.T @ Sk
        y = A.T @ Sk
        K = -np.linalg.solve(Cu + x @ B, x @ A)
        S = Cx + y @ A + y @ B @ K
        return K, S

    @staticmethod
    def riccati_step_cholesky(A, B, Cx, Cu, Sk):
        x = B.T @ Sk
        y = A.T @ Sk
        L = np.linalg.cholesky(Cu + x @ B)
        M = np.linalg.solve(L, x @ A)
        K = -np.linalg.solve(L.T, M)
        S = Cx + y @ A + y @ B @ K
        return K, S
