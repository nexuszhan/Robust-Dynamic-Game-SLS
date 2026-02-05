import casadi as ca
import numpy as np
from scipy.linalg import block_diag
from matplotlib import pyplot as plt
from prettytable import PrettyTable, NONE, HEADER
import time

from solver.ocp import OCP
from solver.nlp import NLP
from dyn.LTI import LTI
from dyn.LTV import LTV
from util.SLS import SLS
from util.minkowski_sum import minkowski_sum_N_ellipsoids_outer

class fast_SLS(OCP):
    """
    This class adjusts fast SLS to use an NLP for trajectory optimization
    """

    def __init__(self, N, Q, R, m, Qf, n_obst, n_other, n_leader, n_follower, m_nonlinear, 
                 Q_reg=None, R_reg=None, Q_reg_f=None):
        """
        :param N: int: The horizon length.
        :param Q: numpy array: The state cost matrix.
        :param R: numpy array: The control cost matrix.
        :param m: object: The dynamical system.
        :param Qf: numpy array: The final state cost matrix.
        """
        super().__init__(N, Q, R, m, Qf, Q_reg, R_reg, Q_reg_f)
        
        self.n_obst = n_obst
        self.n_other = n_other
        self.n_leader = n_leader
        self.n_follower = n_follower
        self.m_nonlinear = m_nonlinear

        # parameter solver
        self.epsilon_backoff = 1e-10
        self.MAX_ITER = 30 
        self.current_iteration = {}  # structure that contains current iteration data
        self.convergence_data = {}  # timings, number of iterations, convergence, etc.
        self.save_it_data = True
        self.it_data = {}  # structure that contains iteration data
        self.verbose = True

        # dynamics matrices
        self.initialize_list_dynamics()
        self.initialize_jacobian_Function()

        # forward solver objects
        self.solver_forward = None
        self.nominal_ubg = None  # nominal upper bound on the inequality constraints (without backoff)
        self.current_adjoint_correction = None
        self.linearization_error = False

        self.init_nlp = True

        # self.initialize_backoff()

    def solve(self, x0, goal, obstacles, other_agents, max_dist, min_dist, leaders, followers, half_cone, 
              ca_weight, prox_weight, lqr=False, init_guess=None):
        """
        This method solves the optimal control problem using the fast-SLS algorithm.
        :param x0:
        :return:
        """
        self.initialize_solver_forward(x0, goal, obstacles, other_agents, max_dist, min_dist, leaders, followers, init_guess)
        self.initialize_solver()
        self.initialize_backoff(other_agents)
        
        if self.verbose:
            table = self.printHeader()

        for i in range(self.MAX_ITER):
            start = time.perf_counter()
            if not self.forward_solve(x0, goal, obstacles, other_agents, min_dist, leaders, followers, half_cone, ca_weight, prox_weight, lqr):
                break
            end = time.perf_counter()
            self.current_iteration["forward_time"] += end-start

            if i == 0:
                self.current_iteration["initial_x"] = self.current_iteration["primal_x"].copy() 
            self.evaluate_dual_eta()
            
            self.update_jacobian()
            start = time.perf_counter()
            self.backward_solve()
            end = time.perf_counter()
            self.current_iteration["backward_time"] += end-start

            start = time.perf_counter()
            self.update_tightening()
            end = time.perf_counter()
            self.current_iteration["tighten_time"] += end-start

            if self.check_convergence_socp():
                self.current_iteration['success'] = True
                self.convergence_data['iterations'] = i
                if self.verbose:
                    print('Fast-SLS: Solution found! Converged in {} iterations'.format(i))
                solution = self.post_processing_solution()
                # self.reset_solver_to_zeros(x0, goal)
                return solution
            
            self.update_solver_forward_ubg(obstacles, other_agents, max_dist, min_dist, leaders)

            if self.verbose:
                self.printLine(i, table)
            # save the iteration data
            if self.save_it_data:
                self.it_data[i] = self.current_iteration.copy()

        self.current_iteration['success'] = False 
        # self.current_iteration['success'] = True
        if self.verbose:
            print('Fast-SLS: Did not converge in {} iterations'.format(i+1))
            print("The result is still valid")
        solution = self.post_processing_solution()
        # self.reset_solver_to_zeros(x0, goal)
        
        return solution

    def initialize_solver(self):
        N = self.N
        ni = self.m.ni
        ni_f = self.m.ni_f
        nc = self.m.nc
        n_obst = self.n_obst
        n_other = self.n_other
        n_leader = self.n_leader
        n_follower = self.n_follower

        self.current_iteration['cost_nominal'] = np.nan
        self.current_iteration['cost_tube'] = np.nan
        self.current_iteration['cost'] = np.nan

        self.current_iteration['primal_vec'] = np.nan
        self.current_iteration['dual_vec'] = np.nan

        self.current_iteration['eta'] = np.full((N, N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)
        self.current_iteration['eta_f'] = np.full((N + 1, ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)

        self.current_iteration["backoff"] = np.zeros((N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + 1e-5 #self.epsilon_backoff
        self.current_iteration["backoff_f"] = np.zeros((ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) +  1e-5 #self.epsilon_backoff

        self.current_iteration["outer_approx"] = np.zeros((N+1, self.m.nx)) + 1e-5

        self.current_iteration["forward_time"] = 0
        self.current_iteration["backward_time"] = 0
        self.current_iteration["tighten_time"] = 0

    def initialize_jacobian_Function(self):
        """
        This method initializes the Jacobian functions.
        :return:
        """
        x = ca.MX.sym('x', self.m.nx)
        u = ca.MX.sym('u', self.m.nu)
        A = ca.jacobian(self.m_nonlinear.ddyn(x, u, self.m_nonlinear.dt), x)
        B = ca.jacobian(self.m_nonlinear.ddyn(x, u, self.m_nonlinear.dt), u)

        self.A_fun = ca.Function('A_fun', [x, u], [A])
        self.B_fun = ca.Function('B_fun', [x, u], [B])

    def reset_solver_to_zeros(self, x0, goal):
        """
        This method resets the solver of the fast-SLS algorithm such that it can be used with new initial conditions. In particular, it
        resets the primal and dual solutions, the backoff, and the solver matrices.
        :return:
        """

        N = self.N
        ni = self.m.ni
        ni_f = self.m.ni_f
        nc = self.m.nc
        n_obst = self.n_obst
        n_other = self.n_other
        n_leader = self.n_leader

        self.current_iteration['primal_vec'] = np.nan
        self.current_iteration['primal_x'] = np.nan
        self.current_iteration['primal_u'] = np.nan
        self.current_iteration['previous_primal_vec'] = np.nan

        self.current_iteration['dual_vec'] = np.nan
        self.current_iteration['dual_mu'] = np.nan
        self.current_iteration['dual_mu_f'] = np.nan
        self.current_iteration['previous_dual_vec'] = np.nan

        self.solver_forward.reset_ubg()
        self.solver_forward.init_guess = np.concatenate([ np.tile(np.concatenate([x0, np.zeros((self.m.nu))]), self.N), goal])

        self.current_iteration['beta'] = np.zeros((N, N, ni + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2)) + self.epsilon_backoff
        self.current_iteration['beta_f'] = np.zeros((N + 1, ni_f + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2)) + self.epsilon_backoff

        self.current_iteration['eta'] = np.zeros((self.N, self.N, ni + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2))
        self.current_iteration['eta_f'] = np.zeros((self.N + 1, ni_f + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2))
        self.current_iteration['previous_eta'] = np.zeros((self.N, self.N, ni + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2))
        self.current_iteration['previous_eta_f'] = np.zeros((self.N + 1, ni_f + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2))

        self.current_iteration['S'] = np.zeros((self.N + 1, self.N + 1, self.m.nx, self.m.nx))
        self.current_iteration['K'] = np.zeros((self.N, self.N + 1, self.m.nu, self.m.nx))

        self.current_iteration['cost_nominal'] = np.nan
        self.current_iteration['cost_tube'] = np.nan
        self.current_iteration['cost'] = np.nan
        
        self.current_iteration['sqrt_beta'] = np.zeros((N, N, ni + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2)) + np.sqrt(self.epsilon_backoff)
        self.current_iteration['sqrt_beta_f'] = np.zeros((N + 1, ni_f + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2)) + np.sqrt(self.epsilon_backoff)
        self.current_iteration["backoff"] = np.zeros((N, ni + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2)) + self.epsilon_backoff
        self.current_iteration["backoff_f"] = np.zeros((ni_f + nc*n_obst + nc*n_other + nc*n_leader + n_leader*2)) + self.epsilon_backoff

        self.current_iteration["forward_time"] = 0
        self.current_iteration["backward_time"] = 0
        self.current_iteration["tighten_time"] = 0

    def reset_solver_to_warm_start(self):
        """
        This method resets the solver of the fast-SLS algorithm such that it can be used with new initial conditions. In particular, it
        resets the primal and dual solutions to a shifted version of the previous solution. This is useful for warmstarting the solver.
        """
        # todo add shift of primal and dual solution and backoff

        self.current_iteration['convergence'] = False
        self.current_iteration['iterations'] = 0

        self.convergence_data = {}
        self.it_data = {}

        A_cl = self.m.A - self.m.B @ self.m.Kf
        Phi_x_last = self.current_iteration['Phi_x'][-1]

        # create a new Phi_x_last matrix, where each [i,:,:] is multiplied by A_cl
        new_Phi_x_last = np.zeros(Phi_x_last.shape)
        for i in range(Phi_x_last.shape[0]):
            new_Phi_x_last[i, :, :] = A_cl @ Phi_x_last[i, :, :]
        # remove the first matrix, and add m.E  at the end
        beta = self.current_iteration['beta']
        beta_f = self.current_iteration['beta_f']

        raise NotImplementedError  # todo implement this function

    def initialize_solver_forward(self, x0, goal, obstacles, other_agents, max_dist, min_dist, LOS_targets, followers, init_guess=None):
        m = self.m

        nx = m.nx
        nu = m.nu
        N = self.N

        self.solver_forward = NLP(self.N, self.Q, self.R, m, self.Qf, x0, goal, obstacles, other_agents, max_dist, min_dist, LOS_targets, followers)
        self.solver_forward.E_func = self.m_nonlinear.E_func
        self.current_adjoint_correction = ca.DM.zeros(((nx + nu) * N + nx, 1))
        if np.any(init_guess):
            self.solver_forward.init_guess = init_guess
        self.other_agent_const_num = self.solver_forward.other_agent_constr_num

        return self.solver_forward

    def forward_solve(self, x0, goal, obstacles, other_agents, min_dist, LOS_targets, followers, half_cone, ca_weight, prox_weight, lqr=False):
        """
        This method solves the forward pass of the fast-SLS algorithm. In particular, it solves a linear trajectory
        optimization problem. For each iteration of the fast-SLS algorithm, it solves the problem with different
        backoff.
        :param x0: initial conditions
        :return:
        """
        
        if self.n_leader > 0 or self.n_follower > 0 or self.init_nlp:
            self.solver_forward.update_nlp(goal, obstacles, other_agents, min_dist, LOS_targets, followers,
                                           self.current_iteration["backoff"][:,:6], self.current_iteration["backoff_f"][:6], self.current_iteration["outer_approx"],
                                           half_cone, ca_weight, prox_weight, lqr)
            self.init_nlp = False
        solution_forward = self.solver_forward.solve(x0)

        if solution_forward['success'] is False:
            if self.verbose:
                print('Fast-SLS: Unfeasible forward solution. Try with another initial condition.')

            return solution_forward['success']

        # extract the solution
        # primal solution
        primal_y = solution_forward['primal_vec']
        primal_x = solution_forward['primal_x']
        primal_u = solution_forward['primal_u']

        # dual solution
        dual = solution_forward['dual_vec']
        # extract the dual associated to the terminal constraints
        mu_f = solution_forward['dual_mu_f']
        mu = solution_forward['dual_mu']

        # save previous primal and dual solutions
        self.current_iteration['previous_primal_vec'] = self.current_iteration['primal_vec']
        self.current_iteration['previous_dual_vec'] = self.current_iteration['dual_vec']

        # update the current iteration data
        self.current_iteration['primal_vec'] = primal_y
        self.current_iteration['primal_x'] = primal_x
        self.current_iteration['primal_u'] = primal_u

        self.current_iteration['dual_vec'] = dual
        self.current_iteration['dual_mu'] = mu  # transpose to have the same index ordering as other time series
        self.current_iteration['dual_mu_f'] = mu_f  # transpose to have the same index ordering as other time series

        self.current_iteration['cost_nominal'] = solution_forward['cost']
        return solution_forward['success']
    
    def update_jacobian(self):
        """
        This method updates the Jacobian of the dynamics function based on the current nominal trajectory.
        :return:
        """
        nx = self.m.nx
        nu = self.m.nu
        N = self.N

        primal_x = self.current_iteration['primal_x']
        primal_u = self.current_iteration['primal_u']
        # initialize A_list as a list of zeros nx x nx matrices
        A_list = [np.zeros((nx, nx)) for _ in range(N)]
        B_list = [np.zeros((nx, nu)) for _ in range(N)]
        if self.linearization_error:
            raise NotImplementedError("Linearization error is not implemented yet")
        else:
            E_list = []
            for t in range(N):
                e = self.m_nonlinear.E_func(primal_x[:,t]) 
                E_list.append(e)
            e = self.m_nonlinear.E_func(primal_x[:,N]) 
            E_list.append(e)

        for i in range(N):
            A_list[i] = self.A_fun(primal_x[:, i], primal_u[:, i])
            B_list[i] = self.B_fun(primal_x[:, i], primal_u[:, i])

        g_list = self.g_list
        gf = self.gf

        self.update_dynamics_list(A_list, B_list, E_list, g_list, gf)

    def backward_solve(self):
        """
        This method solves the backward pass of the fast-SLS algorithm. In particular, it solves the part of the problem related to the Riccati recursions and controller update
        """
        m = self.m
        N = self.N
        nx = m.nx
        nu = m.nu

        G_f = m.Gf
        G = m.G

        A = self.A_list
        B = self.B_list

        S = np.full((N + 1, N + 1, nx, nx), np.nan)
        K = np.full((N, N + 1, nu, nx), np.nan)

        # double for loop to compute the S and K matrices: the Riccati recursions
        # terminal cost
        stacked_Gf = np.vstack([G_f, self.Gf_padding])
        for jj in range(N + 1):
            C_fj = stacked_Gf.T @ np.diag(self.current_iteration['eta_f'][jj, :]) @ stacked_Gf
            S[N, jj, :, :] = C_fj + self.Q_reg_f
        
        stacked_G = np.vstack([G, self.G_padding])
        for jj in range(N):
            for kk in range(N - 1, jj - 1, -1):
                C_kj = stacked_G.T @ np.diag(self.current_iteration['eta'][kk, jj, :]) @ stacked_G
                C_kj_xx = C_kj[:nx, :nx] + self.Q_reg  # extract the part of the matrix related to the state
                C_kj_uu = C_kj[nx:, nx:] + self.R_reg  # extract the part of the matrix related to the control
                # todo : Currently assumed not cross state-input constraints
                K[kk, jj, :, :], S[kk, jj, :, :] = self.riccati_step(A[kk], B[kk], C_kj_xx, C_kj_uu,
                                                                     S[kk + 1, jj, :, :])
        self.current_iteration['S'] = S
        self.current_iteration['K'] = K

    def initialize_backoff(self, other_agents):
        """
        Initialize the backoff to some epsilon values for the first iteration
        :return:
        """
        # initialize the backoff
        N = self.N
        ni = self.m.ni
        ni_f = self.m.ni_f
        nc = self.m.nc
        n_obst = self.n_obst
        n_leader = self.n_leader
        n_follower = self.n_follower

        self.current_iteration['beta'] = np.zeros((N, N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + self.epsilon_backoff
        self.current_iteration['beta_f'] = np.zeros((N + 1, ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + self.epsilon_backoff
        self.current_iteration['sqrt_beta'] = np.zeros((N, N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + np.sqrt(self.epsilon_backoff)
        self.current_iteration['sqrt_beta_f'] = np.zeros((N + 1, ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + np.sqrt(self.epsilon_backoff)
        
        self.current_iteration["backoff"] = np.zeros((N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + self.epsilon_backoff
        self.current_iteration["backoff_f"] = np.zeros((ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)) + self.epsilon_backoff
        
        m = self.m
        Gf_padding = np.kron(np.ones((n_obst, 1)), m.Gcf_2d)
        for other in other_agents:
            if len(other[0][0]) == 2:
                Gf_padding = np.vstack([Gf_padding, m.Gcf_2d])
            elif len(other[0][0]) == 3:
                Gf_padding = np.vstack([Gf_padding, m.Gcf])
            else:
                raise NotImplementedError
        Gf_padding = np.vstack([Gf_padding, np.kron(np.ones((n_leader, 1)), m.Gcf), 
                                np.kron(np.ones((n_leader*2, 1)), m.Gheadingf),
                                np.kron(np.ones((n_follower*2, 1)), m.Gheadingf)])
        self.Gf_padding = Gf_padding
        
        G_padding = np.kron(np.ones((n_obst, 1)), m.Gc_2d)
        for other in other_agents:
            if len(other[0][0]) == 2:
                G_padding = np.vstack([G_padding, m.Gc_2d])
            elif len(other[0][0]) == 3:
                G_padding = np.vstack([G_padding, m.Gc])
            else:
                raise NotImplementedError
        G_padding = np.vstack([G_padding, np.kron(np.ones((n_leader, 1)), m.Gc), 
                                np.kron(np.ones((n_leader*2, 1)), m.Gheading),
                                np.kron(np.ones((n_follower*2, 1)), m.Gheading)])
        self.G_padding = G_padding

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

    def evaluate_dual_eta(self):
        """
        This method computes the dual variables eta_kj.
        :return:
        """
        N = self.N
        n_obst = self.n_obst
        nc = self.m.nc
        n_leader = self.n_leader
        n_follower = self.n_follower

        # initialization the eta matrix with Nan values
        eta = np.full((N, N, self.m.ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)
        eta_f = np.full((N + 1, self.m.ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)

        beta = self.current_iteration['beta']
        beta_f = self.current_iteration['beta_f']

        # if value of beta is too small, set it to self.epsilon_backoff
        beta = np.maximum(beta, self.epsilon_backoff)
        beta_f = np.maximum(beta_f, self.epsilon_backoff)

        for jj in range(N):
            for kk in range(jj, N):
                eta[kk, jj, :] = self.current_iteration['dual_mu'][:, kk] / np.sqrt(beta[kk, jj, :]) / 2.

        for jj in range(N + 1):
            eta_f[jj, :] = self.current_iteration['dual_mu_f'] / np.sqrt(beta_f[jj, :]) / 2.

        # update the current iteration data for eta_kj
        self.current_iteration['previous_eta'] = self.current_iteration['eta']
        self.current_iteration['previous_eta_f'] = self.current_iteration['eta_f']

        self.current_iteration['eta'] = eta
        self.current_iteration['eta_f'] = eta_f

    def update_cost(self):
        raise NotImplementedError  # todo implement this function, such that the function update_tightening can be simplified

    def check_convergence_socp(self):
        """
        This method checks the convergence of the fast SLS algorithm.
        :return:
        """
        # print('Convergence checked on nominal trajectory')
        delta_primal = np.max(
            np.fabs(self.current_iteration['primal_vec'] - self.current_iteration['previous_primal_vec']))

        # replace the NaN values by zeros
        # delta_dual = np.max(np.nan_to_num(
        #     self.current_iteration['eta'] - self.current_iteration['previous_eta']))
        
        delta_tube = np.max(np.fabs(np.nan_to_num(self.current_iteration["backoff"] - self.current_iteration["previous_backoff"])))
        delta_tube_f = np.max(np.fabs(np.nan_to_num(self.current_iteration["backoff_f"] - self.current_iteration["previous_backoff_f"])))
        delta_tube = max(delta_tube, delta_tube_f)

        return delta_primal <= 1e-3 and delta_tube <= 1e-3

    def update_tightening(self):
        """
        This method updates the backoff for the next iteration of the fast-SLS algorithm. It uses the controller K_kj to calculate the values of the matrices Phi_x and Phi_u and compute the corresponding backoff
        :return:
        """
        # import the necessary variables
        N = self.N
        nx = self.m.nx
        nu = self.m.nu
        nw = self.m.nw
        ni = self.m.ni
        ni_f = self.m.ni_f
        nc = self.m.nc
        n_obst = self.n_obst
        n_leader = self.n_leader
        n_follower = self.n_follower

        G = self.m.G
        Gf = self.m.Gf

        A = self.A_list
        B = self.B_list
        E = self.E_list

        # initialize the Phi_x and Phi_u matrices
        Phi_x = np.full((N + 1, N + 1, nx, nw), np.nan)
        Phi_u = np.full((N, N + 1, nu, nw), np.nan)

        # initialize the backoff
        beta = np.full((N, N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)  # backoff
        beta_f = np.full((N + 1, ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)

        sqrt_beta = np.full((N, N, ni + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)  # backoff
        sqrt_beta_f = np.full((N + 1, ni_f + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2), np.nan)

        # import current controller
        K = self.current_iteration['K']

        # forward propagate the value of Phi_x and Phi_u
        for jj in range(N + 1):
            E_j = E[jj]
            Phi_x[jj, jj, :, :] = E_j
            # Omega_j = np.linalg.pinv(E_j) @ E_j
            # todo check if the pseudo inverse works
            Omega_j = np.eye(nx)
            for kk in range(jj, N):
                Phi_u[kk, jj, :, :] = K[kk, jj, :, :] @ Omega_j @ Phi_x[kk, jj, :, :]

                # close loop dynamics
                A_cl = A[kk] + B[kk] @ K[kk, jj, :, :]
                Phi_x[kk + 1, jj, :, :] = A_cl @ Phi_x[kk, jj, :, :]

        # terminal conditions
        for jj in range(N + 1):
            for kk in range(jj, N):
                # evaluate the 2-norm of each component
                beta[kk, jj, :ni] = np.linalg.norm(G @ np.vstack([Phi_x[kk, jj], Phi_u[kk, jj]]), axis=1) ** 2
                sqrt_beta[kk, jj,:ni] = np.linalg.norm(G @ np.vstack([Phi_x[kk, jj], Phi_u[kk, jj]]), axis=1)
                sqrt_beta[kk, jj, ni:] = np.linalg.norm(self.Gf_padding @ Phi_x[kk, jj], axis=1)
                beta[kk, jj, ni:] = sqrt_beta[kk, jj, ni:] ** 2
                # todo evaluate sqrt_beta instead, since that's what we use later
            # terminal conditions
            beta_f[jj, :ni_f] = np.linalg.norm(Gf @ Phi_x[N, jj], axis=1) ** 2
            sqrt_beta_f[jj, :ni_f] = np.linalg.norm(Gf @ Phi_x[N, jj], axis=1)
            sqrt_beta_f[jj, ni_f:] = np.linalg.norm(self.Gf_padding @ Phi_x[N, jj], axis=1)
            beta_f[jj, ni_f:] = sqrt_beta_f[jj, ni_f:] ** 2

        # update system response matrices
        self.current_iteration['Phi_x'] = Phi_x
        self.current_iteration['Phi_u'] = Phi_u

        Q_reg = self.Q_reg
        R_reg = self.R_reg
        Q_reg_f = self.Q_reg_f

        self.current_iteration['cost_tube'] = SLS.eval_cost(N, Q_reg, R_reg, Q_reg_f,
                                                            SLS.convert_tensor_to_matrix(Phi_x),
                                                            SLS.convert_tensor_to_matrix(Phi_u))
        # update backoff of current iteration
        self.current_iteration['beta'] = beta
        self.current_iteration['beta_f'] = beta_f

        # evaluate new backoff as the sum of the contribution of each disturbance
        backoff = np.nansum(sqrt_beta, axis=1)
        backoff_f = np.sum(sqrt_beta_f, axis=0).T  # transpose to recover convention for time propagation
        
        self.current_iteration["previous_backoff"] = self.current_iteration["backoff"]
        self.current_iteration["previous_backoff_f"] = self.current_iteration["backoff_f"]

        self.current_iteration['backoff'] = backoff
        self.current_iteration['backoff_f'] = backoff_f

        self.current_iteration["cost"] = self.current_iteration["cost_tube"] + self.current_iteration["cost_nominal"]

        self.update_reachable_set_outer_approx()

    def update_reachable_set_outer_approx(self):
        N = self.N
        m = self.m
        nx = m.nx

        Phi_x = self.current_iteration["Phi_x"]
        outer_approx = np.zeros((N+1, nx))
        for t in range(N+1):
            vecs, semi_axes, sup = minkowski_sum_N_ellipsoids_outer(Phi_x[t,:t+1,:,:] + 1e-6*np.eye(m.nx))
            outer_approx[t,:] = semi_axes
        self.current_iteration["outer_approx"] = outer_approx

    def update_solver_forward_ubg(self, obstacles, other_agents, max_dist, min_dist, leaders):
        N = self.N
        nx = self.m.nx
        nc = self.m.nc
        n_obst = self.n_obst
        n_other = self.n_other
        n_leader = self.n_leader
        n_follower = self.n_follower
        ni = self.m.ni
        ni_f = self.m.ni_f
        gf = self.gf
        g = self.g_list

        backoff = self.current_iteration['backoff']
        backoff_f = self.current_iteration['backoff_f'] 

        absolute_backoff_table = np.squeeze(g) - backoff[:, :ni]

        if n_obst == 0 and n_other == 0 and n_leader == 0:
            new_ubg_table = np.vstack([np.zeros((nx, N)), absolute_backoff_table.T])
            new_ubg_without_terminal = np.reshape(new_ubg_table, (N * (ni + nx)), order='F')
            new_ubg = np.concatenate([new_ubg_without_terminal, gf - backoff_f[:ni_f]])
        else:
            distance_table = np.zeros((n_obst*2 + self.other_agent_const_num + n_leader*nc, N))
            distance_f = np.zeros((n_obst*2 + self.other_agent_const_num + n_leader*nc))
            for t in range(N):
                for i in range(n_obst):
                    for j in range(2):
                        reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff[t,0]**2 + backoff[t,1]**2))
                        distance_table[i*2+j][t] = -min_dist - obstacles[i][t][1] - reachable_set_approx
                cnt = 0
                for i in range(n_other):
                    for j in range(len(other_agents[i][t][0])):
                        if len(other_agents[i][t][0]) == 2:
                            reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff[t,0]**2 + backoff[t,1]**2))
                            distance_table[n_obst*2 + cnt][t] = -min_dist - other_agents[i][t][1] - reachable_set_approx
                        elif len(other_agents[i][t][0]) == 3:
                            reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff[t,0]**2 + backoff[t,1]**2 + backoff[t,2]**2))
                            distance_table[n_obst*2 + cnt][t] = -min_dist - other_agents[i][t][1] - reachable_set_approx
                        else:
                            raise NotImplementedError
                        cnt += 1
                for i in range(n_leader):
                    for j in range(nc):
                        reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff[t,0]**2 + backoff[t,1]**2))
                        distance_table[n_obst*2+self.other_agent_const_num + i*nc+j][t] = max_dist - leaders[i][t][1] - \
                                                                                            reachable_set_approx
                
            for i in range(n_obst):
                for j in range(2):
                    reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff_f[0]**2 + backoff_f[1]**2))
                    distance_f[i*2+j] = -min_dist - obstacles[i][N][1] - reachable_set_approx
            cnt = 0
            for i in range(n_other):
                for j in range(len(other_agents[i][N][0])):
                    if len(other_agents[i][N][0]) == 2:
                        reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff_f[0]**2 + backoff_f[1]**2))
                        distance_f[n_obst*2 + cnt] = -min_dist - other_agents[i][N][1] - reachable_set_approx
                    elif len(other_agents[i][N][0]) == 3:
                        reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff_f[0]**2 + backoff_f[1]**2 + backoff_f[2]**2))
                        distance_f[n_obst*2 + cnt] = -min_dist - other_agents[i][N][1] - reachable_set_approx
                    else:
                        raise NotImplementedError
                    cnt += 1
            for i in range(n_leader):
                for j in range(nc):
                    reachable_set_approx = min(self.current_iteration["outer_approx"][t,0], np.sqrt(backoff_f[0]**2 + backoff_f[1]**2))
                    distance_f[n_obst*2+self.other_agent_const_num + i*nc+j] = max_dist - leaders[i][N][1] - reachable_set_approx

            LOS_table = np.zeros((n_leader*2+n_follower*2, N))
            LOS_f = np.zeros((n_leader*2+n_follower*2))
            
            if n_leader == 0 and n_follower == 0:
                new_ubg_table = np.vstack([np.zeros((nx, N)), absolute_backoff_table.T, distance_table])
                new_ubg_without_terminal = np.reshape(new_ubg_table, (N * (ni + nx + 2*n_obst + self.other_agent_const_num)), order='F')
                new_ubg = np.concatenate([new_ubg_without_terminal, gf - backoff_f[:ni_f], distance_f])
            else:
                new_ubg_table = np.vstack([np.zeros((nx, N)), absolute_backoff_table.T, distance_table, LOS_table])
                new_ubg_without_terminal = np.reshape(new_ubg_table, (N * (ni + nx + 2*n_obst + self.other_agent_const_num + nc*n_leader + n_leader*2 + n_follower*2)), order='F')
                new_ubg = np.concatenate([new_ubg_without_terminal, gf - backoff_f[:ni_f], distance_f, LOS_f])

        self.solver_forward.update_ubg(new_ubg)

    def post_processing_solution(self):
        if self.current_iteration['success']:
            self.current_iteration['K_mat'] = SLS.convert_tensor_to_matrix(self.current_iteration['K'])
            # replace the NaN by zeros in K_mat
            self.current_iteration['K_mat'] = np.nan_to_num(self.current_iteration['K_mat'])

            Phi_u = self.current_iteration['Phi_u']
            Phi_x = self.current_iteration['Phi_x']

            # replace nan from Phi_fast_sls to 0
            Phi_u = np.nan_to_num(Phi_u, nan=0)
            Phi_x = np.nan_to_num(Phi_x, nan=0)
            self.current_iteration['Phi_u'] = Phi_u
            self.current_iteration['Phi_x'] = Phi_x

            Phi_u_fast_sls = SLS.convert_tensor_to_matrix(Phi_u)
            Phi_x_fast_sls = SLS.convert_tensor_to_matrix(Phi_x)

            self.current_iteration['Phi_u_mat'] = Phi_u_fast_sls
            self.current_iteration['Phi_x_mat'] = Phi_x_fast_sls
        else:
            self.current_iteration['success'] = True

        solution = self.current_iteration.copy()
        solution.update(self.convergence_data.copy())
        solution['it_data'] = self.it_data.copy()
        solution['cost'] = self.current_iteration['cost_tube'] + self.current_iteration['cost_nominal']

        return solution
    
    def update_traj(self, x0, cur_v, cur_z, lr):
        v = self.current_iteration["primal_u"]
        m = self.m_nonlinear

        new_v = (1-lr) * cur_v + lr * v
        new_z = np.zeros_like(cur_z)
        new_z[:,0] = x0
        for t in range(self.N):
            new_z[:,t+1] = np.squeeze(m.ddyn(new_z[:,t], new_v[:,t], m.dt))

        self.current_iteration["primal_x"] = new_z
        self.current_iteration["primal_u"] = new_v
    
    def update_tube(self):
        self.update_jacobian()
        self.backward_solve()
        self.update_tightening()

    def get_updated_sol(self, x0, prev_z, prev_v, alpha):
        self.update_traj(x0, prev_v, prev_z, alpha)
        self.update_tube()
        
        sol = self.post_processing_solution()
        return sol

    def printLine(self, i, table):

        fixed_width = 10
        primal = np.max(self.current_iteration['primal_vec'])
        dual = np.max(np.nan_to_num(self.current_iteration['eta']))
        cost = self.current_iteration['cost']
        iteration_str = f"{i:>{fixed_width}}"
        primal_val = f"{float(primal):>{fixed_width}.2e}"
        dual_val = f"{float(dual):>{fixed_width}.2e}"
        cost_val = f"{float(cost):>{fixed_width}.2e}"
        # todo add absolute primal and dual values

        table.add_row([iteration_str, primal_val, dual_val, cost_val])
        print(table.get_string(start=len(table._rows) - 1, end=len(table._rows), header=False))

    @staticmethod
    def printHeader():
        fixed_width = 10
        # Format headers to have fixed width and right alignment
        headers = ["it (fast-SLS)", "primal", "dual", "cost"]
        formatted_headers = [f"{h:>{fixed_width}}" for h in headers]
        table = PrettyTable()
        table.field_names = formatted_headers
        table.hrules = HEADER  # Horizontal line after header
        table.border = True

        # Align columns
        table.align["it"] = "right"
        table.align["primal"] = "right"
        table.align["dual"] = "right"
        table.align["cost"] = "right"

        # Set a fixed width for all columns
        fixed_width = 10
        # Set fixed widths for each column individually
        table.max_width["it"] = fixed_width
        table.max_width["primal"] = fixed_width
        table.max_width["dual"] = fixed_width
        table.max_width["cost"] = fixed_width

        print(table.get_string(end=0))
        table.hrules = NONE

        return table
