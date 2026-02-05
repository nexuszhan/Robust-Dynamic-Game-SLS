import casadi as ca
import numpy as np
from scipy.linalg import block_diag

from solver.ocp import OCP
from dyn.LTV import LTV
from dyn.LTI import LTI

class NLP():
    def __init__(self, N, Q_all, R_all, model_all, Qf_all, x0_all, goal_all, obstacles, N_agent, radius):
        
        self.ocps = [OCP(N, Q_all[n], R_all[n], model_all[n], Qf_all[n]) for n in range(N_agent)]

        self.nlp_solver_name = "ipopt"
        self.nominal_solver_options = {
            "ipopt.print_level":0,
            "print_time": False,
            'ipopt.tol': 1e-4,
            'ipopt.constr_viol_tol': 1e-4,
            'ipopt.acceptable_tol': 1e-4,
            'ipopt.acceptable_constr_viol_tol': 1e-4,
        }
        
        self.epsilon_backoff = 1e-10
        self.solution = {}
        self.verbose = True

        # dynamics matrices
        for ocp in self.ocps:
            ocp.initialize_list_dynamics()

        # solver matrices
        self.nominal_ubg = None  # nominal upper bound on the inequality constraints (without backoff)
        self.ubg = None

        # # solver objects
        self.solver_nlp = None
        self.ubx_fun = None
        self.lbx_fun = None
        self.lbg = None
        self.current_adjoint_correction = None
        self.init_guess = None
        self.nlp = {}
        self.n_constr = None

        self.n_obst = len(obstacles)

        self.N_agent = N_agent

        self.initialize_nlp(x0_all, goal_all, obstacles, N_agent, radius)

    def solve(self, x0_all):
        """
        This method solves the forward pass of the fast-SLS algorithm. In particular, it solves a linear trajectory
        optimization problem. For each iteration of the fast-SLS algorithm, it solves the problem with different
        backoff.
        :param x0: initial conditions
        :return:
        """
        
        # solve the optimization problem
        # catch unfeasible solution
        try:
            sol = self.solver_nlp(x0=self.init_guess, lbg=self.lbg, ubg=self.ubg, lbx=self.lbx, ubx=self.ubx)
        except Exception as e:
            if self.verbose:
                print(e)
                print('QP: Unfeasible forward solution. Try with another initial condition.')
            # The region of attraction of the fast-SLS algorithm is not guaranteed to be the same as the region of
            # attraction of the controller. This could be fixed with another implementation.
            self.solution['success'] = False
            return self.solution
        
        self.solution['success'] = self.solver_nlp.stats()["success"] #True
        
        start_x = 0
        end_x = 0
        start_g = 0
        end_g = 0
        self.solution['primal_vec'] = np.array(sol['x'])
        self.solution['primal_x_all'] = []
        self.solution['primal_u_all'] = []

        for n in range(self.N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N
            nx = m.nx
            nu = m.nu
            ni = m.ni
            ni_f = m.ni_f
            n_obst = self.n_obst
            nc = m.nc

            # extract the solution
            # primal solution
            end_x += (N+1)*nx + N*nu
            primal_y = np.reshape(np.concatenate([sol['x'][start_x:end_x], np.zeros((nu, 1))]), (nx + nu, N + 1), order='F')
            primal_x = primal_y[:nx, :]
            primal_u = primal_y[nx:, :N]

            # update the current iteration data
            # self.solution['primal_vec_all'].append(np.array(sol['x'][start_x:end_x]))
            self.solution['primal_x_all'].append(primal_x)
            self.solution['primal_u_all'].append(primal_u)

            start_x = end_x
            start_g = end_g
        
        self.solution['cost'] = np.double(sol['f'])

        self.init_guess = np.array(sol['x'])

        return self.solution
    
    def rk4(self, m, x, u):
        ode = m.ode
        h = m.dt

        k_1 = ode(x, u)
        k_2 = ode(x + 0.5 * h * k_1, u)
        k_3 = ode(x + 0.5 * h * k_2, u)
        k_4 = ode(x + h * k_3, u)
        x_p = x + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
        return x_p

    def initialize_nlp(self, x0_all, goal_all, obstacles, N_agent, radius):
        obj = 0
        disturbance_penalty = 5000
        constr = []
        variables = []
        
        n_obst = self.n_obst
        symbol_y = []
        symbol_zf = []

        n_constr = []

        for n in range(N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N

            nx = m.nx
            nu = m.nu
            ni = m.ni
            ni_f = m.ni_f
            nc = m.nc

            Y = ca.MX.sym("state-input", nx+nu, N)
            Z_f = ca.MX.sym("terminal-state", nx, 1)
            Z = Y[:nx, :]
            V = Y[nx:, :]

            # objective
            Q = ca.DM(ocp.Q)
            R = ca.DM(ocp.R)
            Qf = ca.DM(ocp.Qf)
        
            goal = ca.DM(goal_all[n])
            obj += (Z_f-goal).T @ Qf @ (Z_f-goal) + disturbance_penalty*ca.sumsqr(m.E_func(Z_f))

            for t in range(N-1):
                obj += (Z[:,t+1] - Z[:,t]).T @ Q @ (Z[:,t+1] - Z[:,t]) + disturbance_penalty*ca.sumsqr(m.E_func(Z[:,t]))
            obj += (Z_f - Z[:,N-1]).T @ Q @ (Z_f - Z[:,N-1]) + disturbance_penalty*ca.sumsqr(m.E_func(Z[:,N-1]))
        
            # constraints            
            for t in range(N-1): 
                # dynamics
                eq = Z[:, t + 1] - self.rk4(m, Z[:,t], V[:,t])
                # state, input constraints
                ineq = m.G @ ca.vertcat(Z[:, t], V[:, t]) 

                constr = ca.vertcat(constr, eq)
                constr = ca.vertcat(constr, ineq)
                
                # collision avoidance
                for obst in obstacles:
                    center, rad = obst[t]
                    
                    ineq = -ca.sqrt(ca.sumsqr(m.Gcf@Z[:,t] - ca.DM(center)))
                    constr = ca.vertcat(constr, ineq)
        
            eq = Z_f - self.rk4(m, Z[:,N-1], V[:,N-1])
            ineq = m.G @ ca.vertcat(Z[:,N-1], V[:,N-1])
            constr = ca.vertcat(constr, eq)
            constr = ca.vertcat(constr, ineq)
            for obst in obstacles:
                center, rad = obst[N-1]
                
                ineq = -ca.sqrt(ca.sumsqr(m.Gcf@Z[:,N-1] - ca.DM(center)))
                constr = ca.vertcat(constr, ineq)

            ineq = m.Gf @ Z_f
            constr = ca.vertcat(constr, ineq)
            for obst in obstacles:
                center, rad = obst[N]
                
                ineq = -ca.sqrt(ca.sumsqr(m.Gcf@Z_f - ca.DM(center)))
                constr = ca.vertcat(constr, ineq)

            symbol_y.append(Y)
            symbol_zf.append(Z_f)
        
            variables = ca.vertcat(variables, ca.vertcat(ca.reshape(Y, N*(m.nx+m.nu), 1), Z_f))
        n_constr.append(constr.shape[0])

        # inter-agent collision avoidance
        for n in range(N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N

            nx = m.nx
            nu = m.nu
            ni = m.ni
            ni_f = m.ni_f
            nc = m.nc

            Y = symbol_y[n]
            Z_f = symbol_zf[n]
            Z = Y[:nx, :]
            V = Y[nx:, :]

            for t in range(N): 
                for i in range(n+1, N_agent):
                    ineq = -ca.sqrt(ca.sumsqr(m.Gcf_2d@Z[:,t] - self.ocps[i].m.Gc_2d@symbol_y[i][:,t])) + radius[n] + radius[i]
                    constr = ca.vertcat(constr, ineq)

            for i in range(n+1, N_agent):
                ineq = -ca.sqrt(ca.sumsqr(m.Gcf_2d@Z_f - self.ocps[i].m.Gcf_2d@symbol_zf[i])) + radius[n] + radius[i]
                constr = ca.vertcat(constr, ineq)
            
            n_constr.append(constr.shape[0])

        self.nlp = {
            'x': variables,  # Decision variables
            'f': obj,  # Objective function
            'g': constr
        }
        
        self.n_constr = n_constr
        
        self.solver_nlp = ca.nlpsol("solver", self.nlp_solver_name, self.nlp, self.nominal_solver_options)
        
        nominal_ubg = []
        self.lbg = []
        for n in range(N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N
            nc = 1 #m.nc
            agent_rad = radius[n]
            ni = m.ni
            ni_f = m.ni_f
            if isinstance(m, LTI): # assume time invariant constraints
                raise NotImplementedError("LTI not supported yet")
                nominal_ubg = (
                    ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1), m.g - self.epsilon_backoff)))
            elif isinstance(m, LTV): # assume time-varying constraints
                # nominal_ubg = ca.vertcat(*[ca.vertcat(ca.DM.zeros(m.nx, 1), g - self.epsilon_backoff) for g in m.g_list])
                tmp = ca.vertcat(*[ca.vertcat(ca.DM.zeros(m.nx, 1), m.g_list[t] - self.epsilon_backoff, 
                                                  ca.vertcat(*[ca.vertcat(*[-obst[t][1]-agent_rad-self.epsilon_backoff for _ in range(nc)]) for obst in obstacles])) 
                                                  for t in range(N)])
                nominal_ubg = ca.vertcat(nominal_ubg, tmp)

                nominal_ubg = ca.vertcat(nominal_ubg, m.gf-self.epsilon_backoff, 
                                 ca.vertcat(*[ca.vertcat(*[-obst[N][1]-agent_rad-self.epsilon_backoff for _ in range(nc)]) 
                                            for obst in obstacles]))
                
                tmp = ca.vertcat(ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1), -ca.DM.inf(ni + nc*n_obst,1))))
                self.lbg = ca.vertcat(self.lbg, tmp)
                self.lbg = ca.vertcat(self.lbg, -ca.DM.inf(ni_f + nc*n_obst, 1))
            else:
                # not the correct instance
                raise ValueError('The model should be either LTI or LTV')

        for n in range(N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N
            nc = 1 #m.nc
            
            if isinstance(m, LTI): # assume time invariant constraints
                raise NotImplementedError("LTI not supported yet")
                nominal_ubg = (
                    ca.kron(ca.DM.ones(N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1), m.g - self.epsilon_backoff)))
            elif isinstance(m, LTV): # assume time-varying constraints
                nominal_ubg = ca.vertcat(nominal_ubg, ca.DM.zeros((N+1)*(N_agent-n-1)*nc**2, 1))
                 
                self.lbg = ca.vertcat(self.lbg, -ca.DM.inf((N+1)*(N_agent-n-1)*nc**2, 1))
            else:
                # not the correct instance
                raise ValueError('The model should be either LTI or LTV')

        self.nominal_ubg = nominal_ubg
        self.ubg = nominal_ubg
        
        ubx = []
        lbx = []
        for n in range(N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N
            
            ubx = ca.vertcat(ubx, ca.vertcat(ca.DM(x0_all[n]) + self.epsilon_backoff, ca.DM.inf(N*(m.nx+m.nu), 1)))
            lbx = ca.vertcat(lbx, ca.vertcat(ca.DM(x0_all[n]) - self.epsilon_backoff, -ca.DM.inf(N*(m.nx+m.nu), 1)))
        self.ubx = ubx
        self.lbx = lbx
      
        self.init_guess = []
        for n in range(N_agent):
            ocp = self.ocps[n]
            m = ocp.m
            N = ocp.N
            nu = m.nu
            x0 = x0_all[n]
            goal = goal_all[n]

            self.init_guess = np.concatenate([self.init_guess, np.tile(np.concatenate([x0, np.zeros((nu))]), N), goal])

    def update_ubg(self, ubg):
        """
        Update the upper bound on the inequality constraints.
        :param ubg: new upper bound on the inequality constraints
        :return:
        """
        self.ubg = ubg

    def reset_ubg(self):
        """
        Reset the upper bound on the inequality constraints to the nominal value.
        :return:
        """
        self.ubg = self.nominal_ubg