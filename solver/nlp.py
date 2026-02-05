import numpy as np
from scipy.linalg import block_diag
import casadi as ca

from solver.ocp import OCP
from dyn.LTV import LTV
from dyn.LTI import LTI

class NLP(OCP):
    def __init__(self, N, Q, R, m, Qf, x0, goal, obstacles, other_agents, max_dist, min_dist, leaders, followers):
        super().__init__(N, Q, R, m, Qf)
        
        self.nlp_solver_name = "ipopt"
        self.nominal_solver_options = {
            "ipopt.print_level":0,
            "ipopt.sb": "yes",
            "print_time": False,
            'ipopt.tol': 1e-6,
            'ipopt.constr_viol_tol': 1e-6,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_constr_viol_tol': 1e-6,
            'ipopt.dual_inf_tol': 1e-6,
            'ipopt.compl_inf_tol': 1e-6,
            'ipopt.mu_init': 1e-6,
            'ipopt.nlp_scaling_method': 'gradient-based'
        }
        
        self.epsilon_backoff = 1e-10
        self.solution = {}
        self.solver_params = {}
        self.verbose = True#False

        # dynamics matrices
        self.initialize_list_dynamics()

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
        self.prev_lam_g = None
        self.prev_lam_x = None
        self.nlp = {}
        self.n_obst = len(obstacles)
        self.n_other = len(other_agents)
        self.n_leader = len(leaders)
        self.n_follower = len(followers)

        self.ode = m.ode
        self.goal = goal
        
        self.other_agent_constr_num = 0
        self.initialize_nlp(x0, obstacles, other_agents, max_dist, min_dist, leaders, followers)

    def solve(self, x0):
        """
        This method solves the forward pass of the fast-SLS algorithm. In particular, it solves a linear trajectory
        optimization problem. For each iteration of the fast-SLS algorithm, it solves the problem with different
        backoff.
        :param x0: initial conditions
        :return:
        """
        nx = self.m.nx
        nu = self.m.nu
        ni = self.m.ni
        ni_f = self.m.ni_f
        N = self.N
        n_obst = self.n_obst
        n_other = self.n_other
        n_leader = self.n_leader
        n_follower = self.n_follower
        nc = self.m.nc
        
        # solve the optimization problem
        # catch unfeasible solution
        try:
            if np.any(self.prev_lam_g) and np.any(self.prev_lam_x):
                sol = self.solver_nlp(x0=self.init_guess, lam_g0=self.prev_lam_g, lam_x0=self.prev_lam_x, 
                                      lbg=self.lbg, ubg=self.ubg, lbx=self.lbx_fun(x0), ubx=self.ubx_fun(x0))
            else:
                sol = self.solver_nlp(x0=self.init_guess, lbg=self.lbg, ubg=self.ubg, lbx=self.lbx_fun(x0), ubx=self.ubx_fun(x0))
        except Exception as e:
            if self.verbose:
                print(e)
                print('NLP: Unfeasible forward solution. Try with another initial condition.')
            # The region of attraction of the fast-SLS algorithm is not guaranteed to be the same as the region of
            # attraction of the controller. This could be fixed with another implementation.
            self.solution['success'] = False
            return self.solution
        
        self.solution['success'] = self.solver_nlp.stats()["success"] 
        # extract the solution
        # primal solution
        primal_y = np.reshape(np.concatenate([sol['x'], np.zeros((nu, 1))]), (nx + nu, N + 1), order='F')
        primal_x = primal_y[:nx, :]
        primal_u = primal_y[nx:, :N]

        # dual solution
        dual = sol['lam_g']
        # extract the dual associated to the terminal constraints
        mu_f = dual[-ni_f - n_obst*2 - self.other_agent_constr_num - n_leader*nc - n_leader*2 - n_follower*2:]
        mu = dual[:-ni_f - n_obst*2 - self.other_agent_constr_num - n_leader*nc - n_leader*2 - n_follower*2]

        mu_all = np.reshape(mu, (N, nx + ni + n_obst*2 + self.other_agent_constr_num + nc*n_leader + n_leader*2 + n_follower*2))
        # remove the first nx lines, which corresponds to the equality constraints (dynamics)
        mu = mu_all[:, nx:]

        # convert mu_f from DM to numpy array
        mu_f = np.array(mu_f)

        # update the current iteration data
        self.solution['primal_vec'] = np.array(sol['x'])
        self.solution['primal_x'] = primal_x
        self.solution['primal_u'] = primal_u

        self.solution['dual_vec'] = np.array(sol['lam_g'])
        self.solution['dual_mu'] = mu.T  # transpose to have the same index ordering as other time series
        self.solution['dual_mu_f'] = mu_f.T  # transpose to have the same index ordering as other time series

        self.solution['cost'] = np.double(sol['f'])

        self.init_guess = np.array(sol['x'])

        return self.solution
    
    def rk4(self, x, u):
        ode = self.ode
        h = self.m.dt

        k_1 = ode(x, u)
        k_2 = ode(x + 0.5 * h * k_1, u)
        k_3 = ode(x + 0.5 * h * k_2, u)
        k_4 = ode(x + h * k_3, u)
        x_p = x + (1 / 6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
        return x_p

    def initialize_nlp(self, x0, obstacles, other_agents, max_dist, min_dist, leaders, followers):
        nx = self.m.nx
        nu = self.m.nu
        ni = self.m.ni
        ni_f = self.m.ni_f
        N = self.N
        m = self.m
        n_obst = self.n_obst
        n_leader = self.n_leader
        n_follower = self.n_follower
        nc = m.nc

        if isinstance(self.m, LTI): # assume time invariant constraints
            nominal_ubg = (
                ca.kron(ca.DM.ones(self.N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1), m.g - self.epsilon_backoff)))
            raise NotImplementedError
        elif isinstance(self.m, LTV): # assume time-varying constraints
            nominal_ubg = ca.vertcat(*[ca.vertcat(ca.DM.zeros(m.nx, 1)+1e-5, m.g_list[t] - self.epsilon_backoff, 
                                                  ca.vertcat(*[ca.vertcat(*[-min_dist-obst[t][1]-self.epsilon_backoff for _ in range(2)]) for obst in obstacles]),
                                                  ca.vertcat(*[ca.vertcat(*[-min_dist-other[t][1]-self.epsilon_backoff for _ in range(len(other[t][0]))]) for other in other_agents]),
                                                  ca.vertcat(*[ca.vertcat(*[max_dist-leader[t][1]-self.epsilon_backoff for _ in range(nc)]) for leader in leaders]),
                                                  ca.vertcat(*[ca.vertcat(*[0., 0.]) for agent in leaders]),
                                                  ca.vertcat(*[ca.vertcat(*[0., 0.]) for agent in followers])
                                                  ) 
                                                  for t in range(N)])
        else:
            # not the correct instance
            raise ValueError('The model should be either LTI or LTV')
        nominal_ubg = ca.vertcat(nominal_ubg, m.gf - self.epsilon_backoff, 
                                 ca.vertcat(*[ca.vertcat(*[-min_dist-obst[N][1]-self.epsilon_backoff for _ in range(2)]) 
                                            for obst in obstacles]),
                                ca.vertcat(*[ca.vertcat(*[-min_dist-other[N][1]-self.epsilon_backoff for _ in range(len(other[N][0]))]) 
                                             for other in other_agents]),
                                ca.vertcat(*[ca.vertcat(*[max_dist-leader[N][1]-self.epsilon_backoff for _ in range(nc)]) 
                                             for leader in leaders]),
                                ca.vertcat(*[ca.vertcat(*[0., 0.])
                                             for agent in leaders]),
                                ca.vertcat(*[ca.vertcat(*[0., 0.]) 
                                             for agent in followers])
                                )

        self.nominal_ubg = nominal_ubg
        self.ubg = np.array(nominal_ubg).reshape(-1)
    
        self.other_agent_constr_num = 0
        for other in other_agents:
            self.other_agent_constr_num += len(other[0][0])
        self.lbg = ca.vertcat(ca.kron(ca.DM.ones(self.N, 1), ca.vertcat(ca.DM.zeros(m.nx, 1)-1e-5, -ca.DM.inf(ni + 2*n_obst + self.other_agent_constr_num + 
                                                                                                              nc*n_leader + n_leader*2 + n_follower*2, 1))))
        self.lbg = ca.vertcat(self.lbg, -ca.DM.inf(ni_f + 2*n_obst + self.other_agent_constr_num + 
                                                   nc*n_leader + n_leader*2 + n_follower*2, 1))

        x0_sym = ca.SX.sym('x0', nx)
        ubx = ca.vertcat(x0_sym + self.epsilon_backoff, ca.DM.inf(N * (nx + nu), 1))
        lbx = ca.vertcat(x0_sym - self.epsilon_backoff, -ca.DM.inf(N * (nx + nu), 1))

        self.ubx_fun = ca.Function('ubx_fun', [x0_sym], [ubx])
        self.lbx_fun = ca.Function('lbx_fun', [x0_sym], [lbx])
        
        self.init_guess = np.concatenate([ np.tile(np.concatenate([x0, np.zeros((nu))]), N), self.goal])

    def update_nlp(self, goal, obstacles, other_agents, min_dist, leaders, followers, backoff, backoff_f, 
                   outer_approx, half_cone, ca_weight, prox_weight, LQR_cost=False):
        N = self.N
        m = self.m
        nc = m.nc

        Y = ca.MX.sym("state-input", self.m.nx+self.m.nu, self.N)
        Z_f = ca.MX.sym("terminal-state", self.m.nx, 1)
        Z = Y[:self.m.nx, :]
        V = Y[self.m.nx:, :]

        # objective
        Q = ca.DM(self.Q)
        R = ca.DM(self.R)
        Qf = ca.DM(self.Qf)
        
        goal = ca.DM(goal)
        disturbance_penalty = 7500 #5000
        f = (Z_f-goal).T @ Qf @ (Z_f-goal) + disturbance_penalty*ca.sumsqr(self.m.E_func(Z_f))
        if LQR_cost:
            # naive narrow corridor works with LQR cost but not with smooth trajecotry cost
            for t in range(N):
                f += (Z[:, t]-goal).T @ Q @ (Z[:, t]-goal) + V[:, t].T @ R @ V[:, t] + disturbance_penalty*ca.sumsqr(self.m.E_func(Z[:,t]))
            
            # collision avoidance with other agents
            for agent in other_agents:
                for t in range(N):
                    center, rad = agent[t]
                    assert len(center) <= nc
                    if len(center) == 2:
                        f -= ca.sqrt(ca.sumsqr(Z[:2,t] - ca.DM(center)) + 1e-5) * ca_weight
                    elif len(center) == 3:
                        f -= ca.sqrt(ca.sumsqr(Z[:3,t] - ca.DM(center)) + 1e-5) * ca_weight
        else:
            for t in range(N-1):
                f += (Z[:,t+1] - Z[:,t]).T @ Q @ (Z[:,t+1] - Z[:,t]) + disturbance_penalty*ca.sumsqr(self.m.E_func(Z[:,t]))
            f += (Z_f - Z[:,N-1]).T @ Q @ (Z_f - Z[:,N-1]) + disturbance_penalty*ca.sumsqr(self.m.E_func(Z[:,N-1]))

            # collision avoidance with other agents
            for agent in other_agents:
                for t in range(N):
                    center, rad = agent[t]
                    assert len(center) <= nc
                    if len(center) == 2:
                        f -= ca.sqrt(ca.sumsqr(Z[:2,t] - ca.DM(center)) + 1e-5) * ca_weight
                    elif len(center) == 3:
                        f -= ca.sqrt(ca.sumsqr(Z[:3,t] - ca.DM(center)) + 1e-5) * ca_weight
        # proximity with leaders
        for agent in leaders:
            for t in range(N):
                center, rad = agent[t]
                f += ca.sqrt(ca.sumsqr(m.Gcf_2d@Z[:,t] - ca.DM(center)) + 1e-5) * prox_weight
        for agent in followers:
            for t in range(N):
                center, heading, backoff_follower, outer_approx_follower, rad_follower, half_cone_follower = agent[t]
                f += ca.sqrt(ca.sumsqr(m.Gcf_2d@Z[:,t] - ca.DM(center)) + 1e-5) * prox_weight
        
        # constraints
        g = []
        
        for t in range(N-1): 
            # dynamics
            eq = Z[:, t + 1] - self.rk4(Z[:,t], V[:,t])
            # state, input constraints
            ineq = m.G @ ca.vertcat(Z[:, t], V[:, t]) 

            g = ca.vertcat(g, eq)
            g = ca.vertcat(g, ineq)
            
            # collision avoidance with obstacles (assume obstacles are either circle or pillar)
            for obst in obstacles:
                center, rad = obst[t]
                assert len(center) == 2
                
                for _ in range(2):
                    ineq = -ca.sqrt(ca.sumsqr(Z[:2,t] - ca.DM(center)) + 1e-5)
                    g = ca.vertcat(g, ineq)
            
            # collision avoidance with other agents
            for agent in other_agents:
                center, rad = agent[t]
                
                assert len(center) <= nc
                if len(center) == 2:
                    for _ in range(2):
                        ineq = -ca.sqrt(ca.sumsqr(Z[:2,t] - ca.DM(center)) + 1e-5)
                        g = ca.vertcat(g, ineq)
                elif len(center) == 3:
                    for _ in range(3):
                        ineq = -ca.sqrt(ca.sumsqr(Z[:3,t] - ca.DM(center)) + 1e-5)
                        g = ca.vertcat(g, ineq)

            # proximity with leaders
            for agent in leaders:
                center, rad = agent[t]
                for _ in range(nc):
                    ineq = ca.sqrt(ca.sumsqr(m.Gcf_2d@Z[:,t] - ca.DM(center)) + 1e-5)
                    g = ca.vertcat(g, ineq)

            # line-of-sight
            for agent in leaders:
                center, rad = agent[t]
                if nc == 2:
                    heading = Z[2,t]
                    delta_heading = backoff[t,2]
                else:
                    heading = Z[5,t]
                    delta_heading = backoff[t,5]
                
                dx = center[0] - Z[0, t]
                dy = center[1] - Z[1, t]
                r = ca.fmin(ca.sqrt(backoff[t,0]**2+backoff[t,1]**2 + 1e-5), outer_approx[t,0])
                divident = ca.sqrt(dy**2/r**2 + dx**2/r**2 + 1e-5)
                # leftmost point
                x_A_left = Z[0,t] - dy / (divident + 1e-5)
                y_A_left = Z[1,t] + dx / (divident + 1e-5)

                x_diff = x_A_left - center[0]
                y_diff = y_A_left - center[1]
                d_s = x_diff**2 + y_diff**2 + 1e-5
                alpha = rad**2 / d_s
                beta = rad * ca.sqrt(d_s - rad**2 + 1e-5) / d_s
                x_C = center[0] + alpha*x_diff - beta*y_diff
                y_C = center[1] + alpha*y_diff + beta*x_diff

                w2_x = -ca.sin(heading - half_cone + delta_heading)
                w2_y = ca.cos(heading - half_cone + delta_heading)

                ineq = -((x_C-x_A_left)*w2_x + (y_C-y_A_left)*w2_y)
                g = ca.vertcat(g, ineq)
                
                # rightmost point
                x_A_right = Z[0,t] + dy / (divident + 1e-5)
                y_A_right = Z[1,t] - dx / (divident + 1e-5)

                x_diff = x_A_right - center[0]
                y_diff = y_A_right - center[1]
                d_s = x_diff**2 + y_diff**2 + 1e-5
                alpha = rad**2 / d_s
                beta = rad * ca.sqrt(d_s - rad**2 + 1e-5) / d_s
                x_B = center[0] + alpha*x_diff + beta*y_diff
                y_B = center[1] + alpha*y_diff - beta*x_diff

                w1_x = ca.sin(heading + half_cone - delta_heading)
                w1_y = -ca.cos(heading + half_cone - delta_heading)

                ineq = -((x_B-x_A_right)*w1_x + (y_B-y_A_right)*w1_y)
                g = ca.vertcat(g, ineq)
            
            for agent in followers:
                center, heading, backoff_follower, outer_approx_follower, rad_follower, half_cone_follower = agent[t]
                
                dx = -(center[0] - Z[0, t])
                dy = -(center[1] - Z[1, t])
                r = ca.fmin(outer_approx_follower, ca.sqrt(backoff_follower[0]**2 + backoff_follower[1]**2 + 1e-5)) + rad_follower
                divident = ca.sqrt(dy**2/r**2 + dx**2/r**2 + 1e-5)
                # leftmost point
                x_A_left = center[0] - dy / (divident + 1e-5)
                y_A_left = center[1] + dx / (divident + 1e-5)

                x_diff = x_A_left - Z[0,t]
                y_diff = y_A_left - Z[1,t]
                d_s = x_diff**2 + y_diff**2 + 1e-5
                alpha = min_dist**2 / d_s
                beta = min_dist * ca.sqrt(d_s - min_dist**2 + 1e-5) / d_s
                x_C = Z[0,t] + alpha*x_diff - beta*y_diff
                y_C = Z[1,t] + alpha*y_diff + beta*x_diff

                w2_x = -ca.sin(heading - half_cone_follower + backoff_follower[2])
                w2_y = ca.cos(heading - half_cone_follower + backoff_follower[2])

                ineq = -((x_C-x_A_left)*w2_x + (y_C-y_A_left)*w2_y)
                g = ca.vertcat(g, ineq)
                
                # rightmost point
                x_A_right = center[0] + dy / (divident + 1e-5)
                y_A_right = center[1] - dx / (divident + 1e-5)

                x_diff = x_A_right - Z[0,t]
                y_diff = y_A_right - Z[1,t]
                d_s = x_diff**2 + y_diff**2 + 1e-5
                alpha = min_dist**2 / d_s
                beta = min_dist * ca.sqrt(d_s - min_dist**2 + 1e-5) / d_s
                x_B = Z[0,t] + alpha*x_diff + beta*y_diff
                y_B = Z[1,t] + alpha*y_diff - beta*x_diff

                w1_x = ca.sin(heading + half_cone_follower - backoff_follower[2])
                w1_y = -ca.cos(heading + half_cone_follower - backoff_follower[2])

                ineq = -((x_B-x_A_right)*w1_x + (y_B-y_A_right)*w1_y)
                g = ca.vertcat(g, ineq)
        
        eq = Z_f - self.rk4(Z[:,N-1], V[:,N-1])
        ineq = m.G @ ca.vertcat(Z[:,N-1], V[:,N-1])
        g = ca.vertcat(g, eq)
        g = ca.vertcat(g, ineq)
        for obst in obstacles:
            center, rad = obst[N-1]
            assert len(center) == 2
            
            for _ in range(2):
                ineq = -ca.sqrt(ca.sumsqr(Z[:2,N-1] - ca.DM(center)) + 1e-5)
                g = ca.vertcat(g, ineq)
        for agent in other_agents:
            center, rad = agent[N-1]
            
            assert len(center) <= nc
            if len(center) == 2:
                for _ in range(2):
                    ineq = -ca.sqrt(ca.sumsqr(Z[:2,N-1] - ca.DM(center)) + 1e-5)
                    g = ca.vertcat(g, ineq)
            elif len(center) == 3:
                for _ in range(3):
                    ineq = -ca.sqrt(ca.sumsqr(Z[:3,N-1] - ca.DM(center)) + 1e-5)
                    g = ca.vertcat(g, ineq)
        for agent in leaders:
            center, rad = agent[N-1]
            for _ in range(nc):
                ineq = ca.sqrt(ca.sumsqr(m.Gcf_2d@Z[:,N-1] - ca.DM(center)) + 1e-5)
                g = ca.vertcat(g, ineq)
        for agent in leaders:
            center, rad = agent[N-1]
            t = N-1

            if nc == 2:
                heading = Z[2,t]
                delta_heading = backoff[t,2]
            else:
                heading = Z[5,t]
                delta_heading = backoff[t,5]

            dx = center[0] - Z[0, t]
            dy = center[1] - Z[1, t]
            r = ca.fmin(ca.sqrt(backoff[t,0]**2 + backoff[t,1]**2 + 1e-5), outer_approx[t,0])
            divident = ca.sqrt(dy**2/r**2 + dx**2/r**2 + 1e-5)
            # leftmost point
            x_A_left = Z[0,t] - dy / (divident + 1e-5)
            y_A_left = Z[1,t] + dx / (divident + 1e-5)

            x_diff = x_A_left - center[0]
            y_diff = y_A_left - center[1]
            d_s = x_diff**2 + y_diff**2 + 1e-5
            alpha = rad**2 / d_s
            beta = rad * ca.sqrt(d_s - rad**2 + 1e-5) / d_s
            x_C = center[0] + alpha*x_diff - beta*y_diff
            y_C = center[1] + alpha*y_diff + beta*x_diff

            w2_x = -ca.sin(heading - half_cone + delta_heading)
            w2_y = ca.cos(heading - half_cone + delta_heading)

            ineq = -((x_C-x_A_left)*w2_x + (y_C-y_A_left)*w2_y)
            g = ca.vertcat(g, ineq)
            
            # rightmost point
            x_A_right = Z[0,t] + dy / (divident + 1e-5)
            y_A_right = Z[1,t] - dx / (divident + 1e-5)

            x_diff = x_A_right - center[0]
            y_diff = y_A_right - center[1]
            d_s = x_diff**2 + y_diff**2 + 1e-5
            alpha = rad**2 / d_s
            beta = rad * ca.sqrt(d_s - rad**2 + 1e-5) / d_s
            x_B = center[0] + alpha*x_diff + beta*y_diff
            y_B = center[1] + alpha*y_diff - beta*x_diff

            w1_x = ca.sin(heading + half_cone - delta_heading)
            w1_y = -ca.cos(heading + half_cone - delta_heading)

            ineq = -((x_B-x_A_right)*w1_x + (y_B-y_A_right)*w1_y)
            g = ca.vertcat(g, ineq)

        for agent in followers:
            center, heading, backoff_follower, outer_approx_follower, rad_follower, half_cone_follower = agent[N-1]
            t = N-1
            
            dx = -(center[0] - Z[0, t])
            dy = -(center[1] - Z[1, t])
            r = ca.fmin(outer_approx_follower, ca.sqrt(backoff_follower[0]**2 + backoff_follower[1]**2 + 1e-5)) + rad_follower
            divident = ca.sqrt(dy**2/r**2 + dx**2/r**2 + 1e-5)
            # leftmost point
            x_A_left = center[0] - dy / (divident + 1e-5)
            y_A_left = center[1] + dx / (divident + 1e-5)

            x_diff = x_A_left - Z[0,t]
            y_diff = y_A_left - Z[1,t]
            d_s = x_diff**2 + y_diff**2 + 1e-5
            alpha = min_dist**2 / d_s
            beta = min_dist * ca.sqrt(d_s - min_dist**2 + 1e-5) / d_s
            x_C = Z[0,t] + alpha*x_diff - beta*y_diff
            y_C = Z[1,t] + alpha*y_diff + beta*x_diff

            w2_x = -ca.sin(heading - half_cone_follower + backoff_follower[2])
            w2_y = ca.cos(heading - half_cone_follower + backoff_follower[2])

            ineq = -((x_C-x_A_left)*w2_x + (y_C-y_A_left)*w2_y)
            g = ca.vertcat(g, ineq)
            
            # rightmost point
            x_A_right = center[0] + dy / (divident + 1e-5)
            y_A_right = center[1] - dx / (divident + 1e-5)

            x_diff = x_A_right - Z[0,t]
            y_diff = y_A_right - Z[1,t]
            d_s = x_diff**2 + y_diff**2 + 1e-5
            alpha = min_dist**2 / d_s
            beta = min_dist * ca.sqrt(d_s - min_dist**2 + 1e-5) / d_s
            x_B = Z[0,t] + alpha*x_diff + beta*y_diff
            y_B = Z[1,t] + alpha*y_diff - beta*x_diff

            w1_x = ca.sin(heading + half_cone_follower - backoff_follower[2])
            w1_y = -ca.cos(heading + half_cone_follower - backoff_follower[2])

            ineq = -((x_B-x_A_right)*w1_x + (y_B-y_A_right)*w1_y)
            g = ca.vertcat(g, ineq)

        ineq = m.Gf @ Z_f
        g = ca.vertcat(g, ineq)
        for obst in obstacles:
            center, rad = obst[N]
            
            assert len(center) == 2
            for _ in range(2):
                ineq = -ca.sqrt(ca.sumsqr(Z_f[:2] - ca.DM(center)) + 1e-5)
                g = ca.vertcat(g, ineq)
        for agent in other_agents:
            center, rad = agent[N]
            
            assert len(center) <= nc
            if len(center) == 2:
                for _ in range(2):
                    ineq = -ca.sqrt(ca.sumsqr(Z_f[:2] - ca.DM(center)) + 1e-5)
                    g = ca.vertcat(g, ineq)
            elif len(center) == 3:
                for _ in range(3):
                    ineq = -ca.sqrt(ca.sumsqr(Z_f[:3] - ca.DM(center)) + 1e-5)
                    g = ca.vertcat(g, ineq)
        for agent in leaders:
            center, rad = agent[N]
            for _ in range(nc):
                ineq = ca.sqrt(ca.sumsqr(m.Gcf_2d@Z_f - ca.DM(center)) + 1e-5)
                g = ca.vertcat(g, ineq)
        for agent in leaders:
            center, rad = agent[N]

            dx = center[0] - Z_f[0]
            dy = center[1] - Z_f[1]
            r = ca.fmin(ca.sqrt(backoff_f[0]**2 + backoff_f[1]**2 + 1e-5), outer_approx[N,0])
            divident = ca.sqrt(dy**2/r**2 + dx**2/r**2 + 1e-5)
            # leftmost point
            x_A_left = Z_f[0] - dy / (divident + 1e-5)
            y_A_left = Z_f[1] + dx / (divident + 1e-5)

            x_diff = x_A_left - center[0]
            y_diff = y_A_left - center[1]
            d_s = x_diff**2 + y_diff**2 + 1e-5
            alpha = rad**2 / d_s
            beta = rad * ca.sqrt(d_s - rad**2 + 1e-5) / d_s
            x_C = center[0] + alpha*x_diff - beta*y_diff
            y_C = center[1] + alpha*y_diff + beta*x_diff

            w2_x = -ca.sin(Z_f[2] - half_cone + backoff_f[2])
            w2_y = ca.cos(Z_f[2] - half_cone + backoff_f[2])

            ineq = -((x_C-x_A_left)*w2_x + (y_C-y_A_left)*w2_y)
            # g = ca.vertcat(g, ineq)
            g = ca.vertcat(g, -1)
            
            # rightmost point
            x_A_right = Z_f[0] + dy / (divident + 1e-5)
            y_A_right = Z_f[1] - dx / (divident + 1e-5)

            x_diff = x_A_right - center[0]
            y_diff = y_A_right - center[1]
            d_s = x_diff**2 + y_diff**2 + 1e-5
            alpha = rad**2 / d_s
            beta = rad * ca.sqrt(d_s - rad**2 + 1e-5) / d_s
            x_B = center[0] + alpha*x_diff + beta*y_diff
            y_B = center[1] + alpha*y_diff - beta*x_diff

            w1_x = ca.sin(Z_f[2] + half_cone - backoff_f[2])
            w1_y = -ca.cos(Z_f[2] + half_cone - backoff_f[2])

            ineq = -((x_B-x_A_right)*w1_x + (y_B-y_A_right)*w1_y)
            # g = ca.vertcat(g, ineq)
            g = ca.vertcat(g, -1)
        for agent in followers:
            g = ca.vertcat(g, -1)
            g = ca.vertcat(g, -1)
        
        y = ca.vertcat(ca.reshape(Y, N*(self.m.nx+self.m.nu), 1), Z_f)
        self.g_func = ca.Function("g_func", [y], [g])

        self.nlp = {
            'x': y,  # Decision variables
            'f': f,  # Objective function
            'g': g
        }

        self.solver_nlp = ca.nlpsol("solver", self.nlp_solver_name, self.nlp, self.nominal_solver_options)

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