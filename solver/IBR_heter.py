import numpy as np
import time, copy

from solver.fast_SLS_nlp import fast_SLS
from dyn.LTV import LTV

class IBR:
    def __init__(self, T, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all, 
                 N_agent, models, init_states, goals, static_obstacles, max_dists, min_dists, half_cones, LOS_targets, followers, 
                 ca_weight, prox_weight, use_LQR, alpha=0.5):
        """
        :param T: plan horizon
        :param Q_all: regularization of nominal states
        :param R_all: regularization of nominal inputs
        :param Qf_all: regularization of nominal terminal states
        :param Q_reg_all: regularization of Phi_x
        :param R_reg_all: regularization of Phi_u
        :param Qf_reg_all: regularization of last row of Phi_x
        :param N_agent: number of agents
        :param models: dynamic models of agents
        :param init_states: initial states of agents
        :param goals: goal states of agents
        :param static_obstacles: center and radius of static obstacles
        :param max_dists: max distance allowed between leader and follower
        :param min_dists: radius of agents
        :param half_cones: half of FOV of an agent with LOS constraint
        :param LOS_targets: leader of agents
        :param ca_weight: weight of coupled collision avoidance cost
        :param prox_weight: weight of coupled proximity cost
        :param use_LQR: use LQR or smoothness trajectory cost
        :param alpha: update rate for each IBR iteration
        """
        self.T = T
        self.Q_all = Q_all
        self.R_all = R_all
        self.Qf_all = Qf_all
        self.Q_reg_all = Q_reg_all
        self.R_reg_all = R_reg_all
        self.Qf_reg_all = Qf_reg_all

        self.max_dists = max_dists
        self.min_dists = min_dists
        self.half_cones = half_cones
        self.LOS_targets = LOS_targets
        self.followers = followers
        self.use_LQR = use_LQR

        self.ca_weight = ca_weight
        self.prox_weight = prox_weight

        self.N_agent = N_agent
        self.models = models
        self.LTV_models = self.generate_LTV_models(models)
        self.init_states = init_states
        self.goals = goals
        print("init states: ", init_states)
        print("goals: ", goals)

        self.static_obstacles = []
        self.generate_static_obstacles(static_obstacles)

        self.MAX_ITER = 5 
        self.CONV_THRES = 1e-3

        self.solutions = {"nominal_trajs": [np.array([]) for _ in range(N_agent)], 
                          "nominal_inputs": [np.array([]) for _ in range(N_agent)],
                          "nominal_vecs": [np.array([]) for _ in range(N_agent)],
                          "tubes": [np.array([]) for _ in range(N_agent)], 
                          "tubes_f": [np.array([]) for _ in range(N_agent)],
                          "outer_approxes": [None for _ in range(N_agent)],
                          "K_mats": [np.array([]) for _ in range(N_agent)],
                          "phi_x_mats": [np.array([]) for _ in range(N_agent)],
                          "phi_u_mats": [np.array([]) for _ in range(N_agent)],
                          "initial_trajs": [np.array([]) for _ in range(N_agent)],
                          "initial_tubes": [np.array([]) for _ in range(N_agent)],
                          "initial_tubes_f": [np.array([]) for _ in range(N_agent)],
                          "initial_outer_approxes": [None for _ in range(N_agent)],
                          "initial_inputs": [np.array([]) for _ in range(N_agent)],
                          "initial_phi_x_mats": [np.array([]) for _ in range(N_agent)],
                          "initial_phi_u_mats": [np.array([]) for _ in range(N_agent)],
                          "costs": [0] * N_agent,
                          "cost_nominals": [0] * N_agent}

        self.alpha = alpha

    def generate_LTV_models(self, models):
        LTV_models = []
        for m in models:
            init_LTV = LTV(m, self.T)
            init_LTV.g_list = [m.g for _ in range(self.T)]
            LTV_models.append(init_LTV)
        return LTV_models
    
    def generate_static_obstacles(self, static_obstacles):
        for static_obst in static_obstacles:
            obstacle = np.array([static_obst for _ in range(self.T+1)], dtype=object)
            self.static_obstacles.append(obstacle)

    def initialize_solutions(self, init_guess_file):
        init_sol = np.load(init_guess_file, allow_pickle=True)

        for n in range(self.N_agent):
            self.solutions["nominal_trajs"][n] = init_sol["initial_trajs"][n]
            self.solutions["nominal_inputs"][n] = init_sol["initial_inputs"][n]
            self.solutions["initial_trajs"][n] = copy.deepcopy(init_sol["initial_trajs"][n])
            self.solutions["initial_inputs"][n] = copy.deepcopy(init_sol["initial_inputs"][n])
            self.solutions["nominal_vecs"][n] = init_sol["initial_vecs"][n]
            # self.solutions["initial_nominal_vecs"][n] = copy.deepcopy(init_sol["initial_vecs"])[n]
            self.solutions["tubes"][n] = init_sol["initial_tubes"][n]
            self.solutions["tubes_f"][n] = init_sol["initial_tubes_f"][n]
            self.solutions["initial_tubes"][n] = copy.deepcopy(init_sol["initial_tubes"][n])
            self.solutions["initial_tubes_f"][n] = copy.deepcopy(init_sol["initial_tubes_f"][n])
            self.solutions["outer_approxes"][n] = init_sol["initial_outer_approxes"][n]
            self.solutions["initial_outer_approxes"][n] = copy.deepcopy(init_sol["initial_outer_approxes"][n])

    def plan(self, idx, obstacles, other_agents, leaders, followers, it, init_guess=None):
        x0 = self.init_states[idx]
        goal = self.goals[idx]
        Q = self.Q_all[idx]
        R = self.R_all[idx]
        Qf = self.Qf_all[idx]
        Q_reg = self.Q_reg_all[idx]
        R_reg = self.R_reg_all[idx]
        Qf_reg = self.Qf_reg_all[idx]

        solver = fast_SLS(self.T, Q, R, self.LTV_models[idx], Qf, 
                          len(self.static_obstacles), len(other_agents), len(leaders), len(followers), self.models[idx], 
                          Q_reg, R_reg, Qf_reg)
        solver.verbose = True
        if init_guess is None:
            solution = solver.solve(x0, goal, obstacles, other_agents, self.max_dists[idx], self.min_dists[idx], 
                                    leaders, followers, self.half_cones[idx], self.ca_weight, self.prox_weight, self.use_LQR[idx], self.solutions["nominal_vecs"][idx])
        else:
            solution = solver.solve(x0, goal, obstacles, other_agents, self.max_dists[idx], self.min_dists[idx], 
                                    leaders, followers, self.half_cones[idx], self.ca_weight, self.prox_weight, self.use_LQR[idx], init_guess[idx])
            
        if it != self.MAX_ITER:
            solution = solver.get_updated_sol_with_lr(self.init_states[idx], self.solutions["nominal_trajs"][idx], self.solutions["nominal_inputs"][idx], 
                                                      [], [], self.alpha)

        return solution
    
    def generate_dynamic_obstacles(self, idx):
        dynamic_obstacles = []
        
        # project quadcopter reachable set to ground as a 2D elllips to take care of collision avoidance
        if self.models[idx].nc == 3: # quadcopter
            # for i in range(idx):
            for i in range(self.N_agent):
                if i != idx:
                    obstacle = np.empty((self.T+1), dtype=object)
                    traj = self.solutions["nominal_trajs"][i]
                    tube = self.solutions["tubes"][i]
                    tube_f = self.solutions["tubes_f"][i]
                    outer_approx = self.solutions["outer_approxes"][i]
                    nc = self.models[i].nc
                    if nc == 2: # unicycle modeled as a pillar (because quadcopter need to lead followers to avoid collison)
                        for t in range(self.T):
                            center = np.array([traj[0,t], traj[1,t], 0.05])
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (center, reachable_set_approx + self.min_dists[i]) # inflate other agents by their radius
                        center = np.array([traj[0, self.T], traj[1, self.T], 0.05])
                        reachable_set_approx = min(outer_approx[self.T, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.T] = (center, reachable_set_approx + self.min_dists[i])
                    elif nc == 3: # quadcopter
                        for t in range(self.T):
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                        reachable_set_approx = min(outer_approx[self.T, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.T] = (traj[:2, self.T], reachable_set_approx + self.min_dists[i])
                        raise NotImplementedError
                    
                    dynamic_obstacles.append(obstacle)
        elif self.models[idx].nc == 2: # unicycle
            for i in range(self.N_agent):
                if i != idx:
                    obstacle = np.empty((self.T+1), dtype=object)
                    traj = self.solutions["nominal_trajs"][i]
                    tube = self.solutions["tubes"][i]
                    tube_f = self.solutions["tubes_f"][i]
                    outer_approx = self.solutions["outer_approxes"][i]
                    nc = self.models[i].nc
                    if nc == 2: # unicycle
                        for t in range(self.T):
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                        reachable_set_approx = min(outer_approx[self.T, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.T] = (traj[:2, self.T], reachable_set_approx + self.min_dists[i])
                    elif nc == 3: # quadcopter (maybe can be omitted when flying high enough)
                        # continue
                        if np.all(traj[2,:] >= 0.15):
                            # print("omit")
                            continue
                        for t in range(self.T):
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                        reachable_set_approx = min(outer_approx[self.T, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.T] = (traj[:2, self.T], reachable_set_approx + self.min_dists[i])
                    else:
                        raise NotImplementedError
                    
                    dynamic_obstacles.append(obstacle)
        else:
            raise NotImplementedError

        return dynamic_obstacles
    
    def generate_leaders(self, idx):
        leaders = []
        
        for i in self.LOS_targets[idx]:
            target = np.empty((self.T+1), dtype=object)
            traj = self.solutions["nominal_trajs"][i]
            tube = self.solutions["tubes"][i]
            tube_f = self.solutions["tubes_f"][i]
            outer_approx = self.solutions["outer_approxes"][i]
            nc = self.models[i].nc
            for t in range(self.T):
                reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                target[t] = (traj[:2,t], reachable_set_approx + self.min_dists[i])
            reachable_set_approx = min(outer_approx[self.T, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
            target[self.T] = (traj[:2, self.T], reachable_set_approx + self.min_dists[i])
            leaders.append(target)
        return leaders
    
    def generate_followers(self, idx):
        followers = []

        for i in self.followers[idx]:
            target = np.empty((self.T+1), dtype=object)
            traj = self.solutions["nominal_trajs"][i]
            tube = self.solutions["tubes"][i]
            tube_f = self.solutions["tubes_f"][i]
            outer_approx = self.solutions["outer_approxes"][i]
            nc = self.models[i].nc
            for t in range(self.T):
                target[t] = (traj[:2,t], traj[2,t], tube[t,:3], outer_approx[t,0], self.min_dists[i], self.half_cones[i])
            target[self.T] = (traj[:2, self.T], traj[2,self.T], tube_f[:3], outer_approx[self.T,0], self.min_dists[i], self.half_cones[i])
            followers.append(target)

        return followers

    def plan_all(self, init_guess=None):
        forward_runtime = 0
        backward_runtime = 0
        tighten_runtime = 0
        for it in range(self.MAX_ITER):
            print("start IBR iteration: ", it)
            convergence = np.zeros(self.N_agent, dtype=np.int8)

            for idx in range(self.N_agent):
                dynamic_obstacles = self.generate_dynamic_obstacles(idx)
                leaders = self.generate_leaders(idx)
                followers = self.generate_followers(idx)

                sol = self.plan(idx, self.static_obstacles, dynamic_obstacles, leaders, followers, it, init_guess)
                if not sol["success"]:
                    print("IBR failed at iteration {:} for agent {:}".format(it, idx))
                    return self.solutions, False
                
                convergence[idx] = np.all(np.fabs(sol["primal_x"] - self.solutions["nominal_trajs"][idx]) <= self.CONV_THRES)

                forward_runtime += sol["forward_time"]
                backward_runtime += sol["backward_time"]
                tighten_runtime += sol["tighten_time"]
                
                self.solutions["nominal_trajs"][idx] = sol["primal_x"].copy()
                self.solutions["nominal_inputs"][idx] = sol["primal_u"].copy()
                self.solutions["tubes"][idx] = sol["backoff"].copy()
                self.solutions["tubes_f"][idx] = sol["backoff_f"].copy()
                self.solutions["outer_approxes"][idx] = sol["outer_approx"].copy()
                self.solutions["K_mats"][idx] = sol["K_mat"].copy()
                self.solutions["phi_x_mats"][idx] = sol["Phi_x_mat"].copy()
                self.solutions["phi_u_mats"][idx] = sol["Phi_u_mat"].copy()
                self.solutions["nominal_vecs"][idx] = sol["primal_vec"].copy()
                self.solutions["costs"][idx] = sol["cost"]
                self.solutions["cost_nominals"][idx] = sol["cost_nominal"]

            print("end IBR iteration: ", it)

            if np.all(convergence):
                print("IBR converged")
                print("forward: {:.3f}".format(forward_runtime))
                print("backward: {:.3f}".format(backward_runtime))
                print("tightening: {:.3f}".format(tighten_runtime))
                return self.solutions, True

        print("IBR didn't converge in {:} iterations".format(self.MAX_ITER))
        print("The result is still valid")
        print("forward: {:.3f}".format(forward_runtime))
        print("backward: {:.3f}".format(backward_runtime))
        print("tightening: {:.3f}".format(tighten_runtime))
        return self.solutions, False
