import matplotlib.pyplot as plt
import numpy as np
import time
from solver.fast_SLS_nlp import fast_SLS
from dyn.LTV import LTV
from solver.centralized_init import NLP
import copy

class IBR:
    def __init__(self, N, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all, 
                 N_agent, models, init_states, goals, static_obstacles, max_dists, min_dists, half_cones, LOS_targets,
                 ca_weight, prox_weight, use_LQR, init_guess_file=None, lr=0.2):
        # static obstalces: list of tuples (center: numpy array, radius)
        
        self.N = N
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
        self.use_LQR = use_LQR
        
        self.ca_weight = ca_weight
        self.prox_weight = prox_weight
        self.init_guess_file = init_guess_file

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

        # self.solvers = self.initialize_solvers()
        self.solutions = {"nominal_trajs": [np.array([]) for _ in range(N_agent)], 
                          "nominal_inputs": [np.array([]) for _ in range(N_agent)],
                          "nominal_vecs": [np.array([]) for _ in range(N_agent)],
                          "tubes": [np.array([]) for _ in range(N_agent)], 
                          "tubes_f": [np.array([]) for _ in range(N_agent)],
                          "outer_approxes": [None for _ in range(N_agent)],
                          "etas": [None for _ in range(N_agent)],
                          "eta_fs": [None for _ in range(N_agent)],
                          "K_mats": [np.array([]) for _ in range(N_agent)],
                          "phi_x_mats": [np.array([]) for _ in range(N_agent)],
                          "phi_u_mats": [np.array([]) for _ in range(N_agent)],
                          "initial_trajs": [np.array([]) for _ in range(N_agent)],
                          "initial_tubes": [np.array([]) for _ in range(N_agent)],
                          "initial_nominal_vecs": [None for _ in range(N_agent)],
                          "initial_outer_approxes": [None for _ in range(N_agent)],
                          "initial_inputs": [np.array([]) for _ in range(N_agent)],
                          "initial_phi_x_mats": [np.array([]) for _ in range(N_agent)],
                          "initial_phi_u_mats": [np.array([]) for _ in range(N_agent)],
                          "costs": [0] * N_agent,
                          "cost_nominals": [0] * N_agent}

        self.initialize_solutions()

        self.lr = lr 

    def generate_LTV_models(self, models):
        LTV_models = []
        for m in models:
            init_LTV = LTV(m, self.N)
            init_LTV.g_list = [m.g for _ in range(self.N)]
            LTV_models.append(init_LTV)
        return LTV_models

    # def initialize_solvers(self):
    #     solvers = []
    #     for idx, m in enumerate(self.models):
    #         Q = self.Q_all[idx]
    #         R = self.R_all[idx]
    #         Qf = self.Qf_all[idx]
    #         fast_SLS_solver = fast_SLS(self.N, Q, R, self.LTV_models[idx], Qf, 
    #                                    len(self.static_obstacles)+self.N_agent-1, m, 
    #                                     Q*100, R*100, Qf*100)
    #         solvers.append(fast_SLS_solver)
    #     return solvers
    
    def generate_static_obstacles(self, static_obstacles):
        for static_obst in static_obstacles:
            obstacle = np.array([static_obst for _ in range(self.N+1)], dtype=object)
            self.static_obstacles.append(obstacle)

    # def initialize_solutions(self, init_guess=None):
    #     print("start initialize solutions")
    #     start = time.perf_counter()

    #     for idx in range(self.N_agent):
    #         x0 = self.init_states[idx]
    #         goal = self.goals[idx]
    #         Q = self.Q_all[idx]
    #         R = self.R_all[idx]
    #         Qf = self.Qf_all[idx]
    #         Q_reg = self.Q_reg_all[idx]
    #         R_reg = self.R_reg_all[idx]
    #         Qf_reg = self.Qf_reg_all[idx]
            
    #         solver = fast_SLS(self.N, Q, R, self.LTV_models[idx], Qf, 
    #                           len(self.static_obstacles), 0, 0, 0, self.models[idx],
    #                           Q_reg, R_reg, Qf_reg, self.solver_name)
    #         solver.verbose = True
            
    #         if init_guess != None:
    #             solution = solver.solve(x0, goal, self.static_obstacles, [], self.max_dists[idx], self.min_dists[idx], [], [], 0, False, self.use_local_Lip, init_guess[idx])
    #         else:
    #             solution = solver.solve(x0, goal, self.static_obstacles, [], self.max_dists[idx], self.min_dists[idx], [], [], 0, self.use_LQR[idx], self.use_local_Lip)

    #         self.solutions["nominal_trajs"][idx] = copy.deepcopy(solution["primal_x"])
    #         self.solutions["nominal_inputs"][idx] = copy.deepcopy(solution["primal_u"])
    #         self.solutions["nominal_vecs"][idx] = copy.deepcopy(solution["primal_vec"])
    #         # print(solution["primal_vec"])
    #         self.solutions["tubes"][idx] = solution["backoff"]#[:,:self.models[idx].nx]
    #         self.solutions["tubes_f"][idx] = solution["backoff_f"]#[:self.models[idx].nx]
    #         self.solutions["outer_approxes"][idx] = solution["outer_approx"]
    #         self.solutions["etas"][idx] = solution["eta"]
    #         self.solutions["eta_fs"][idx] = solution["eta_f"]
    #         self.solutions["K_mats"][idx] = solution["K_mat"]
    #         self.solutions["phi_x_mats"][idx] = solution["Phi_x_mat"]
    #         self.solutions["phi_u_mats"][idx] = solution["Phi_u_mat"]
    #         self.solutions["initial_trajs"][idx] = copy.deepcopy(solution["primal_x"])
    #         self.solutions["initial_tubes"][idx] = copy.deepcopy(solution["backoff"])
    #         self.solutions["initial_outer_approxes"][idx] = copy.deepcopy(solution["outer_approx"]) 
    #         self.solutions["initial_inputs"][idx] = copy.deepcopy(solution["primal_u"])
    #         self.solutions["initial_phi_x_mats"][idx] = solution["Phi_x_mat"]
    #         self.solutions["initial_phi_u_mats"][idx] = solution["Phi_u_mat"]
    #         self.solutions["costs"][idx] = solution["cost"]
    #         self.solutions["cost_nominals"][idx] = solution["cost_nominal"]
        
    #     end = time.perf_counter()
    #     print("end initialize solutions")
    #     print("initialization time: {:.5f}".format(end-start))

    def initialize_solutions(self, init_guess=None):
        print("start initialize solutions")
        start = time.perf_counter()

        if self.init_guess_file is not None:
            init_sol = np.load(self.init_guess_file)
            
            for n in range(self.N_agent):
                self.solutions["nominal_trajs"][n] = init_sol["initial_trajs"][n]
                self.solutions["nominal_inputs"][n] = init_sol["initial_inputs"][n]
                self.solutions["initial_trajs"][n] = copy.deepcopy(init_sol["initial_trajs"][n])
                self.solutions["initial_inputs"][n] = copy.deepcopy(init_sol["initial_inputs"][n])
                self.solutions["nominal_vecs"][n] = init_sol["initial_vecs"][n]
                self.solutions["initial_nominal_vecs"][n] = copy.deepcopy(init_sol["initial_vecs"])[n]
                self.solutions["tubes"][n] = init_sol["initial_tubes"][n]
                self.solutions["tubes_f"][n] = init_sol["initial_tubes_f"][n]
                self.solutions["initial_tubes"][n] = copy.deepcopy(init_sol["initial_tubes"][n])
                self.solutions["outer_approxes"][n] = init_sol["initial_outer_approxes"][n]
                self.solutions["initial_outer_approxes"][n] = copy.deepcopy(init_sol["initial_outer_approxes"][n])
            return
        
        # offsets = [0.2, 0.23, 0.25] # N=16
        # offsets = [0.1, 0.1] # N=16 2
        offsets = [0.1, 0.1, 0.25, 0.8] # N=24
        # offsets = [0.1, 0.1, 0.25, 0.25] # N=32
        if self.N_agent > 8 and self.N_agent % 8 == 0:
            print("break to several 8-agent teams")
            for it in range(int(self.N_agent / 8)):
                Q_all = self.Q_all[it*8:(it+1)*8]
                R_all = self.R_all[it*8:(it+1)*8]
                LTV_models = self.LTV_models[it*8:(it+1)*8]
                Qf_all = self.Qf_all[it*8:(it+1)*8]
                init_states = self.init_states[it*8:(it+1)*8]
                goals = self.goals[it*8:(it+1)*8]
                min_dists = np.array(self.min_dists[it*8:(it+1)*8]) + offsets[it]

                solver = NLP(self.N, Q_all, R_all, LTV_models, Qf_all, init_states, goals, self.static_obstacles, 8, min_dists)
            
                solution = solver.solve(init_states)

                idx_start = 0 
                idx_end = 0
                for n in range(it*8, (it+1)*8):
                    m = self.models[n]
                    idx_end += (m.nx+m.nu)*self.N + m.nx
                    self.solutions["nominal_vecs"][n] = solution["primal_vec"][idx_start:idx_end]
                    self.solutions["nominal_trajs"][n] = copy.deepcopy(solution["primal_x_all"][n-it*8])
                    self.solutions["nominal_inputs"][n] = copy.deepcopy(solution["primal_u_all"][n-it*8])
                    self.solutions["initial_trajs"][n] = copy.deepcopy(solution["primal_x_all"][n-it*8])
                    self.solutions["initial_inputs"][n] = copy.deepcopy(solution["primal_u_all"][n-it*8])
                    idx_start = idx_end
                    
                    # self.solutions["tubes"][n] = np.zeros((self.N, m.nx)) 
                    # self.solutions["tubes_f"][n] = np.zeros((m.nx)) 
                    # self.solutions["initial_tubes"][n] = np.zeros((self.N, m.nx)) 
                    # self.solutions["outer_approxes"][n] = np.zeros((self.N+1, m.nx)) 
                    # self.solutions["initial_outer_approxes"][n] = np.zeros((self.N+1, m.nx)) 
                    # N=16 24
                    self.solutions["tubes"][n] = np.zeros((self.N, m.nx)) + 0.05
                    self.solutions["tubes_f"][n] = np.zeros((m.nx)) + 0.05
                    self.solutions["initial_tubes"][n] = np.zeros((self.N, m.nx)) + 0.05
                    self.solutions["outer_approxes"][n] = np.zeros((self.N+1, m.nx)) + 0.075
                    self.solutions["initial_outer_approxes"][n] = np.zeros((self.N+1, m.nx)) + 0.075
                    # self.solutions["tubes"][n] = np.zeros((self.N, m.nx)) + 0.05
                    # self.solutions["tubes_f"][n] = np.zeros((m.nx)) + 0.05
                    # self.solutions["initial_tubes"][n] = np.zeros((self.N, m.nx)) + 0.05
                    # self.solutions["outer_approxes"][n] = np.zeros((self.N+1, m.nx)) + 0.075
                    # self.solutions["initial_outer_approxes"][n] = np.zeros((self.N+1, m.nx)) + 0.075
        else:
            min_dists = np.array(self.min_dists) 
            solver = NLP(self.N, self.Q_all, self.R_all, self.LTV_models, self.Qf_all, self.init_states, self.goals, self.static_obstacles, self.N_agent, min_dists)
            
            solution = solver.solve(self.init_states)

            self.solutions["nominal_trajs"] = copy.deepcopy(solution["primal_x_all"])
            self.solutions["nominal_inputs"] = copy.deepcopy(solution["primal_u_all"])
            idx_start = 0 
            idx_end = 0
            for n in range(self.N_agent):
                m = self.models[n]
                idx_end += (m.nx+m.nu)*self.N + m.nx
                self.solutions["nominal_vecs"][n] = solution["primal_vec"][idx_start:idx_end]
                self.solutions["initial_nominal_vecs"][n] = copy.deepcopy(solution["primal_vec"][idx_start:idx_end])
                idx_start = idx_end

                self.solutions["tubes"][n] = np.zeros((self.N, m.nx)) #+ 0.05
                self.solutions["tubes_f"][n] = np.zeros((m.nx)) #+ 0.05
                self.solutions["initial_tubes"][n] = np.zeros((self.N, m.nx)) #+ 0.05
                self.solutions["outer_approxes"][n] = np.zeros((self.N+1, m.nx)) #+ 0.075
                self.solutions["initial_outer_approxes"][n] = np.zeros((self.N+1, m.nx)) #+ 0.075
            self.solutions["initial_trajs"] = copy.deepcopy(solution["primal_x_all"])
            self.solutions["initial_inputs"] = copy.deepcopy(solution["primal_u_all"])
        
        end = time.perf_counter()
        print("end initialize solutions")
        print("initialization time: {:.5f}".format(end-start))

    def plan(self, idx, obstacles, other_agents, LOS_targets, it, init_guess=None):
        x0 = self.init_states[idx]
        goal = self.goals[idx]
        Q = self.Q_all[idx]
        R = self.R_all[idx]
        Qf = self.Qf_all[idx]
        Q_reg = self.Q_reg_all[idx]
        R_reg = self.R_reg_all[idx]
        Qf_reg = self.Qf_reg_all[idx]

        solver = fast_SLS(self.N, Q, R, self.LTV_models[idx], Qf, 
                                       len(self.static_obstacles), len(other_agents), len(LOS_targets), 0, self.models[idx], 
                                        Q_reg, R_reg, Qf_reg)
        solver.verbose = True
        if init_guess is None:
            solution = solver.solve(x0, goal, obstacles, other_agents, self.max_dists[idx], self.min_dists[idx], 
                                    LOS_targets, [], self.half_cones[idx], self.ca_weight, self.prox_weight, self.use_LQR[idx], self.solutions["nominal_vecs"][idx])
        else:
            solution = solver.solve(x0, goal, obstacles, other_agents, self.max_dists[idx], self.min_dists[idx], 
                                    LOS_targets, [], self.half_cones[idx], self.ca_weight, self.prox_weight, self.use_LQR[idx], init_guess[idx])
        
        if it != self.MAX_ITER-1:
            solution = solver.get_updated_sol_with_lr(self.init_states[idx], self.solutions["nominal_trajs"][idx], self.solutions["nominal_inputs"][idx], 
                                                  self.solutions["etas"][idx], self.solutions["eta_fs"][idx], self.lr)

        return solution
    
    def generate_dynamic_obstacles(self, idx):
        dynamic_obstacles = []
        for i in range(self.N_agent):
            if i != idx:
                obstacle = np.empty((self.N+1), dtype=object)
                traj = self.solutions["nominal_trajs"][i]
                tube = self.solutions["tubes"][i]
                tube_f = self.solutions["tubes_f"][i]
                outer_approx = self.solutions["outer_approxes"][i]
                nc = self.models[i].nc
                for t in range(self.N):
                    reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                    obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                obstacle[self.N] = (traj[:2, self.N], reachable_set_approx + self.min_dists[i])
                
                dynamic_obstacles.append(obstacle)

        return dynamic_obstacles
    
    def generate_LOS_targets(self, idx):
        LOS_targets = []
        for i in self.LOS_targets[idx]:
            target = np.empty((self.N+1), dtype=object)
            traj = self.solutions["nominal_trajs"][i]
            tube = self.solutions["tubes"][i]
            tube_f = self.solutions["tubes_f"][i]
            outer_approx = self.solutions["outer_approxes"][i]
            nc = self.models[i].nc
            for t in range(self.N):
                reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                target[t] = (traj[:2,t], reachable_set_approx + self.min_dists[i])
            reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
            target[self.N] = (traj[:2, self.N], reachable_set_approx + self.min_dists[i])
            LOS_targets.append(target)

        return LOS_targets

    def plan_all(self, init_guess=None):
        forward_runtime = 0
        backward_runtime = 0
        tighten_runtime = 0
        for it in range(self.MAX_ITER):
            print("start IBR iteration: ", it)
            convergence = np.zeros(self.N_agent, dtype=np.int8)

            for idx in range(self.N_agent):
                dynamic_obstacles = self.generate_dynamic_obstacles(idx)
                LOS_targets = self.generate_LOS_targets(idx)

                sol = self.plan(idx, self.static_obstacles, dynamic_obstacles, LOS_targets, it, init_guess)
                if not sol["success"]:
                    print("IBR failed at iteration {:} for agent {:}".format(it, idx))
                    return self.solutions, False
                
                convergence[idx] = np.all(np.fabs(sol["primal_x"] - self.solutions["nominal_trajs"][idx]) <= self.CONV_THRES)
                # print(np.max(np.fabs(sol["primal_x"] - self.solutions["nominal_trajs"][idx])))

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
                self.solutions["etas"][idx] = sol["eta"]
                self.solutions["eta_fs"][idx] = sol["eta_f"]

            print("end IBR iteration: ", it)

            if np.all(convergence):
                print("IBR converged")
                print("forward: {:.3f}".format(forward_runtime))
                print("backward: {:.3f}".format(backward_runtime))
                print("tightening: {:.3f}".format(tighten_runtime))
                return self.solutions, True

        print("IBR failed to converge")
        print("forward: {:.3f}".format(forward_runtime))
        print("backward: {:.3f}".format(backward_runtime))
        print("tightening: {:.3f}".format(tighten_runtime))
        return self.solutions, False
    