import matplotlib.pyplot as plt
import numpy as np
import time
from solver.fast_SLS_nlp import fast_SLS
from dyn.LTV import LTV
from solver.centralized_init import NLP
import copy

class IBR:
    def __init__(self, N, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all, 
                 N_agent, models, init_states, goals, static_obstacles, max_dists, min_dists, half_cones, LOS_targets, followers, 
                 ca_weight, prox_weight, use_LQR, init_guess_file=None, lr=0.5):
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
        self.followers = followers
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

    def initialize_solutions(self):
        print("start initialize solutions")
        start = time.perf_counter()

        if self.init_guess_file is not None:
            init_sol = np.load(self.init_guess_file, allow_pickle=True)
            
            for n in range(self.N_agent):
                self.solutions["nominal_trajs"][n] = init_sol["initial_trajs"][n]
                # print(init_sol["initial_trajs"][n].shape)
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

            end = time.perf_counter()
            print("end initialize solutions")
            print("initialization time: {:.5f}".format(end-start))
            return

        # initialize quadcopter in centralized way
        lead_models = [self.LTV_models[i] for i in range(0, self.N_agent, 3)]
        lead_init_states = [self.init_states[i] for i in range(0, self.N_agent, 3)]
        lead_goals = [self.goals[i] for i in range(0, self.N_agent, 3)]
        lead_min_dists = [self.min_dists[i]+0.15 for i in range(0, self.N_agent ,3)]
        # lead_min_dists = [self.min_dists[i]+0.2 for i in range(0, self.N_agent ,3)] # Robotarium
        # lead_min_dists = [self.min_dists[i]+0.25 for i in range(0, self.N_agent ,3)]
        solver = NLP(self.N, [self.Q_all[0]]*int(self.N_agent/3), [self.R_all[0]]*int(self.N_agent/3), lead_models, [self.Qf_all[0]]*int(self.N_agent/3), lead_init_states, lead_goals, self.static_obstacles, int(self.N_agent/3), lead_min_dists)
        
        solution = solver.solve(lead_init_states)
        
        idx_start = 0 
        idx_end = 0
        for idx in range(int(self.N_agent/3)):
            self.solutions["nominal_trajs"][3*idx] = copy.deepcopy(solution["primal_x_all"][idx])
            self.solutions["nominal_inputs"][3*idx] = copy.deepcopy(solution["primal_u_all"][idx])
            
            m = self.models[3*idx]
            idx_end += (m.nx+m.nu)*self.N + m.nx
            self.solutions["nominal_vecs"][3*idx] = solution["primal_vec"][idx_start:idx_end]
            idx_start = idx_end

            self.solutions["tubes"][3*idx] = np.zeros((self.N, m.nx))
            self.solutions["tubes_f"][3*idx] = np.zeros((m.nx))
            self.solutions["outer_approxes"][3*idx] = np.zeros((self.N+1, m.nx))
            self.solutions["initial_tubes"][3*idx] = np.zeros((self.N, m.nx))
            self.solutions["initial_tubes_f"][3*idx] = np.zeros((m.nx))
            self.solutions["initial_outer_approxes"][3*idx] = np.zeros((self.N+1, m.nx))
            self.solutions["initial_trajs"][3*idx] = copy.deepcopy(solution["primal_x_all"][idx])
            self.solutions["initial_inputs"][3*idx] = copy.deepcopy(solution["primal_u_all"][idx])
        # return
        for idx in range(self.N_agent):
            if idx % 3 != 0:
                x0 = self.init_states[idx]
                goal = self.goals[idx]
                Q = self.Q_all[idx]
                R = self.R_all[idx]
                Qf = self.Qf_all[idx]
                # Qf = 10 * np.array([[1,0,0,0],
                #                     [0,1,0,0],
                #                     [0,0,0,0],
                #                     [0,0,0,1]])
                Q_reg = self.Q_reg_all[idx]
                R_reg = self.R_reg_all[idx]
                Qf_reg = self.Qf_reg_all[idx]
                
                # Let followers follow to initialize
                solver = fast_SLS(self.N, Q, R, self.LTV_models[idx], Qf, 
                                len(self.static_obstacles), 0, len(self.LOS_targets[idx]), 0, self.models[idx],
                                Q_reg, R_reg, Qf_reg)
                solver.verbose = True

                leaders = self.generate_leaders(idx)
                followers = self.generate_followers(idx)
                half_cones = [np.pi/6 for _ in range(self.N_agent)]
                # half_cones = self.half_cones
               
                solution = solver.solve(x0, goal, self.static_obstacles, [], self.max_dists[idx], self.min_dists[idx], leaders, [], half_cones[idx], self.ca_weight, self.prox_weight, self.use_LQR[idx])

                self.solutions["nominal_trajs"][idx] = solution["primal_x"]
                self.solutions["nominal_inputs"][idx] = solution["primal_u"]
                self.solutions["nominal_vecs"][idx] = solution["primal_vec"]
                self.solutions["tubes"][idx] = solution["backoff"]#[:,:self.models[idx].nx]
                self.solutions["tubes_f"][idx] = solution["backoff_f"]#[:self.models[idx].nx]
                self.solutions["outer_approxes"][idx] = solution["outer_approx"]
                self.solutions["K_mats"][idx] = solution["K_mat"].copy()
                self.solutions["phi_x_mats"][idx] = solution["Phi_x_mat"].copy()
                self.solutions["phi_u_mats"][idx] = solution["Phi_u_mat"].copy()
                self.solutions["initial_trajs"][idx] = solution["primal_x"]
                self.solutions["initial_tubes"][idx] = solution["backoff"]
                self.solutions["initial_tubes_f"][idx] = solution["backoff_f"]
                self.solutions["initial_outer_approxes"][idx] = solution["outer_approx"]
                self.solutions["initial_inputs"][idx] = solution["primal_u"]
                self.solutions["initial_phi_x_mats"][idx] = solution["Phi_x_mat"]
                self.solutions["initial_phi_u_mats"][idx] = solution["Phi_u_mat"]
                self.solutions["costs"][idx] = solution["cost"]
                self.solutions["cost_nominals"][idx] = solution["cost_nominal"]
            
        
        # for idx in range(self.N_agent):
        #     x0 = self.init_states[idx]
        #     goal = self.goals[idx]
        #     Q = self.Q_all[idx]
        #     R = self.R_all[idx]
        #     Qf = self.Qf_all[idx]
        #     Q_reg = self.Q_reg_all[idx]
        #     R_reg = self.R_reg_all[idx]
        #     Qf_reg = self.Qf_reg_all[idx]
            
        #     # Let followers follow to initialize
        #     solver = fast_SLS(self.N, Q, R, self.LTV_models[idx], Qf, 
        #                       len(self.static_obstacles), 0, len(self.LOS_targets[idx]), 0, self.models[idx],
        #                       Q_reg, R_reg, Qf_reg, self.solver_name)
        #     solver.verbose = True

        #     leaders = self.generate_leaders(idx)
            
        #     if init_guess != None:
        #         solution = solver.solve(x0, goal, self.static_obstacles, [], self.max_dists[idx], self.min_dists[idx], [], [], 0, False, self.use_local_Lip, init_guess[idx])
        #     else:
        #         solution = solver.solve(x0, goal, self.static_obstacles, [], self.max_dists[idx], self.min_dists[idx], leaders, [], self.half_cones[idx], self.use_LQR[idx], self.use_local_Lip)

        #     self.solutions["nominal_trajs"][idx] = solution["primal_x"]
        #     self.solutions["nominal_inputs"][idx] = solution["primal_u"]
        #     self.solutions["nominal_vecs"][idx] = solution["primal_vec"]
        #     self.solutions["tubes"][idx] = solution["backoff"]#[:,:self.models[idx].nx]
        #     self.solutions["tubes_f"][idx] = solution["backoff_f"]#[:self.models[idx].nx]
        #     self.solutions["outer_approxes"][idx] = solution["outer_approx"]
        #     self.solutions["K_mats"][idx] = solution["K_mat"]
        #     self.solutions["phi_x_mats"][idx] = solution["Phi_x_mat"]
        #     self.solutions["phi_u_mats"][idx] = solution["Phi_u_mat"]
        #     self.solutions["initial_trajs"][idx] = solution["primal_x"]
        #     self.solutions["initial_tubes"][idx] = solution["backoff"]
        #     self.solutions["initial_tubes_f"][idx] = solution["backoff_f"]
        #     self.solutions["initial_inputs"][idx] = solution["primal_u"]
        #     self.solutions["initial_phi_x_mats"][idx] = solution["Phi_x_mat"]
        #     self.solutions["initial_phi_u_mats"][idx] = solution["Phi_u_mat"]
        #     self.solutions["costs"][idx] = solution["cost"]
        #     self.solutions["cost_nominals"][idx] = solution["cost_nominal"]
        
        end = time.perf_counter()
        print("end initialize solutions")
        print("initialization time: {:.5f}".format(end-start))

    # def initialize_solutions(self, init_guess=None):
    #     print("start initialize solutions")
    #     start = time.perf_counter()

    #     solver = NLP(self.N, self.Q_all, self.R_all, self.LTV_models, self.Qf_all, self.init_states, self.goals, self.static_obstacles, self.N_agent, self.min_dists)
        
    #     solution = solver.solve(self.init_states)

    #     self.solutions["nominal_trajs"] = copy.deepcopy(solution["primal_x_all"])
    #     self.solutions["nominal_inputs"] = copy.deepcopy(solution["primal_u_all"])
    #     idx_start = 0 
    #     idx_end = 0
    #     for n in range(self.N_agent):
    #         m = self.models[n]
    #         idx_end += (m.nx+m.nu)*self.N + m.nx
    #         self.solutions["nominal_vecs"][n] = solution["primal_vec"][idx_start:idx_end]
    #         idx_start = idx_end

    #         self.solutions["tubes"][n] = np.zeros((self.N, m.nx))
    #         self.solutions["tubes_f"][n] = np.zeros((m.nx))
    #         self.solutions["outer_approxes"][n] = np.zeros((self.N+1, m.nx))
    #         self.solutions["initial_tubes"][n] = np.zeros((self.N, m.nx))
    #         self.solutions["initial_tubes_f"][n] = np.zeros((m.nx))
    #         self.solutions["initial_outer_approxes"][n] = np.zeros((self.N+1, m.nx))
    #     self.solutions["initial_trajs"] = copy.deepcopy(solution["primal_x_all"])
    #     self.solutions["initial_inputs"] = copy.deepcopy(solution["primal_u_all"])
        
    #     # self.solutions["etas"][idx] = solution["eta"]
    #     # self.solutions["eta_fs"][idx] = solution["eta_f"]
    #     # self.solutions["K_mats"][idx] = solution["K_mat"]
    #     # self.solutions["phi_x_mats"][idx] = solution["Phi_x_mat"]
    #     # self.solutions["phi_u_mats"][idx] = solution["Phi_u_mat"]
    #     # self.solutions["initial_phi_x_mats"][idx] = solution["Phi_x_mat"]
    #     # self.solutions["initial_phi_u_mats"][idx] = solution["Phi_u_mat"]
    #     # self.solutions["costs"][idx] = solution["cost"]
    #     # self.solutions["cost_nominals"][idx] = solution["cost_nominal"]
        
    #     end = time.perf_counter()
    #     print("end initialize solutions")
    #     print("initialization time: {:.5f}".format(end-start))


    def plan(self, idx, obstacles, other_agents, leaders, followers, it, init_guess=None):
        x0 = self.init_states[idx]
        goal = self.goals[idx]
        Q = self.Q_all[idx]
        R = self.R_all[idx]
        Qf = self.Qf_all[idx]
        Q_reg = self.Q_reg_all[idx]
        R_reg = self.R_reg_all[idx]
        Qf_reg = self.Qf_reg_all[idx]

        solver = fast_SLS(self.N, Q, R, self.LTV_models[idx], Qf, 
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
                                                  [], [], self.lr)

        return solution
    
    def generate_dynamic_obstacles(self, idx):
        dynamic_obstacles = []
        # for i in range(self.N_agent):
        #     if i != idx:
        #         obstacle = np.empty((self.N+1), dtype=object)
        #         traj = self.solutions["nominal_trajs"][i]
        #         tube = self.solutions["tubes"][i]
        #         nc = self.models[i].nc
        #         for t in range(self.N):
        #             obstacle[t] = (traj[:nc, t], np.max(tube[t,:nc]))
        #         obstacle[self.N] = (traj[:nc, self.N], np.max(self.solutions["tubes_f"][i][:nc]))
                
        #         dynamic_obstacles.append(obstacle)
        # project quadcopter reachable set to ground as a 2D elllips to take care of collision avoidance
        if self.models[idx].nc == 3: # quadcopter
            # for i in range(idx):
            for i in range(self.N_agent):
                if i != idx:
                    obstacle = np.empty((self.N+1), dtype=object)
                    traj = self.solutions["nominal_trajs"][i]
                    tube = self.solutions["tubes"][i]
                    tube_f = self.solutions["tubes_f"][i]
                    outer_approx = self.solutions["outer_approxes"][i]
                    nc = self.models[i].nc
                    if nc == 2: # unicycle modeled as a pillar (because quadcopter need to lead followers to avoid collison)
                        for t in range(self.N):
                            center = np.array([traj[0,t], traj[1,t], 0.05])
                            # center = traj[:2, t]
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (center, reachable_set_approx + self.min_dists[i]) # inflate other agents by their radius
                        center = np.array([traj[0, self.N], traj[1, self.N], 0.05])
                        # center = traj[:2, self.N]
                        reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.N] = (center, reachable_set_approx + self.min_dists[i])
                    elif nc == 3: # quadcopter
                        for t in range(self.N):
                            # reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2+tube[t,2]**2))
                            # obstacle[t] = (traj[:3, t], reachable_set_approx + self.min_dists[i]) # inflate other agents by their radius
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                        reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.N] = (traj[:2, self.N], reachable_set_approx + self.min_dists[i])
                        # reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2+tube_f[2]**2))
                        # obstacle[self.N] = (traj[:3, self.N], reachable_set_approx + self.min_dists[i])
                    else:
                        raise NotImplementedError
                    
                    dynamic_obstacles.append(obstacle)
        elif self.models[idx].nc == 2: # unicycle
            # for i in range(idx):
            for i in range(self.N_agent):
                if i != idx:
                    obstacle = np.empty((self.N+1), dtype=object)
                    traj = self.solutions["nominal_trajs"][i]
                    tube = self.solutions["tubes"][i]
                    tube_f = self.solutions["tubes_f"][i]
                    outer_approx = self.solutions["outer_approxes"][i]
                    nc = self.models[i].nc
                    if nc == 2: # unicycle
                        for t in range(self.N):
                            # obstacle[t] = (traj[:2, t], np.max(tube[t,:2]) + self.min_dists[i]) # inflate other agents by their radius
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                        # obstacle[self.N] = (traj[:2, self.N], np.max(self.solutions["tubes_f"][i][:2]) + self.min_dists[i])
                        reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.N] = (traj[:2, self.N], reachable_set_approx + self.min_dists[i])
                    elif nc == 3: # quadcopter (maybe can be omitted when flying high enough)
                        # continue
                        if np.all(traj[2,:] >= 0.15):
                            print("omit")
                            continue
                        # print(traj[2,:])
                        for t in range(self.N):
                            # obstacle[t] = (traj[:2, t], np.max(tube[t,:2]) + self.min_dists[i]) # inflate other agents by their radius
                            reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                            obstacle[t] = (traj[:2, t], reachable_set_approx + self.min_dists[i])
                        # obstacle[self.N] = (traj[:2, self.N], np.max(self.solutions["tubes_f"][i][:2]) + self.min_dists[i])
                        reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
                        obstacle[self.N] = (traj[:2, self.N], reachable_set_approx + self.min_dists[i])
                    else:
                        raise NotImplementedError
                    
                    dynamic_obstacles.append(obstacle)
        else:
            raise NotImplementedError

        return dynamic_obstacles
    
    def generate_leaders(self, idx):
        leaders = []
        
        for i in self.LOS_targets[idx]:
            target = np.empty((self.N+1), dtype=object)
            traj = self.solutions["nominal_trajs"][i]
            tube = self.solutions["tubes"][i]
            tube_f = self.solutions["tubes_f"][i]
            outer_approx = self.solutions["outer_approxes"][i]
            nc = self.models[i].nc
            for t in range(self.N):
                # target[t] = (traj[:2,t], np.max(tube[t,:2]) + self.min_dists[i])
                # target[t] = (traj[:2,t], np.sqrt(tube[t,0]**2+tube[t,1]**2)) #+ self.min_dists[i])
                reachable_set_approx = min(outer_approx[t,0], np.sqrt(tube[t,0]**2+tube[t,1]**2))
                target[t] = (traj[:2,t], reachable_set_approx + self.min_dists[i])
            # target[self.N] = (traj[:2, self.N], np.max(self.solutions["tubes_f"][i][:2]))
            reachable_set_approx = min(outer_approx[self.N, 0], np.sqrt(tube_f[0]**2+tube_f[1]**2))
            target[self.N] = (traj[:2, self.N], reachable_set_approx + self.min_dists[i])
            leaders.append(target)
        # print(LOS_targets)
        return leaders
    
    def generate_followers(self, idx):
        followers = []

        for i in self.followers[idx]:
            target = np.empty((self.N+1), dtype=object)
            traj = self.solutions["nominal_trajs"][i]
            tube = self.solutions["tubes"][i]
            tube_f = self.solutions["tubes_f"][i]
            outer_approx = self.solutions["outer_approxes"][i]
            nc = self.models[i].nc
            for t in range(self.N):
                # target[t] = (traj[:2,t], np.max(tube[t,:2]) + self.min_dists[i])
                # target[t] = (traj[:2,t], np.sqrt(tube[t,0]**2+tube[t,1]**2)) #+ self.min_dists[i])
                target[t] = (traj[:2,t], traj[2,t], tube[t,:3], outer_approx[t,0], self.min_dists[i], self.half_cones[i])
            # target[self.N] = (traj[:2, self.N], np.max(self.solutions["tubes_f"][i][:2]))
            target[self.N] = (traj[:2, self.N], traj[2,self.N], tube_f[:3], outer_approx[self.N,0], self.min_dists[i], self.half_cones[i])
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
                # obstacles = self.static_obstacles + dynamic_obstacles
                leaders = self.generate_leaders(idx)
                followers = self.generate_followers(idx)

                sol = self.plan(idx, self.static_obstacles, dynamic_obstacles, leaders, followers, it, init_guess)
                if not sol["success"]:
                    print("IBR failed at iteration {:} for agent {:}".format(it, idx))
                    return self.solutions, False
                
                convergence[idx] = np.all(np.fabs(sol["primal_x"] - self.solutions["nominal_trajs"][idx]) <= self.CONV_THRES)
                print(np.max(np.fabs(sol["primal_x"] - self.solutions["nominal_trajs"][idx])))

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

        print("IBR failed to converge")
        print("forward: {:.3f}".format(forward_runtime))
        print("backward: {:.3f}".format(backward_runtime))
        print("tightening: {:.3f}".format(tighten_runtime))
        return self.solutions, False
