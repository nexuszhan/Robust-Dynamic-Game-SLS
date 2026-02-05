import time, copy
import numpy as np

from solver.centralized_init import NLP
from solver.IBR_heter import IBR
from solver.fast_SLS_nlp import fast_SLS

def generate_initial_guess(planner:IBR, initial_guess_file:str):
    print("start initialize solutions")
    start = time.perf_counter()

    # initialize quadcopter in centralized way
    lead_models = [planner.LTV_models[i] for i in range(0, planner.N_agent, 3)]
    lead_init_states = [planner.init_states[i] for i in range(0, planner.N_agent, 3)]
    lead_goals = [planner.goals[i] for i in range(0, planner.N_agent, 3)]
    lead_min_dists = [planner.min_dists[i]+0.15 for i in range(0, planner.N_agent ,3)]
    solver = NLP(planner.T, [planner.Q_all[0]]*int(planner.N_agent/3), [planner.R_all[0]]*int(planner.N_agent/3), lead_models, [planner.Qf_all[0]]*int(planner.N_agent/3), lead_init_states, lead_goals, planner.static_obstacles, int(planner.N_agent/3), lead_min_dists)
    
    solution = solver.solve(lead_init_states)
    
    idx_start = 0 
    idx_end = 0
    for idx in range(int(planner.N_agent/3)):
        planner.solutions["nominal_trajs"][3*idx] = copy.deepcopy(solution["primal_x_all"][idx])
        planner.solutions["nominal_inputs"][3*idx] = copy.deepcopy(solution["primal_u_all"][idx])
        
        m = planner.models[3*idx]
        idx_end += (m.nx+m.nu)*planner.T + m.nx
        planner.solutions["nominal_vecs"][3*idx] = solution["primal_vec"][idx_start:idx_end]
        idx_start = idx_end

        planner.solutions["tubes"][3*idx] = np.zeros((planner.T, m.nx))
        planner.solutions["tubes_f"][3*idx] = np.zeros((m.nx))
        planner.solutions["outer_approxes"][3*idx] = np.zeros((planner.T+1, m.nx))
        planner.solutions["initial_tubes"][3*idx] = np.zeros((planner.T, m.nx))
        planner.solutions["initial_tubes_f"][3*idx] = np.zeros((m.nx))
        planner.solutions["initial_outer_approxes"][3*idx] = np.zeros((planner.T+1, m.nx))
        planner.solutions["initial_trajs"][3*idx] = copy.deepcopy(solution["primal_x_all"][idx])
        planner.solutions["initial_inputs"][3*idx] = copy.deepcopy(solution["primal_u_all"][idx])
    
    for idx in range(planner.N_agent):
        if idx % 3 != 0:
            x0 = planner.init_states[idx]
            goal = planner.goals[idx]
            Q = planner.Q_all[idx]
            R = planner.R_all[idx]
            Qf = planner.Qf_all[idx]
            
            Q_reg = planner.Q_reg_all[idx]
            R_reg = planner.R_reg_all[idx]
            Qf_reg = planner.Qf_reg_all[idx]
            
            # Let followers follow to initialize
            solver = fast_SLS(planner.T, Q, R, planner.LTV_models[idx], Qf, 
                            len(planner.static_obstacles), 0, len(planner.LOS_targets[idx]), 0, planner.models[idx],
                            Q_reg, R_reg, Qf_reg)
            solver.verbose = True

            leaders = planner.generate_leaders(idx)
            followers = planner.generate_followers(idx)
            half_cones = [np.pi/6 for _ in range(planner.N_agent)]
            
            solution = solver.solve(x0, goal, planner.static_obstacles, [], planner.max_dists[idx], planner.min_dists[idx], leaders, [], half_cones[idx], 0., 0., planner.use_LQR[idx])

            planner.solutions["nominal_trajs"][idx] = solution["primal_x"]
            planner.solutions["nominal_inputs"][idx] = solution["primal_u"]
            planner.solutions["nominal_vecs"][idx] = solution["primal_vec"]
            planner.solutions["tubes"][idx] = solution["backoff"]
            planner.solutions["tubes_f"][idx] = solution["backoff_f"]
            planner.solutions["outer_approxes"][idx] = solution["outer_approx"]
            planner.solutions["K_mats"][idx] = solution["K_mat"].copy()
            planner.solutions["phi_x_mats"][idx] = solution["Phi_x_mat"].copy()
            planner.solutions["phi_u_mats"][idx] = solution["Phi_u_mat"].copy()
            planner.solutions["initial_trajs"][idx] = solution["primal_x"]
            planner.solutions["initial_tubes"][idx] = solution["backoff"]
            planner.solutions["initial_tubes_f"][idx] = solution["backoff_f"]
            planner.solutions["initial_outer_approxes"][idx] = solution["outer_approx"]
            planner.solutions["initial_inputs"][idx] = solution["primal_u"]
            planner.solutions["initial_phi_x_mats"][idx] = solution["Phi_x_mat"]
            planner.solutions["initial_phi_u_mats"][idx] = solution["Phi_u_mat"]
            planner.solutions["costs"][idx] = solution["cost"]
            planner.solutions["cost_nominals"][idx] = solution["cost_nominal"]
        
    end = time.perf_counter()
    print("end initialize solutions")
    print("initialization time: {:.5f}".format(end-start))

    save_dict = {}
    save_dict[f"initial_trajs"] = np.empty(planner.N_agent, dtype=object)
    save_dict[f"initial_inputs"] = np.empty(planner.N_agent, dtype=object)
    save_dict[f"initial_vecs"] = np.empty(planner.N_agent, dtype=object)
    save_dict[f"initial_tubes"] = np.empty(planner.N_agent, dtype=object)
    save_dict[f"initial_tubes_f"] = np.empty(planner.N_agent, dtype=object)
    save_dict[f"initial_outer_approxes"] = np.empty(planner.N_agent, dtype=object)
    for i in range(planner.N_agent):
        save_dict[f"initial_trajs"][i] = np.asarray(planner.solutions["nominal_trajs"][i])
        save_dict[f"initial_inputs"][i] = np.asarray(planner.solutions["nominal_inputs"][i])
        save_dict[f"initial_vecs"][i] = np.asarray(planner.solutions["nominal_vecs"][i])
        save_dict[f"initial_tubes"][i] = np.asarray(planner.solutions["tubes"][i])
        save_dict[f"initial_tubes_f"][i] = np.asarray(planner.solutions["tubes_f"][i])
        save_dict[f"initial_outer_approxes"][i] = np.asarray(planner.solutions["outer_approxes"][i])
    np.savez(initial_guess_file, **save_dict)