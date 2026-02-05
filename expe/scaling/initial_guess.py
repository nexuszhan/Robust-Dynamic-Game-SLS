import time, copy
import numpy as np

from solver.centralized_init import NLP
from solver.IBR import IBR

def generate_initial_guess(planner:IBR, initial_guess_file:str):
    print("start initialize solutions")
    start = time.perf_counter()

    if planner.N_agent > 8 and planner.N_agent % 8 == 0:
        # print("break to several 8-agent teams")
        offsets = [0.1, 0.1, 0.25, 0.8] 
        for it in range(int(planner.N_agent / 8)):
            Q_all = planner.Q_all[it*8:(it+1)*8]
            R_all = planner.R_all[it*8:(it+1)*8]
            LTV_models = planner.LTV_models[it*8:(it+1)*8]
            Qf_all = planner.Qf_all[it*8:(it+1)*8]
            init_states = planner.init_states[it*8:(it+1)*8]
            goals = planner.goals[it*8:(it+1)*8]
            min_dists = np.array(planner.min_dists[it*8:(it+1)*8]) + offsets[it]

            solver = NLP(planner.T, Q_all, R_all, LTV_models, Qf_all, init_states, goals, planner.static_obstacles, 8, min_dists)
        
            solution = solver.solve(init_states)

            idx_start = 0 
            idx_end = 0
            for n in range(it*8, (it+1)*8):
                m = planner.models[n]
                idx_end += (m.nx+m.nu)*planner.T + m.nx
                planner.solutions["nominal_vecs"][n] = solution["primal_vec"][idx_start:idx_end]
                planner.solutions["nominal_trajs"][n] = copy.deepcopy(solution["primal_x_all"][n-it*8])
                planner.solutions["nominal_inputs"][n] = copy.deepcopy(solution["primal_u_all"][n-it*8])
                planner.solutions["initial_trajs"][n] = copy.deepcopy(solution["primal_x_all"][n-it*8])
                planner.solutions["initial_inputs"][n] = copy.deepcopy(solution["primal_u_all"][n-it*8])
                idx_start = idx_end
                
                planner.solutions["tubes"][n] = np.zeros((planner.T, m.nx)) + 0.05
                planner.solutions["tubes_f"][n] = np.zeros((m.nx)) + 0.05
                planner.solutions["initial_tubes"][n] = np.zeros((planner.T, m.nx)) + 0.05
                planner.solutions["outer_approxes"][n] = np.zeros((planner.T+1, m.nx)) + 0.075
                planner.solutions["initial_outer_approxes"][n] = np.zeros((planner.T+1, m.nx)) + 0.075
    else:
        min_dists = np.array(planner.min_dists) 
        solver = NLP(planner.T, planner.Q_all, planner.R_all, planner.LTV_models, planner.Qf_all, planner.init_states, planner.goals, planner.static_obstacles, planner.N_agent, min_dists)
        
        solution = solver.solve(planner.init_states)

        planner.solutions["nominal_trajs"] = copy.deepcopy(solution["primal_x_all"])
        planner.solutions["nominal_inputs"] = copy.deepcopy(solution["primal_u_all"])
        idx_start = 0 
        idx_end = 0
        for n in range(planner.N_agent):
            m = planner.models[n]
            idx_end += (m.nx+m.nu)*planner.T + m.nx
            planner.solutions["nominal_vecs"][n] = solution["primal_vec"][idx_start:idx_end]
            planner.solutions["initial_nominal_vecs"][n] = copy.deepcopy(solution["primal_vec"][idx_start:idx_end])
            idx_start = idx_end

            planner.solutions["tubes"][n] = np.zeros((planner.T, m.nx)) 
            planner.solutions["tubes_f"][n] = np.zeros((m.nx)) 
            planner.solutions["initial_tubes"][n] = np.zeros((planner.T, m.nx)) 
            planner.solutions["outer_approxes"][n] = np.zeros((planner.T+1, m.nx)) 
            planner.solutions["initial_outer_approxes"][n] = np.zeros((planner.T+1, m.nx)) 
        planner.solutions["initial_trajs"] = copy.deepcopy(solution["primal_x_all"])
        planner.solutions["initial_inputs"] = copy.deepcopy(solution["primal_u_all"])
        
    end = time.perf_counter()
    print("end initialize solutions")
    print("initialization time: {:.5f}".format(end-start))
    
    save_dict = {}
    save_dict[f"initial_trajs"] = np.array(planner.solutions["nominal_trajs"])
    save_dict[f"initial_inputs"] = np.array(planner.solutions["nominal_inputs"])
    save_dict[f"initial_vecs"] = np.array(planner.solutions["nominal_vecs"])
    save_dict[f"initial_tubes"] = np.array(planner.solutions["tubes"])
    save_dict[f"initial_tubes_f"] = np.array(planner.solutions["tubes_f"])
    save_dict[f"initial_outer_approxes"] = np.array(planner.solutions["outer_approxes"])
    np.savez(initial_guess_file, **save_dict)
    