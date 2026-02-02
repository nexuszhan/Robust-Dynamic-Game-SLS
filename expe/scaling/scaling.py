import matplotlib.pyplot as plt
import numpy as np
import time

from dyn.unicycle import Unicycle
from solver.IBR import IBR
from init_config import ScalingConfig4, ScalingConfig8, ScalingConfig16, ScalingConfig24
from util.plot import plot_traj

if __name__ == "__main__":
    def E_func(X):
        e = np.eye(4) * 2e-3

        return e

    N_agent = 4 #24 #16 #8 #4 

    if N_agent == 4:
        config = ScalingConfig4()
    elif N_agent == 8:
        config = ScalingConfig8()
    elif N_agent == 16:
        config = ScalingConfig16()
    elif N_agent == 24:
        config= ScalingConfig24()
    else:
        raise NotImplementedError
    
    init_states = config.init_states
    goals = config.goals

    models = [Unicycle(E_func) for _ in range(N_agent)]

    T = 80 # horizon length
    
    max_delta_goal = 0.1
    for i, model in enumerate(models):
        model.dt = 0.1 

        x_max = config.x_max
        x_min = config.x_min
        u_max = config.u_max
        u_min = config.u_min
        
        model.g = np.concatenate((x_max, u_max, -x_min, -u_min))

    static_obst = []

    agent_rad = 0.05
    min_dists = [agent_rad for _ in range(N_agent)]
    max_dists = [1.] * N_agent

    half_cones = [np.pi for _ in range(N_agent)]

    m = Unicycle()
    Q = 2 * np.eye(m.nx)
    R = np.eye(m.nu)
    Q_all = [Q for _ in range(N_agent)] 
    R_all = [R for _ in range(N_agent)]
    Qf = 10 * np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,0],
                        [0,0,0,1]])
    Qf_all = [Qf for _ in range(N_agent)] 

    Q_reg_all = [np.eye(m.nx)*5e4 for _ in range(N_agent)]
    R_reg_all = [np.eye(m.nu)*5e4 for _ in range(N_agent)]
    Qf_reg_all = [np.eye(m.nx)*5e4 for _ in range(N_agent)]

    LOS_targets = [[] for _ in range(N_agent)]

    use_LQR = [False for _ in range(N_agent)]

    init_guess = None
    
    if N_agent <= 8:
        planner = IBR(T, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all, 
                                N_agent, models, init_states, goals, static_obst, max_dists, min_dists, half_cones, LOS_targets, 
                                config.ca_weight, config.prox_weight, use_LQR, init_guess, config.init_file, 0.1)
    else:
        planner = IBR(T, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all, 
                                N_agent, models, init_states, goals, static_obst, max_dists, min_dists, half_cones, LOS_targets, 
                                config.ca_weight, config.prox_weight, use_LQR, init_guess, None, 0.5)
    
    solutions = planner.solutions
    # save_dict = {}
    # save_dict[f"initial_trajs"] = np.array(solutions["nominal_trajs"])
    # save_dict[f"initial_inputs"] = np.array(solutions["nominal_inputs"])
    # save_dict[f"initial_vecs"] = np.array(solutions["nominal_vecs"])
    # save_dict[f"initial_tubes"]          = np.array(solutions["tubes"])
    # save_dict[f"initial_tubes_f"]        = np.array(solutions["tubes_f"])
    # save_dict[f"initial_outer_approxes"]  = np.array(solutions["outer_approxes"])
    # np.savez(f"scaling_{N_agent}_init.npz", **save_dict)
    
    plot_traj(solutions["initial_trajs"], solutions["initial_tubes"], [np.zeros(m.nx) for _ in range(N_agent)], solutions["initial_outer_approxes"], models, [], N_agent, init_states, goals, T, agent_rad)
    plt.savefig(f"scaling_{N_agent}_init.png", format="png")

    start = time.perf_counter()
    solutions, success = planner.plan_all()
    end = time.perf_counter()
    print("Runtime: {:.5f}".format(end-start))

    plot_traj(solutions["nominal_trajs"], solutions["tubes"], solutions["tubes_f"], solutions["outer_approxes"], models, [], N_agent, init_states, goals, T, agent_rad)
    plt.savefig(f"scaling_{N_agent}.png", format="png")

    save_dict = {}
    for i in range(N_agent):  
        idx = i + 1    
        save_dict[f"nominal_traj{idx}"]  = np.array(solutions["nominal_trajs"][i])
        save_dict[f"nominal_input{idx}"] = np.array(solutions["nominal_inputs"][i])
        save_dict[f"Phi_x{idx}"]         = np.array(solutions["phi_x_mats"][i])
        save_dict[f"Phi_u{idx}"]         = np.array(solutions["phi_u_mats"][i])
        save_dict[f"tube{idx}"]          = np.array(solutions["tubes"][i])
        save_dict[f"tube_f{idx}"]        = np.array(solutions["tubes_f"][i])
        save_dict[f"outer_approx{idx}"]  = np.array(solutions["outer_approxes"][i])

    np.savez(f"scaling_{N_agent}.npz", **save_dict)