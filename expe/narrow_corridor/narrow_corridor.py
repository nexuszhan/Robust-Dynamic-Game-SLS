import matplotlib.pyplot as plt
import numpy as np
import time, os

from dyn.unicycle import Unicycle
from solver.IBR import IBR
from util.plot import plot_traj, check_feedback
from initial_guess import generate_initial_guess

if __name__ == "__main__":
    def E_func(X):
        e = np.eye(4) * 2e-3
        return e
    
    N_agent = 2

    models = [Unicycle(E_func) for _ in range(N_agent)]
    m = Unicycle(E_func)

    T = 60 # horizon length

    init_states = np.array([[-2., 0.1, 0., 0.],
                            [2., -0.1, np.pi, 0.]])
    goals = np.array([[2.5, 0.1, 0., 0.],
                    [-2.5, -0.1, np.pi, 0.]])

    max_delta_goal = 1. 
    for i, model in enumerate(models):
        model.dt = 0.1 
        x_max = np.array([100, 100, 10., 2.2]) # x, y, theta, v
        x_min = np.array([-100, -100, -10., -1.])
        u_max = np.array([np.pi/2, 1.]) # angular_vel, linear acceleration
        u_min = np.array([-np.pi/2, -1.])

        model.g = np.concatenate((x_max, u_max, -x_min, -u_min))

    static_obst = [(np.array([0., 0.5]), 0.3), (np.array([0., -0.6]), 0.4)]

    agent_rad = 0.1
    min_dists = [agent_rad for _ in range(N_agent)]
    max_dists = [1.] * N_agent

    half_cones = [np.pi, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6, np.pi/6]

    Q = 2 * np.eye(m.nx)
    Q_ego = 2 * np.eye(m.nx)
    Q_ego[2,2] = 0.
    R = np.eye(m.nu)
    Q_all = [Q_ego, Q_ego] 
    R_all = [R for _ in range(N_agent)]
    Qf = 5 * np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,0],
                        [0,0,0,1]])
    Qf_all = [Qf, Qf] 

    Q_reg_all = [Q*5e4, Q*5e4]
    R_reg_all = [R*5e4, R*5e4]
    Qf_reg_all = [Qf*5e4, Qf*5e4]

    LOS_targets = [[], []]
    followers = [[], []]

    use_LQR = [True, True]
    ca_weight = 0.001
    prox_weight = 0.1 # no effect

    init_guess_file = "narrow_corridor_init.npz"

    planner = IBR(T, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all, 
                        N_agent, models, init_states, goals, static_obst, max_dists, min_dists, half_cones, LOS_targets, 
                        ca_weight, prox_weight, use_LQR, 0.3)
    
    if not os.path.isfile(init_guess_file):
        generate_initial_guess(planner, init_guess_file)
    
    planner.initialize_solutions(init_guess_file)
    solutions = planner.solutions
    
    plot_traj(solutions["initial_trajs"], solutions["initial_tubes"], [np.zeros(m.nx) for _ in range(N_agent)], solutions["initial_outer_approxes"], models, static_obst, N_agent, init_states, goals, T, agent_rad)
    plt.savefig("narrow_corridor_init.png", format="png")
    
    start = time.perf_counter()
    solutions, success = planner.plan_all()
    end = time.perf_counter()
    print("Runtime: {:.5f}".format(end-start))

    plot_traj(solutions["nominal_trajs"], solutions["tubes"], solutions["tubes_f"], solutions["outer_approxes"], models, static_obst, N_agent, init_states, goals, T, agent_rad)
    plt.savefig("narrow_corridor.png", format="png")

    for idx in range(N_agent):
        phi_x = solutions["phi_x_mats"][idx]
        phi_u = solutions["phi_u_mats"][idx]
        primal_x = solutions["nominal_trajs"][idx]
        primal_u = solutions["nominal_inputs"][idx]
        tube = solutions["tubes"][idx]
        tube_f = solutions["tubes_f"][idx]
        check_feedback(phi_u @ np.linalg.inv(phi_x), primal_x, primal_u, tube, tube_f, init_states[idx], models[idx], T)
        plt.savefig(f"narrow_corridor_rollout_{idx}.png", format="png")

    np.savez("narrow_corridor.npz", 
         nominal_traj1=np.array(solutions["nominal_trajs"][0]),
         nominal_input1=np.array(solutions["nominal_inputs"][0]),
         Phi_x1=np.array(solutions["phi_x_mats"][0]),
         Phi_u1=np.array(solutions["phi_u_mats"][0]),
         tube1=np.array(solutions["tubes"][0]),
         tube_f1=np.array(solutions["tubes_f"][0]),
         outer_approx1=np.array(solutions["outer_approxes"][0]),

         nominal_traj2=np.array(solutions["nominal_trajs"][1]),
         nominal_input2=np.array(solutions["nominal_inputs"][1]),
         Phi_x2=np.array(solutions["phi_x_mats"][1]),
         Phi_u2=np.array(solutions["phi_u_mats"][1]),
         tube2=np.array(solutions["tubes"][1]),
         tube_f2=np.array(solutions["tubes_f"][1]),
         outer_approx2=np.array(solutions["outer_approxes"][1]),
         )
