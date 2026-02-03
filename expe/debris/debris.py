import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from scipy.stats import multivariate_normal

from dyn.unicycle import Unicycle
from solver.IBR import IBR
from util.plot import check_feedback, plot_traj

if __name__ == "__main__":
    def E_func(X):
        e = np.exp(-0.5 * (((X[0]) ** 2 + (X[1]) ** 2) / 0.02)) / (2 * np.pi * 0.02)
        return np.eye(4) * e * 0.002


    def plot_disturbance_map(ax):
        x_min, x_max = -1.0, 1.0
        y_min, y_max = -1.0, 1.0
        nx, ny = 300, 300

        rv = multivariate_normal(mean=np.array([0.0, 0.0]), cov=np.array([[0.025, 0], [0, 0.025]]))

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        grid = np.dstack((X, Y))
        Z = rv.pdf(grid) * 0.002

        return ax.contourf(X, Y, Z, levels=10, cmap="Greys", zorder=0)

    N_agent = 4
    T = 80

    models = [Unicycle(E_func) for _ in range(N_agent)]

    init_states = np.array(
        [
            [-0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, -np.pi / 2, 0.0],
            [0.5, 0.0, np.pi, 0.0],
            [0.0, -0.5, np.pi / 2, 0.0],
        ]
    )
    goals = np.array(
        [
            [0.5, 0.0, 0.0, 0.0],
            [0.0, -0.5, -np.pi / 2, 0.0],
            [-0.5, 0.0, np.pi, 0.0],
            [0.0, 0.5, np.pi / 2, 0.0],
        ]
    )

    for model in models:
        model.dt = 0.1
        x_max = np.array([5.0, 5.0, 10.0, 0.2])
        x_min = np.array([-5.0, -5.0, -10.0, -0.1])
        u_max = np.array([np.pi / 2, 0.1])
        u_min = np.array([-np.pi / 2, -0.5])
        model.g = np.concatenate((x_max, u_max, -x_min, -u_min))

    static_obst = []

    agent_rad = 0.06
    min_dists = [agent_rad for _ in range(N_agent)]
    max_dists = [1.0 for _ in range(N_agent)]

    half_cones = [np.pi, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6]

    m = Unicycle()
    Q = 2 * np.eye(m.nx)
    R = np.eye(m.nu)
    Q_all = [Q for _ in range(N_agent)]
    R_all = [R for _ in range(N_agent)]
    Qf = 10 * np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    Qf_all = [Qf for _ in range(N_agent)]

    Q_reg_all = [Q * 100 for _ in range(N_agent)]
    R_reg_all = [R * 100 for _ in range(N_agent)]
    Qf_reg_all = [Qf * 100 for _ in range(N_agent)]

    LOS_targets = [[] for _ in range(N_agent)]
    use_LQR = [False for _ in range(N_agent)]

    ca_weight = 0.001
    prox_weight = 0.1 # no effect
    init_file = "debris_init.npz"

    planner = IBR(
        T,
        Q_all,
        R_all,
        Qf_all,
        Q_reg_all,
        R_reg_all,
        Qf_reg_all,
        N_agent,
        models,
        init_states,
        goals,
        static_obst,
        max_dists,
        min_dists,
        half_cones,
        LOS_targets,
        ca_weight,
        prox_weight,
        use_LQR,
        init_file,
        1.0,
    )

    solutions = planner.solutions
    # save_dict = {}
    # save_dict[f"initial_trajs"] = np.array(solutions["nominal_trajs"])
    # save_dict[f"initial_inputs"] = np.array(solutions["nominal_inputs"])
    # save_dict[f"initial_vecs"] = np.array(solutions["nominal_vecs"])
    # save_dict[f"initial_tubes"]          = np.array(solutions["tubes"])
    # save_dict[f"initial_tubes_f"]        = np.array(solutions["tubes_f"])
    # save_dict[f"initial_outer_approxes"]  = np.array(solutions["outer_approxes"])
    # np.savez(f"debris_init.npz", **save_dict)
    ax = plot_traj(
        solutions["initial_trajs"],
        solutions["initial_tubes"],
        [np.zeros(m.nx) for _ in range(N_agent)],
        solutions["initial_outer_approxes"],
        models,
        [],
        N_agent,
        init_states,
        goals,
        T,
        agent_rad,
    )
    cf = plot_disturbance_map(ax)
    plt.colorbar(cf)
    plt.savefig("debris_init.png", format="png")

    solutions, success = planner.plan_all()
    ax = plot_traj(
        solutions["nominal_trajs"],
        solutions["tubes"],
        solutions["tubes_f"],
        solutions["outer_approxes"],
        models,
        [],
        N_agent,
        init_states,
        goals,
        T,
        agent_rad,
    )
    cf = plot_disturbance_map(ax)
    plt.colorbar(cf)
    plt.savefig("debris.png", format="png")

    for idx in range(N_agent):
        phi_x = solutions["phi_x_mats"][idx]
        phi_u = solutions["phi_u_mats"][idx]
        primal_x = solutions["nominal_trajs"][idx]
        primal_u = solutions["nominal_inputs"][idx]
        tube = solutions["tubes"][idx]
        tube_f = solutions["tubes_f"][idx]
        check_feedback(phi_u @ np.linalg.inv(phi_x), primal_x, primal_u, tube, tube_f, init_states[idx], models[idx], T)

    np.savez(
        "debris.npz",
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
        nominal_traj3=np.array(solutions["nominal_trajs"][2]),
        nominal_input3=np.array(solutions["nominal_inputs"][2]),
        Phi_x3=np.array(solutions["phi_x_mats"][2]),
        Phi_u3=np.array(solutions["phi_u_mats"][2]),
        tube3=np.array(solutions["tubes"][2]),
        tube_f3=np.array(solutions["tubes_f"][2]),
        outer_approx3=np.array(solutions["outer_approxes"][2]),
        nominal_traj4=np.array(solutions["nominal_trajs"][3]),
        nominal_input4=np.array(solutions["nominal_inputs"][3]),
        Phi_x4=np.array(solutions["phi_x_mats"][3]),
        Phi_u4=np.array(solutions["phi_u_mats"][3]),
        tube4=np.array(solutions["tubes"][3]),
        tube_f4=np.array(solutions["tubes_f"][3]),
        outer_approx4=np.array(solutions["outer_approxes"][3]),
    )
