import copy, time, os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from dyn.quadrotor import Quadrotor
from dyn.unicycle import Unicycle
from solver.IBR_heter import IBR
from util.plot import plot_reachable_set
from initial_guess import generate_initial_guess

def save_solutions(path, solutions, n_agent):
    save_dict = {}
    for i in range(n_agent):
        idx = i + 1
        save_dict[f"nominal_traj{idx}"] = np.array(solutions["nominal_trajs"][i])
        save_dict[f"nominal_input{idx}"] = np.array(solutions["nominal_inputs"][i])
        save_dict[f"Phi_x{idx}"] = np.array(solutions["phi_x_mats"][i])
        save_dict[f"Phi_u{idx}"] = np.array(solutions["phi_u_mats"][i])
        save_dict[f"tube{idx}"] = np.array(solutions["tubes"][i])
        save_dict[f"tube_f{idx}"] = np.array(solutions["tubes_f"][i])
        save_dict[f"outer_approx{idx}"] = np.array(solutions["outer_approxes"][i])
    np.savez(path, **save_dict)

def plot_traj(trajs, tubes, tubes_f, approxes, models, static_obst, N_agent, init_states, goals, T, min_dists):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    colors = ['red', 'orange', 'yellow', 'blue', 'navy', 'cyan']
    
    for obst in static_obst:
        ellipsoid = patches.Ellipse(tuple(obst[0]), 2*obst[1], 2*obst[1], angle=0, facecolor='k', edgecolor='k')
        ax.add_patch(ellipsoid)
    
    for i in range(0, N_agent, 3):
        n = i
        m = models[n]
        x_plan = trajs[n][0,:]
        y_plan = trajs[n][1,:]
        # theta_plan = trajs[n][2,:]
        agent_rad = min_dists[n]

        ax.plot(x_plan, y_plan, markersize=1, c=colors[n])
        ax.plot(init_states[n][0], init_states[n][1], 'x', c=colors[n])
        ax.plot(goals[n][0], goals[n][1], '*', c=colors[n])
        tube = tubes[n]
        approx = approxes[n]
        for t in range(1, T):
            plot_reachable_set((x_plan[t], y_plan[t]), agent_rad, tube[t,0], tube[t,1], approx[t,0], colors[n], ax=ax)
        tube_f = tubes_f[n]
        plot_reachable_set((x_plan[T], y_plan[T]), agent_rad, tube_f[0], tube_f[1], approx[T,0], colors[n], ax=ax)
        
        n = i+1
        m = models[n]
        x_plan = trajs[n][0,:]
        y_plan = trajs[n][1,:]
        theta_plan = trajs[n][2,:]
        agent_rad = min_dists[n]

        ax.plot(x_plan, y_plan, markersize=1, c=colors[n])
        ax.plot(init_states[n][0], init_states[n][1], 'x', c=colors[n])
        ax.plot(goals[n][0], goals[n][1], '*', c=colors[n])
        ax.quiver(x_plan, y_plan, np.cos(theta_plan), np.sin(theta_plan), color=colors[n], width=0.002, scale=20)

        tube = tubes[n]
        approx = approxes[n]
        for t in range(1, T):
            plot_reachable_set((x_plan[t], y_plan[t]), agent_rad, tube[t,0], tube[t,1], approx[t,0], colors[n], ax=ax)
        tube_f = tubes_f[n]
        plot_reachable_set((x_plan[T], y_plan[T]), agent_rad, tube_f[0], tube_f[1], approx[T,0], colors[n], ax=ax)

        n = i+2
        m = models[n]
        x_plan = trajs[n][0,:]
        y_plan = trajs[n][1,:]
        theta_plan = trajs[n][2,:]
        agent_rad = min_dists[n]

        ax.plot(x_plan, y_plan, markersize=1, c=colors[n])
        ax.plot(init_states[n][0], init_states[n][1], 'x', c=colors[n])
        ax.plot(goals[n][0], goals[n][1], '*', c=colors[n])
        ax.quiver(x_plan, y_plan, np.cos(theta_plan), np.sin(theta_plan), color=colors[n], width=0.002, scale=20)

        tube = tubes[n]
        approx = approxes[n]
        for t in range(1, T):
            plot_reachable_set((x_plan[t], y_plan[t]), agent_rad, tube[t,0], tube[t,1], approx[t,0], colors[n], ax=ax)
        tube_f = tubes_f[n]
        plot_reachable_set((x_plan[T], y_plan[T]), agent_rad, tube_f[0], tube_f[1], approx[T,0], colors[n], ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("auto")
    ax.set_xlim(left=0., right=3.2)
    ax.set_ylim(bottom=0., top=2.)
    ax.set_xbound(lower=0., upper=3.2)
    ax.set_ybound(lower=0., upper=2)
    ax.grid(True)
    ax.axis("equal")

if __name__ == "__main__":
    N_agent = 6
    T = 80

    init_states = [
        [-0.5, -0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75, -0.75, np.pi / 4, 0.0],
        [-1.1, -0.75, 0.0, 0.0],
        [0.5, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.75, 0.75, -3 * np.pi / 4, 0.0],
        [1.1, 0.75, np.pi, 0.0],
    ]
    goals = [
        [0.5, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [-0.5, -0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-0.5, -0.5, 0.0, 0.0],
        [-0.5, -0.5, 0.0, 0.0],
    ]

    m_leader = Quadrotor()
    m_leader.dt = 0.1
    e = np.eye(m_leader.nx) * 1e-5
    e[0, 0] = 4e-3
    e[1, 1] = 4e-3
    e[2, 2] = 4e-3
    m_leader.E_func = lambda X: e
    x_max = np.array([100.0, 100.0, 2.0, 0.5, 0.5, 100.0, 0.22, 0.22, 0.1, 5.0, 5.0, 5.0])
    x_min = np.array([-100.0, -100.0, 0.05, -0.5, -0.5, -100.0, -0.22, -0.22, -0.1, -5.0, -5.0, -5.0])
    u_max = np.array([2 * 9.81, 0.2, 0.2, 0.2])
    u_min = np.array([0.0, -0.2, -0.2, -0.2])
    m_leader.g = np.concatenate((x_max, u_max, -x_min, -u_min))
    m_leader.gf = np.concatenate((x_max, -x_min))

    m_follower = Unicycle()
    m_follower.dt = 0.1
    e_follower = np.eye(m_follower.nx) * 5e-4
    m_follower.E_func = lambda X: e_follower
    x_max = np.array([100.0, 100.0, 100.0, 0.5])
    x_min = np.array([-100.0, -100.0, -100.0, -0.5])
    u_max = np.array([np.pi, 1.0])
    u_min = np.array([-np.pi, -1.0])
    m_follower.g = np.concatenate((x_max, u_max, -x_min, -u_min))

    models = [None for _ in range(N_agent)]
    for i in range(0, N_agent, 3):
        models[i] = copy.deepcopy(m_leader)
        models[i + 1] = copy.deepcopy(m_follower)
        models[i + 2] = copy.deepcopy(m_follower)

    static_obst = []

    n_teams = N_agent // 3
    min_dists = [0.05, 0.06, 0.06] * n_teams
    max_dists = [1.0, 0.5, 0.5] * n_teams
    half_cones = [np.pi, np.pi / 4, np.pi / 4] * n_teams

    Q_leader = 2 * np.eye(m_leader.nx)
    Q_leader[2, 2] = 5
    Q_leader[3:6, :] = 0
    R_leader = np.eye(m_leader.nu)
    Qf_leader = 10 * np.eye(m_leader.nx)
    Qf_leader[3:6, :] = 0

    Q_follower = 2 * np.eye(m_follower.nx)
    R_follower = np.eye(m_follower.nu)
    Qf_follower = np.zeros((m_follower.nx, m_follower.nx))

    Q_all = [Q_leader, Q_follower, Q_follower] * n_teams
    R_all = [R_leader, R_follower, R_follower] * n_teams
    Qf_all = [Qf_leader, Qf_follower, Qf_follower] * n_teams

    Q_reg_all = [np.eye(m.nx) * 50000 for m in models]
    R_reg_all = [np.eye(m.nu) * 50000 for m in models]
    Qf_reg_all = [np.eye(m.nx) * 50000 for m in models]

    LOS_targets = []
    followers = []
    for i in range(0, N_agent, 3):
        LOS_targets.extend([[], [i], [i + 1]])
        followers.extend([[i + 1], [], []])

    use_LQR = [False] * N_agent

    ca_weight = 0.0075 #0.001
    prox_weight = 0.001 #0.0005 
    init_guess_file = "heter_team_init.npz"

    planner = IBR(T, Q_all, R_all, Qf_all, Q_reg_all, R_reg_all, Qf_reg_all,
                  N_agent, models, init_states, goals, static_obst,
                  max_dists, min_dists, half_cones, LOS_targets, followers,
                  ca_weight, prox_weight, use_LQR, 0.6)
    
    if not os.path.isfile(init_guess_file):
        generate_initial_guess(planner, init_guess_file)

    planner.initialize_solutions(init_guess_file)
    solutions = planner.solutions

    plot_traj(solutions["initial_trajs"], solutions["initial_tubes"],
              solutions["initial_tubes_f"], solutions["initial_outer_approxes"],
              models, static_obst, N_agent, init_states, goals, T, min_dists)
    plt.savefig("heter_team_init.png", format="png")

    start = time.perf_counter()
    solutions, success = planner.plan_all()
    end = time.perf_counter()
    print(f"Runtime: {end - start:.5f}")

    plot_traj(solutions["nominal_trajs"], solutions["tubes"],
              solutions["tubes_f"], solutions["outer_approxes"],
              models, static_obst, N_agent, init_states, goals, T, min_dists)
    plt.savefig("heter_team.png", format="png")

    save_solutions("heter_team.npz", solutions, N_agent)
