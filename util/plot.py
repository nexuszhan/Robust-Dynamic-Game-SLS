from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_reachable_set(center, rad, dx, dy, approx, color, alpha=0.8, ax=None):
    set_rad = rad + min(approx, np.sqrt(dx**2+dy**2))
    circle = patches.Ellipse((center[0], center[1]), 2*set_rad, 2*set_rad, angle=0, facecolor=color, edgecolor=color, alpha=alpha)
    ax.add_patch(circle)

def plot_traj(trajs, tubes, tubes_f, approxes, models, static_obst, N_agent, init_states, goals, T, agent_rad):
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 6)

    # colors = ['r', 'b', 'g', 'c', 'y', 'grey', 'm', 'navy', 'orange', 'brown']
    colors = [
        'brown', 'red', 'orange', 'olive', 'yellow', 'm',
        'lightgreen', 'green', 'darkseagreen', 'cyan', 'blue', 'navy',
        
        'deeppink', 'coral', 'gold', 'lime', 'springgreen',
        'turquoise', 'teal', 'deepskyblue', 'slateblue',
        'indigo', 'purple', 'gray',

        'brown', 'red', 'orange', 'olive', 'yellow', 'm',
        'lightgreen', 'green', 'darkseagreen', 'cyan', 'blue', 'navy',
        
        'deeppink', 'coral', 'gold', 'lime', 'springgreen',
        'turquoise', 'teal', 'deepskyblue', 'slateblue',
        'indigo', 'purple', 'gray'
    ]
    
    for obst in static_obst:
        ellipsoid = patches.Ellipse(tuple(obst[0]), 2*obst[1], 2*obst[1], angle=0, facecolor='k', edgecolor='k')
        ax.add_patch(ellipsoid)

    for n in range(N_agent):
        m = models[n]
        x_plan = trajs[n][0,:]
        y_plan = trajs[n][1,:]
        theta_plan = trajs[n][2,:]

        ax.plot(x_plan, y_plan, markersize=1, c=colors[n])
        ax.plot(init_states[n][0], init_states[n][1], 'x', c=colors[n])
        ax.plot(goals[n][0], goals[n][1], '*', c=colors[n])
        ax.quiver(x_plan, y_plan, np.cos(theta_plan), np.sin(theta_plan), color=colors[n], width=0.002, scale=20)

        tube = tubes[n]
        approx = approxes[n]
        for t in range(1, T):
            plot_reachable_set((x_plan[t], y_plan[t]), agent_rad, tube[t,0], tube[t,1], approx[t,0], colors[n], ax=ax)
        tube_f = tubes_f[n]
        plot_reachable_set((x_plan[T], y_plan[T]), agent_rad, tube_f[0], tube_f[1], approx[t,0], colors[n], ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    ax.axis("equal")

    return ax

def check_feedback(K_mat, primal_x, primal_u, tube, tube_f, x0, m, T):
    fig, axes = plt.subplots(m.nx, 1, figsize=(10, 6))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    np.random.seed(0)
    
    # tube and nominal
    for idx in range(m.nx):
        for t in range(T):
            axes[idx].plot([t-0.1, t+0.1], [primal_x[idx,t]+tube[t,idx], primal_x[idx,t]+tube[t,idx]], 'k-', linewidth=1)
            axes[idx].plot([t-0.1, t+0.1], [primal_x[idx,t]-tube[t,idx], primal_x[idx,t]-tube[t,idx]], 'k-', linewidth=1)
        axes[idx].plot([T-0.1, T+0.1], [primal_x[idx,T]+tube_f[idx], primal_x[idx,T]+tube_f[idx]], 'k-', linewidth=1)
        axes[idx].plot([T-0.1, T+0.1], [primal_x[idx,T]-tube_f[idx], primal_x[idx,T]-tube_f[idx]], 'k-', linewidth=1)

        axes[idx].plot(np.arange(T+1), primal_x[idx,:], label="nominal")
        axes[idx].legend()
    
    # rollouts
    for rollout in range(10):
        states = np.zeros((m.nx, T+1))
        states[:,0] = x0
        disturbance = np.zeros((m.nx, T))
        
        if rollout < 5:
            for i in range(T):
                disturbance[:,i] = (np.random.random(m.nx)-0.5) 
                disturbance[:,i] = np.random.random() * disturbance[:,i]/np.linalg.norm(disturbance[:,i])
        else:
            for i in range(T):
                disturbance[:,i] = (np.random.random(m.nx)-0.5) 
                disturbance[:,i] = disturbance[:,i]/np.linalg.norm(disturbance[:,i])
        
        for t in range(1, T+1):
            delta_u = np.zeros((m.nu))
            for j in range(t):
                delta_u += K_mat[(t-1)*m.nu:t*m.nu, j*m.nx:(j+1)*m.nx] @ (states[:,j]-primal_x[:,j])
            u = primal_u[:,t-1]+delta_u
            states[:,t] = np.array(m.ddyn(states[:,t-1], u, m.dt)).squeeze() + m.E_func(states[:,t-1]) @ disturbance[:,t-1]
        
        for idx in range(m.nx):
            axes[idx].plot(np.arange(T+1), states[idx,:], c='r', linewidth=1)
    
    return axes

def plot_ground_robot(pose, rad, alpha=0.8, ax=None):
    # robot_length = rad*np.sqrt(2)*0.9
    # robot_width = rad*np.sqrt(2)*0.9
    robot_length = rad*np.sqrt(2)*0.8
    robot_width = rad*np.sqrt(2)*0.8

    fwd = np.array([np.cos(pose[2]), np.sin(pose[2])])      # forward
    lft = np.array([-np.sin(pose[2]), np.cos(pose[2])])     # left

    low_left = pose[:2] - (robot_length/2)*fwd - (robot_width/2)*lft
    # p = patches.Rectangle((pose[:2] + robot_length/2*np.array((np.cos(pose[2]+np.pi/2), np.sin(pose[2]+np.pi/2)))+\
    #                     0.04*np.array((-np.sin(pose[2]+np.pi/2), np.cos(pose[2]+np.pi/2)))), robot_length, robot_width, angle=(pose[2] + np.pi/4) * 180/np.pi, facecolor='#FFD700', edgecolor='k')
    # p = patches.Rectangle((pose[:2] + robot_length/2*np.array((np.cos(pose[2]+np.pi/2), np.sin(pose[2]+np.pi/2)))+\
    #                     0.04*np.array((-np.sin(pose[2]+np.pi/2), np.cos(pose[2]+np.pi/2)))), robot_length, robot_width, angle=pose[2], facecolor='#FFD700', edgecolor='k')
    p = patches.Rectangle(
        low_left, robot_length, robot_width,
        angle=np.degrees(pose[2]),
        # facecolor='#FFD700', edgecolor='k', alpha=alpha
        facecolor="grey", edgecolor='k', alpha=alpha
    )

    # rled = patches.Circle(pose[:2]+0.75*robot_length/2*np.array((np.cos(pose[2]), np.sin(pose[2]))+0.04*np.array((-np.sin(pose[2]+np.pi/2), np.cos(pose[2]+np.pi/2)))),
    #                     robot_length/2/5, fill=False)
    # lled = patches.Circle(pose[:2]+0.75*robot_length/2*np.array((np.cos(pose[2]), np.sin(pose[2]))+\
    #                         0.015*np.array((-np.sin(pose[2]+np.pi/2), np.cos(pose[2]+np.pi/2)))),\
    #                         robot_length/2/5, fill=False)
    # rw = patches.Circle(pose[:2]+robot_length/2*np.array((np.cos(pose[2]+np.pi/2), np.sin(pose[2]+np.pi/2)))+\
    #                                 0.04*np.array((-np.sin(pose[2]+np.pi/2), np.cos(pose[2]+np.pi/2))),\
    #                                 0.02, facecolor='k')
    # lw = patches.Circle(pose[:2]+robot_length/2*np.array((np.cos(pose[2]-np.pi/2), np.sin(pose[2]-np.pi/2)))+\
    #                                 0.04*np.array((-np.sin(pose[2]+np.pi/2))),\
    #                                 0.02, facecolor='k')
    # ---- “Headlights” (or front markers), near front edge with slight lateral offsets ----
    front_dist   = 0.75 * (robot_length/2)          # how far forward from center
    # lat_off_r    = 0.04                  # right marker offset (meters)
    # lat_off_l    = 0.04               # left marker offset (meters)
    lat_off_r    = 0.02                  # right marker offset (meters)
    lat_off_l    = 0.02               # left marker offset (meters)
    led_radius   = (robot_length/2)/5               # your original scale

    # Right = negative left-direction; Left = positive left-direction
    rled_center = pose[:2] + front_dist*fwd - lat_off_r*lft
    lled_center = pose[:2] + front_dist*fwd + lat_off_l*lft
    rled = patches.Circle(rled_center, led_radius, fill=False, edgecolor='k', alpha=alpha)
    lled = patches.Circle(lled_center, led_radius, fill=False, edgecolor='k', alpha=alpha)
    
    # wheel
    # axle_fwd = -0.07    # small forward shift
    # wheel_r  = 0.02
    axle_fwd = -0.03    # small forward shift
    wheel_r  = 0.01

    rw_center = pose[:2] + axle_fwd*fwd - (robot_width/2)*lft   # right wheel
    lw_center = pose[:2] + axle_fwd*fwd + (robot_width/2)*lft   # left wheel
    rw = patches.Circle(rw_center, wheel_r, facecolor="grey", edgecolor='k', alpha=alpha)
    lw = patches.Circle(lw_center, wheel_r, facecolor="grey", edgecolor='k', alpha=alpha)

    # self.chassis_patches.append(p)
    # self.left_led_patches.append(lled)
    # self.right_led_patches.append(rled)
    # self.right_wheel_patches.append(rw)
    # self.left_wheel_patches.append(lw)
    # self.base_patches.append(base)

    ax.add_patch(rw)
    ax.add_patch(lw)
    ax.add_patch(p)
    ax.add_patch(lled)
    ax.add_patch(rled)
