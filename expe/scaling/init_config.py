from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class ScalingConfig4:
    init_states: np.ndarray = np.array(
        [
            [-5.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, np.pi, 0.0],
            [0.0, 5.0, -np.pi / 2, 0.0],
            [0.0, -5.0, np.pi / 2, 0.0],
        ]
    )
    goals: np.ndarray = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],
            [-5.0, 0.0, np.pi, 0.0],
            [0.0, -5.0, -np.pi / 2, 0.0],
            [0.0, 5.0, np.pi / 2, 0.0],
        ]
    )
    x_max: np.ndarray = np.array([100., 100., 100., 1.5]) 
    x_min: np.ndarray = np.array([-100., -100., -100., -1.5])
    u_max: np.ndarray = np.array([np.pi/2, 8.]) 
    u_min: np.ndarray = np.array([-np.pi/2, -8.])
    ca_weight: float = 0.0075
    prox_weight: float = 0.1 # no effect
    init_file: str = "scaling_4_init.npz"

@dataclass(frozen=True)
class ScalingConfig8:
    init_states: np.ndarray = np.array(
        [
            [-5.0, 0.0, 0.0, 0.0],
            [5.0, 0.0, np.pi, 0.0],
            [0.0, 5.0, -np.pi / 2, 0.0],
            [0.0, -5.0, np.pi / 2, 0.0],
            [2.5 * 1.41, 2.5 * 1.41, -3 * np.pi / 4, 0.0],
            [2.5 * 1.41, -2.5 * 1.41, 3 * np.pi / 4, 0.0],
            [-2.5 * 1.41, -2.5 * 1.41, np.pi / 4, 0.0],
            [-2.5 * 1.41, 2.5 * 1.41, -np.pi / 4, 0.0],
        ]
    )
    goals: np.ndarray = np.array(
        [
            [5.0, 0.0, 0.0, 0.0],
            [-5.0, 0.0, np.pi, 0.0],
            [0.0, -5.0, -np.pi / 2, 0.0],
            [0.0, 5.0, np.pi / 2, 0.0],
            [-2.5 * 1.41, -2.5 * 1.41, -3 * np.pi / 4, 0.0],
            [-2.5 * 1.41, 2.5 * 1.41, 3 * np.pi / 4, 0.0],
            [2.5 * 1.41, 2.5 * 1.41, np.pi / 4, 0.0],
            [2.5 * 1.41, -2.5 * 1.41, -np.pi / 4, 0.0],
        ]
    )
    x_max: np.ndarray = np.array([100., 100., 100., 3.]) 
    x_min: np.ndarray = np.array([-100., -100., -100., -3.])
    u_max: np.ndarray = np.array([np.pi/2, 3.]) 
    u_min: np.ndarray = np.array([-np.pi/2, -3.])
    ca_weight: float = 0.001
    prox_weight: float = 0.1 # no effect
    init_file: str = "scaling_8_init.npz"

@dataclass(frozen=True)
class ScalingConfig16:
    base = 10.
    cos45 = np.cos(np.pi/4)
    sin45 = np.sin(np.pi/4)
    cos22 = np.cos(np.pi/8)
    sin22 = np.sin(np.pi/8)
    cos67 = np.cos(3*np.pi/8)
    sin67 = np.sin(3*np.pi/8)
    init_states: np.ndarray = np.array([[-base, 0., 0., 0.],
                            [base, 0., np.pi, 0.],
                            [0., base, -np.pi/2, 0.],
                            [0., -base, np.pi/2, 0.],
                            [base*cos45, base*sin45, -3*np.pi/4, 0.],
                            [base*cos45, -base*sin45, 3*np.pi/4, 0.],
                            [-base*cos45, -base*sin45, np.pi/4, 0.],
                            [-base*cos45, base*sin45, -np.pi/4, 0.],
                            [base*cos22, base*sin22, -7*np.pi/8, 0.],
                            [base*cos67, base*sin67, -5*np.pi/8, 0.],
                            [-base*cos67, base*sin67, -3*np.pi/8, 0.],
                            [-base*cos22, base*sin22, -np.pi/8, 0.],
                            [-base*cos22, -base*sin22, np.pi/8, 0.],
                            [-base*cos67, -base*sin67, 3*np.pi/8, 0.],
                            [base*cos67, -base*sin67, 5*np.pi/8, 0.],
                            [base*cos22, -base*sin22, 7*np.pi/8, 0.]])
    goals: np.ndarray = np.array([[base, 0., 0., 0.],
                    [-base, 0, np.pi, 0.],
                    [0., -base, -np.pi/2, 0.],
                    [0., base, np.pi/2, 0.],
                    [-base*cos45, -base*sin45, -3*np.pi/4, 0.],
                    [-base*cos45, base*sin45, 3*np.pi/4, 0.],
                    [base*cos45, base*sin45, np.pi/4, 0.],
                    [base*cos45, -base*sin45, -np.pi/4, 0.],
                    [-base*cos22, -base*sin22, -7*np.pi/8, 0.],
                    [-base*cos67, -base*sin67, -5*np.pi/8, 0.],
                    [base*cos67, -base*sin67, -3*np.pi/8, 0.],
                    [base*cos22, -base*sin22, -np.pi/8, 0.],
                    [base*cos22, base*sin22, np.pi/8, 0.],
                    [base*cos67, base*sin67, 3*np.pi/8, 0.],
                    [-base*cos67, base*sin67, 5*np.pi/8, 0.],
                    [-base*cos22, base*sin22, 7*np.pi/8, 0.]])
    x_max: np.ndarray = np.array([100., 100., 100., 5.]) 
    x_min: np.ndarray = np.array([-100., -100., -100., -5.])
    u_max: np.ndarray = np.array([np.pi/2, 5.]) 
    u_min: np.ndarray = np.array([-np.pi/2, -5.])
    ca_weight: float = 0.0001
    prox_weight: float = 0.1 # no effect
    init_file: str = "scaling_16_init.npz"
    
@dataclass(frozen=True)
class ScalingConfig24:
    base = 15.0
    dalpha = np.pi / 12.0  # 11.25 degrees
    
    init_states = []
    goals = []
    for i in range(3):
        for j in range(8):
            angle = (j*np.pi/4 + i*dalpha)
            init_states.append([base*np.cos(angle), base*np.sin(angle), angle, 0.])
            goals.append([base*np.cos(np.pi+angle), base*np.sin(np.pi+angle), angle, 0.])
    init_states: np.ndarray = np.array(init_states)
    goals: np.ndarray = np.array(goals)
    x_max: np.ndarray = np.array([100., 100., 100., 5.]) 
    x_min: np.ndarray = np.array([-100., -100., -100., -5.])
    u_max: np.ndarray = np.array([np.pi/2, 5.]) 
    u_min: np.ndarray = np.array([-np.pi/2, -5.])
    ca_weight: float = 0.0001
    prox_weight: float = 0.1 # no effect
    init_file: str = "scaling_24_init.npz"